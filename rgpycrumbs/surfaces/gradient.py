import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import jax.scipy.optimize as jopt
import numpy as np
from jax import jit, vmap

from rgpycrumbs.surfaces._base import BaseGradientSurface, generic_negative_mll
from rgpycrumbs.surfaces._kernels import (
    imq_kernel_elem,
    k_matrix_imq_grad_map,
    k_matrix_matern_grad_map,
    k_matrix_rq_grad_map,
    k_matrix_se_grad_map,
    matern_kernel_elem,
    rq_kernel_elem,
    se_kernel_elem,
)

# ==============================================================================
# GRADIENT-ENHANCED MATERN HELPERS
# ==============================================================================


def negative_mll_matern_grad(log_params, x, y_flat, D_plus_1):
    length_scale = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K_blocks = k_matrix_matern_grad_map(x, x, length_scale)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    return generic_negative_mll(K_full, y_flat, noise_scalar)


@jit
def _grad_matern_solve(x, y_full, noise_scalar, length_scale):
    K_blocks = k_matrix_matern_grad_map(x, x, length_scale)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_matern_predict(x_query, x_obs, alpha, length_scale):
    def get_query_row(xq, xo):
        kee = matern_kernel_elem(xq, xo, length_scale)
        ked = jax.grad(matern_kernel_elem, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_matern_var(x_query, x_obs, K_inv, length_scale):
    def get_query_row(xq, xo):
        kee = matern_kernel_elem(xq, xo, length_scale)
        ked = jax.grad(matern_kernel_elem, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)
    var = 1.0 - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientMatern(BaseGradientSurface):
    """Gradient-enhanced Matérn 5/2 surface implementation."""

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
            init_ls = jnp.mean(span) * 0.5
        else:
            init_ls = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_matern_grad(log_p, self.x, self.y_flat, self.D_plus_1)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.ls = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))
            if jnp.isnan(self.ls) or jnp.isnan(self.noise):
                self.ls, self.noise = init_ls, init_noise
        else:
            self.ls, self.noise = init_ls, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _grad_matern_solve(
            self.x, self.y_full, self.noise, self.ls
        )

    def _predict_chunk(self, chunk):
        return _grad_matern_predict(chunk, self.x, self.alpha, self.ls)

    def _var_chunk(self, chunk):
        return _grad_matern_var(chunk, self.x, self.K_inv, self.ls)


# ==============================================================================
# GRADIENT-ENHANCED SE HELPERS
# ==============================================================================


def negative_mll_se_grad(log_params, x, y_flat, D_plus_1):
    length_scale = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K_blocks = k_matrix_se_grad_map(x, x, length_scale)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    return generic_negative_mll(K_full, y_flat, noise_scalar)


@jit
def _grad_se_solve(x, y_full, noise_scalar, length_scale):
    K_blocks = k_matrix_se_grad_map(x, x, length_scale)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_se_predict(x_query, x_obs, alpha, length_scale):
    def get_query_row(xq, xo):
        kee = se_kernel_elem(xq, xo, length_scale)
        ked = jax.grad(se_kernel_elem, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_se_var(x_query, x_obs, K_inv, length_scale):
    def get_query_row(xq, xo):
        kee = se_kernel_elem(xq, xo, length_scale)
        ked = jax.grad(se_kernel_elem, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)
    var = 1.0 - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientSE(BaseGradientSurface):
    """Gradient-enhanced Squared Exponential (SE) surface implementation."""

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
            init_ls = jnp.mean(span) * 0.4
        else:
            init_ls = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_se_grad(log_p, self.x, self.y_flat, self.D_plus_1)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.ls = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))
            if jnp.isnan(self.ls) or jnp.isnan(self.noise):
                self.ls, self.noise = init_ls, init_noise
        else:
            self.ls, self.noise = init_ls, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _grad_se_solve(self.x, self.y_full, self.noise, self.ls)

    def _predict_chunk(self, chunk):
        return _grad_se_predict(chunk, self.x, self.alpha, self.ls)

    def _var_chunk(self, chunk):
        return _grad_se_var(chunk, self.x, self.K_inv, self.ls)


# ==============================================================================
# GRADIENT-ENHANCED IMQ HELPERS
# ==============================================================================


def negative_mll_imq_grad(log_params, x, y_flat, D_plus_1):
    epsilon = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    return generic_negative_mll(K_full, y_flat, noise_scalar)


def negative_mll_imq_map(log_params, init_eps, x, y_flat, D_plus_1):
    log_eps = log_params[0]
    log_noise = log_params[1]
    epsilon = jnp.exp(log_eps)
    noise_scalar = jnp.exp(log_noise)

    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    mll_cost = generic_negative_mll(K_full, y_flat, noise_scalar)

    alpha_g = 2.0
    beta_g = 1.0 / (init_eps + 1e-6)
    eps_penalty = -(alpha_g - 1.0) * log_eps + beta_g * epsilon

    noise_target = jnp.log(1e-2)
    noise_penalty = (log_noise - noise_target) ** 2 / 0.5
    return mll_cost + eps_penalty + noise_penalty


@jit
def _grad_imq_solve(x, y_full, noise_scalar, epsilon):
    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_imq_predict(x_query, x_obs, alpha, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_imq_var(x_query, x_obs, K_inv, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)
    var = (1.0 / epsilon) - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientIMQ(BaseGradientSurface):
    """Gradient-enhanced Inverse Multi-Quadratic (IMQ) surface implementation."""

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
            init_eps = jnp.mean(span) * 0.8
        else:
            init_eps = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_eps), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_imq_map(
                    log_p, init_eps, self.x, self.y_flat, self.D_plus_1
                )

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.epsilon = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))
            if jnp.isnan(self.epsilon) or jnp.isnan(self.noise):
                self.epsilon, self.noise = init_eps, init_noise
        else:
            self.epsilon, self.noise = init_eps, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _grad_imq_solve(
            self.x, self.y_full, self.noise, self.epsilon
        )

    def _predict_chunk(self, chunk):
        return _grad_imq_predict(chunk, self.x, self.alpha, self.epsilon)

    def _var_chunk(self, chunk):
        return _grad_imq_var(chunk, self.x, self.K_inv, self.epsilon)


# ==============================================================================
# GRADIENT-ENHANCED RQ HELPERS
# ==============================================================================


def negative_mll_rq_map(log_params, x, y_flat, D_plus_1):
    log_ls = log_params[0]
    log_alpha = log_params[1]
    log_noise = log_params[2]
    length_scale = jnp.exp(log_ls)
    alpha = jnp.exp(log_alpha)
    noise_scalar = jnp.exp(log_noise)

    params = jnp.array([length_scale, alpha])
    K_blocks = k_matrix_rq_grad_map(x, x, params)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    mll_cost = generic_negative_mll(K_full, y_flat, noise_scalar)

    ls_target = jnp.log(1.5)
    ls_penalty = (log_ls - ls_target) ** 2 / 0.05
    noise_target = jnp.log(1e-2)
    noise_penalty = (log_noise - noise_target) ** 2 / 1.0
    alpha_target = jnp.log(0.8)
    alpha_penalty = (log_alpha - alpha_target) ** 2 / 0.5
    return mll_cost + ls_penalty + noise_penalty + alpha_penalty


@jit
def _grad_rq_solve(x, y_full, noise_scalar, params):
    K_blocks = k_matrix_rq_grad_map(x, x, params)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_rq_predict(x_query, x_obs, alpha, params):
    def get_query_row(xq, xo):
        kee = rq_kernel_elem(xq, xo, params)
        ked = jax.grad(rq_kernel_elem, argnums=1)(xq, xo, params)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_rq_var(x_query, x_obs, K_inv, params):
    def get_query_row(xq, xo):
        kee = rq_kernel_elem(xq, xo, params)
        ked = jax.grad(rq_kernel_elem, argnums=1)(xq, xo, params)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)

    def self_var(xq):
        return rq_kernel_elem(xq, xq, params)

    base_var = vmap(self_var)(x_query)
    var = base_var - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientRQ(BaseGradientSurface):
    """Symmetric Gradient-enhanced Rational Quadratic (RQ) surface implementation."""

    def _fit(self, _smoothing, length_scale, optimize):
        init_ls = length_scale if length_scale is not None else 1.5
        init_alpha = 1.0
        init_noise = 1e-2

        if optimize:
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_alpha), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_rq_map(log_p, self.x, self.y_flat, self.D_plus_1)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.ls = float(jnp.exp(results.x[0]))
            self.alpha_param = float(jnp.exp(results.x[1]))
            self.noise = float(jnp.exp(results.x[2]))

            if jnp.isnan(self.ls) or jnp.isnan(self.noise):
                self.ls, self.alpha_param, self.noise = init_ls, init_alpha, init_noise
        else:
            self.ls, self.alpha_param, self.noise = init_ls, init_alpha, init_noise
        self.params = jnp.array([self.ls, self.alpha_param])

    def _solve(self):
        self.alpha, self.K_inv = _grad_rq_solve(
            self.x, self.y_full, self.noise, self.params
        )

    def _predict_chunk(self, chunk):
        return _grad_rq_predict(chunk, self.x, self.alpha, self.params)

    def _var_chunk(self, chunk):
        return _grad_rq_var(chunk, self.x, self.K_inv, self.params)


# ==============================================================================
# NYSTRÖM GRADIENT-ENHANCED IMQ HELPERS
# ==============================================================================


@jit
def _stable_nystrom_grad_imq_solve(x, y_full, x_inducing, noise_scalar, epsilon):
    N = x.shape[0]
    M = x_inducing.shape[0]
    D_plus_1 = x.shape[1] + 1
    K_mm_blocks = k_matrix_imq_grad_map(x_inducing, x_inducing, epsilon)
    K_mm = K_mm_blocks.transpose(0, 2, 1, 3).reshape(M * D_plus_1, M * D_plus_1)
    jitter = (noise_scalar + 1e-4) * jnp.eye(M * D_plus_1)
    K_mm = K_mm + jitter
    L = jnp.linalg.cholesky(K_mm)
    K_nm_blocks = k_matrix_imq_grad_map(x, x_inducing, epsilon)
    K_nm = K_nm_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, M * D_plus_1)
    K_mn = K_nm.T
    V = jlinalg.solve_triangular(L, K_mn, lower=True)
    sigma2 = noise_scalar + 1e-6
    S = V @ V.T + sigma2 * jnp.eye(M * D_plus_1)
    L_S = jnp.linalg.cholesky(S)
    Vy = V @ y_full.flatten()
    beta = jlinalg.cho_solve((L_S, True), Vy)
    alpha_m = jlinalg.solve_triangular(L.T, beta, lower=False)
    I_M = jnp.eye(M * D_plus_1)
    S_inv = jlinalg.cho_solve((L_S, True), I_M)
    inner = I_M - sigma2 * S_inv
    L_inv = jlinalg.solve_triangular(L, I_M, lower=True)
    W = L_inv.T @ inner @ L_inv
    return alpha_m, W


@jit
def _nystrom_grad_imq_predict(x_query, x_inducing, alpha_m, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_qm = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_inducing)
    Q, M, D_plus_1 = K_qm.shape
    return K_qm.reshape(Q, M * D_plus_1) @ alpha_m


@jit
def _nystrom_grad_imq_var(x_query, x_inducing, W, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_qm = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_inducing)
    Q, M, D_plus_1 = K_qm.shape
    K_qm_flat = K_qm.reshape(Q, M * D_plus_1)
    var = (1.0 / epsilon) - jnp.sum((K_qm_flat @ W) * K_qm_flat, axis=1)
    return jnp.maximum(var, 0.0)


class NystromGradientIMQ(BaseGradientSurface):
    """Memory-efficient Nystrom-approximated gradient-enhanced IMQ surface."""

    def __init__(
        self,
        x,
        y,
        gradients=None,
        n_inducing=300,
        nimags=None,
        smoothing=1e-3,
        length_scale=None,
        optimize=True,
        **kwargs,
    ):
        """
        Initializes the Nystrom-approximated model.

        Args:
            x: Training inputs.
            y: Training values.
            gradients: Training gradients.
            n_inducing: Number of inducing points to sample.
            nimags: Path-based image count for structured sampling.
            smoothing: Noise level.
            length_scale: Initial epsilon for IMQ.
            optimize: Optimization toggle.
        """
        self.n_inducing = n_inducing
        self.nimags = nimags
        super().__init__(x, y, gradients, smoothing, length_scale, optimize, **kwargs)

    def _fit(self, smoothing, length_scale, _optimize):
        N_total = self.x.shape[0]
        if N_total <= self.n_inducing:
            self.x_inducing = self.x
            self.y_full_inducing = self.y_full
        else:
            rng = np.random.RandomState(42)
            if self.nimags is not None and self.nimags > 0:
                n_paths = N_total // self.nimags
                paths_to_sample = max(1, self.n_inducing // self.nimags)
                if n_paths > 1:
                    start_idx = max(0, n_paths - paths_to_sample)
                    path_indices = np.arange(start_idx, n_paths)
                else:
                    path_indices = np.array([0])
                idx = np.concatenate(
                    [
                        np.arange(p * self.nimags, (p + 1) * self.nimags)
                        for p in path_indices
                    ]
                )
                idx = idx[idx < N_total]
                self.x_inducing = self.x[idx]
                self.y_full_inducing = self.y_full[idx]
            else:
                idx = rng.choice(N_total, min(self.n_inducing, N_total), replace=False)
                self.x_inducing = self.x[idx]
                self.y_full_inducing = self.y_full[idx]
        self.epsilon = length_scale if length_scale is not None else 0.5
        self.noise = smoothing

    def _solve(self):
        self.alpha_m, self.W = _stable_nystrom_grad_imq_solve(
            self.x, self.y_full, self.x_inducing, self.noise, self.epsilon
        )

    def _predict_chunk(self, chunk):
        return _nystrom_grad_imq_predict(
            chunk, self.x_inducing, self.alpha_m, self.epsilon
        )

    def _var_chunk(self, chunk):
        return _nystrom_grad_imq_var(chunk, self.x_inducing, self.W, self.epsilon)
