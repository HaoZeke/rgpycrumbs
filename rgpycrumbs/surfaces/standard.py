import jax.numpy as jnp
import jax.scipy.optimize as jopt
from jax import jit

from rgpycrumbs.surfaces._base import BaseSurface, generic_negative_mll
from rgpycrumbs.surfaces._kernels import (
    _imq_kernel_matrix,
    _matern_kernel_matrix,
    _tps_kernel_matrix,
)

# ==============================================================================
# TPS HELPERS
# ==============================================================================


def negative_mll_tps(log_params, x, y):
    # TPS only really has a smoothing parameter to tune in this context
    # (Length scale is inherent to the radial basis).
    smoothing = jnp.exp(log_params[0])
    K = _tps_kernel_matrix(x)
    return generic_negative_mll(K, y, smoothing)


@jit
def _tps_solve(x, y, sm):
    K = _tps_kernel_matrix(x)
    K = K + jnp.eye(x.shape[0]) * sm

    # Polynomial Matrix
    N = x.shape[0]
    P = jnp.concatenate([jnp.ones((N, 1), dtype=jnp.float32), x], axis=1)
    M = P.shape[1]

    # Solve System
    zeros = jnp.zeros((M, M), dtype=jnp.float32)
    top = jnp.concatenate([K, P], axis=1)
    bot = jnp.concatenate([P.T, zeros], axis=1)
    lhs = jnp.concatenate([top, bot], axis=0)
    rhs = jnp.concatenate([y, jnp.zeros(M, dtype=jnp.float32)])

    coeffs = jnp.linalg.solve(lhs, rhs)
    lhs_inv = jnp.linalg.inv(lhs)
    return coeffs[:N], coeffs[N:], lhs_inv


@jit
def _tps_predict(x_query, x_obs, w, v):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K_q = r**2 * jnp.log(r)

    P_q = jnp.concatenate(
        [jnp.ones((x_query.shape[0], 1), dtype=jnp.float32), x_query], axis=1
    )
    return K_q @ w + P_q @ v


@jit
def _tps_var(x_query, x_obs, lhs_inv):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K_q = r**2 * jnp.log(r)
    P_q = jnp.concatenate(
        [jnp.ones((x_query.shape[0], 1), dtype=jnp.float32), x_query], axis=1
    )
    KP_q = jnp.concatenate([K_q, P_q], axis=1)
    var = -jnp.sum((KP_q @ lhs_inv) * KP_q, axis=1)
    return jnp.maximum(var, 0.0)


class FastTPS:
    """
    Thin Plate Spline (TPS) surface implementation.
    Includes a polynomial mean function and supports smoothing optimization.
    """

    def __init__(self, x_obs, y_obs, smoothing=1e-3, optimize=True, **_kwargs):
        """
        Initializes the TPS model.

        Args:
            x_obs: Training inputs (N, D).
            y_obs: Training observations (N,).
            smoothing: Initial smoothing parameter.
            optimize: Whether to optimize the smoothing parameter.
        """
        self.x_obs = jnp.asarray(x_obs, dtype=jnp.float32)
        self.y_obs = jnp.asarray(y_obs, dtype=jnp.float32)

        # TPS handles mean via polynomial, but centering helps optimization stability
        self.y_mean = jnp.mean(self.y_obs)
        y_centered = self.y_obs - self.y_mean

        init_sm = max(smoothing, 1e-4)

        if optimize:
            # Optimize [log_smoothing]
            x0 = jnp.array([jnp.log(init_sm)])

            def loss_fn(log_p):
                return negative_mll_tps(log_p, self.x_obs, y_centered)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)

            self.sm = float(jnp.exp(results.x[0]))
            if jnp.isnan(self.sm):
                self.sm = init_sm
        else:
            self.sm = init_sm

        self.w, self.v, self.K_inv = _tps_solve(self.x_obs, self.y_obs, self.sm)

    def __call__(self, x_query, chunk_size=500):
        """
        Predict values at query points using chunking.

        Args:
            x_query: Query inputs (M, D).
            chunk_size: Processing batch size.

        Returns:
            jnp.ndarray: Predicted values (M,).
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        preds = []
        for i in range(0, x_query.shape[0], chunk_size):
            chunk = x_query[i : i + chunk_size]
            preds.append(_tps_predict(chunk, self.x_obs, self.w, self.v))
        return jnp.concatenate(preds, axis=0)

    def predict_var(self, x_query, chunk_size=500):
        """
        Predict posterior variance at query points.

        Args:
            x_query: Query inputs (M, D).
            chunk_size: Processing batch size.

        Returns:
            jnp.ndarray: Predicted variances (M,).
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        vars_list = []
        for i in range(0, x_query.shape[0], chunk_size):
            chunk = x_query[i : i + chunk_size]
            vars_list.append(_tps_var(chunk, self.x_obs, self.K_inv))
        return jnp.concatenate(vars_list, axis=0)


# ==============================================================================
# MATERN 5/2
# ==============================================================================


def negative_mll_matern_std(log_params, x, y):
    length_scale = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K = _matern_kernel_matrix(x, length_scale)
    return generic_negative_mll(K, y, noise_scalar)


@jit
def _matern_solve(x, y, sm, length_scale):
    K = _matern_kernel_matrix(x, length_scale)
    K = K + jnp.eye(x.shape[0]) * sm
    L = jnp.linalg.cholesky(K)
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))

    eye = jnp.eye(K.shape[0])
    L_inv = jnp.linalg.solve(L, eye)
    K_inv = L_inv.T @ L_inv
    return alpha, K_inv


@jit
def _matern_predict(x_query, x_obs, alpha, length_scale):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    K_q = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(-sqrt5_r_l)
    return K_q @ alpha


@jit
def _matern_var(x_query, x_obs, K_inv, length_scale):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    K_q = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(-sqrt5_r_l)
    var = 1.0 - jnp.sum((K_q @ K_inv) * K_q, axis=1)
    return jnp.maximum(var, 0.0)


class FastMatern(BaseSurface):
    """Matérn 5/2 surface implementation."""

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x_obs, axis=0) - jnp.min(self.x_obs, axis=0)
            init_ls = jnp.mean(span) * 0.2
        else:
            init_ls = length_scale

        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_matern_std(log_p, self.x_obs, self.y_centered)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.ls = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))

            if jnp.isnan(self.ls) or jnp.isnan(self.noise):
                self.ls, self.noise = init_ls, init_noise
        else:
            self.ls, self.noise = init_ls, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _matern_solve(
            self.x_obs, self.y_centered, self.noise, self.ls
        )

    def _predict_chunk(self, chunk):
        return _matern_predict(chunk, self.x_obs, self.alpha, self.ls)

    def _var_chunk(self, chunk):
        return _matern_var(chunk, self.x_obs, self.K_inv, self.ls)


# ==============================================================================
# STANDARD IMQ (Optimizable)
# ==============================================================================


def negative_mll_imq_std(log_params, x, y):
    epsilon = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K = _imq_kernel_matrix(x, epsilon)
    return generic_negative_mll(K, y, noise_scalar)


@jit
def _imq_solve(x, y, sm, epsilon):
    K = _imq_kernel_matrix(x, epsilon)
    K = K + jnp.eye(x.shape[0]) * sm
    L = jnp.linalg.cholesky(K)
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))

    eye = jnp.eye(K.shape[0])
    L_inv = jnp.linalg.solve(L, eye)
    K_inv = L_inv.T @ L_inv
    return alpha, K_inv


@jit
def _imq_predict(x_query, x_obs, alpha, epsilon):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    K_q = 1.0 / jnp.sqrt(d2 + epsilon**2)
    return K_q @ alpha


@jit
def _imq_var(x_query, x_obs, K_inv, epsilon):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    K_q = 1.0 / jnp.sqrt(d2 + epsilon**2)
    var = (1.0 / epsilon) - jnp.sum((K_q @ K_inv) * K_q, axis=1)
    return jnp.maximum(var, 0.0)


class FastIMQ(BaseSurface):
    """Inverse Multi-Quadratic (IMQ) surface implementation."""

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x_obs, axis=0) - jnp.min(self.x_obs, axis=0)
            init_eps = jnp.mean(span) * 0.8
        else:
            init_eps = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_eps), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_imq_std(log_p, self.x_obs, self.y_centered)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.epsilon = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))
            if jnp.isnan(self.epsilon) or jnp.isnan(self.noise):
                self.epsilon, self.noise = init_eps, init_noise
        else:
            self.epsilon, self.noise = init_eps, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _imq_solve(
            self.x_obs, self.y_centered, self.noise, self.epsilon
        )

    def _predict_chunk(self, chunk):
        return _imq_predict(chunk, self.x_obs, self.alpha, self.epsilon)

    def _var_chunk(self, chunk):
        return _imq_var(chunk, self.x_obs, self.K_inv, self.epsilon)
