import logging

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import jax.scipy.optimize as jopt
import numpy as np
from jax import jit, vmap

# Force float32 for speed/viz
jax.config.update("jax_enable_x64", False)

# ==============================================================================
# HELPER: GENERIC LOSS FUNCTIONS
# ==============================================================================


def safe_cholesky_solve(K, y, noise_scalar, jitter_steps=3):
    """
    Retries Cholesky decomposition with increasing jitter if it fails.

    Args:
        K: Covariance matrix.
        y: Observation vector.
        noise_scalar: Initial noise level.
        jitter_steps: Number of retry attempts with increasing jitter.

    Returns:
        tuple: (alpha, log_det) where alpha is the solution vector and
               log_det is the log determinant of the jittered matrix.
    """
    N = K.shape[0]

    # Try successively larger jitters: 1e-6, 1e-5, 1e-4
    for i in range(jitter_steps):
        jitter = (noise_scalar + 10 ** (-6 + i)) * jnp.eye(N)
        try:
            L = jnp.linalg.cholesky(K + jitter)
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            return alpha, log_det
        except Exception as e:
            logging.debug(f"Cholesky failed: {e}")
            continue

    # Fallback for compilation safety (NaN propagation)
    return jnp.zeros_like(y), jnp.nan


def generic_negative_mll(K, y, noise_scalar):
    """
    Calculates the negative Marginal Log-Likelihood (MLL).

    Args:
        K: Covariance matrix.
        y: Observation vector.
        noise_scalar: Noise level for regularization.

    Returns:
        float: The negative MLL value, or a high penalty if Cholesky fails.
    """
    alpha, log_det = safe_cholesky_solve(K, y, noise_scalar)

    data_fit = 0.5 * jnp.dot(y.flatten(), alpha.flatten())
    complexity = 0.5 * log_det

    cost = data_fit + complexity
    # heavy penalty if Cholesky failed (NaN)
    return jnp.where(jnp.isnan(cost), 1e9, cost)


# ==============================================================================
# BASE CLASSES
# ==============================================================================


class BaseSurface:
    """
    Abstract base class for standard (non-gradient) surface models.

    Derived classes must implement `_fit`, `_solve`, `_predict_chunk`, and `_var_chunk`.
    """

    def __init__(
        self, x_obs, y_obs, smoothing=1e-3, length_scale=None, optimize=True, **kwargs
    ):
        """
        Initializes and fits the surface model.

        Args:
            x_obs: Training inputs (N, D).
            y_obs: Training observations (N,).
            smoothing: Initial noise/smoothing parameter.
            length_scale: Initial length scale parameter(s).
            optimize: Whether to optimize parameters via MLE.
            **kwargs: Additional model-specific parameters.
        """
        self.x_obs = jnp.asarray(x_obs, dtype=jnp.float32)
        self.y_obs = jnp.asarray(y_obs, dtype=jnp.float32)
        # Center the data
        self.y_mean = jnp.mean(self.y_obs)
        self.y_centered = self.y_obs - self.y_mean

        self._fit(smoothing, length_scale, optimize)
        self._solve()

    def _fit(self, smoothing, length_scale, optimize):
        """Internal method to perform parameter optimization."""
        raise NotImplementedError

    def _solve(self):
        """Internal method to solve the linear system for weights."""
        raise NotImplementedError

    def __call__(self, x_query, chunk_size=500):
        """
        Predict values at query points.

        Args:
            x_query: Query inputs (M, D).
            chunk_size: Number of points to process per batch to avoid OOM.

        Returns:
            jnp.ndarray: Predicted values (M,).
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        preds = []
        for i in range(0, x_query.shape[0], chunk_size):
            chunk = x_query[i : i + chunk_size]
            preds.append(self._predict_chunk(chunk))
        return jnp.concatenate(preds, axis=0) + self.y_mean

    def predict_var(self, x_query, chunk_size=500):
        """
        Predict posterior variance at query points.

        Args:
            x_query: Query inputs (M, D).
            chunk_size: Number of points to process per batch.

        Returns:
            jnp.ndarray: Predicted variances (M,).
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        vars_list = []
        for i in range(0, x_query.shape[0], chunk_size):
            chunk = x_query[i : i + chunk_size]
            vars_list.append(self._var_chunk(chunk))
        return jnp.concatenate(vars_list, axis=0)

    def _predict_chunk(self, chunk):
        """Internal method for batch prediction."""
        raise NotImplementedError

    def _var_chunk(self, chunk):
        """Internal method for batch variance."""
        raise NotImplementedError


class BaseGradientSurface:
    """
    Abstract base class for gradient-enhanced surface models.

    Derived classes must implement `_fit`, `_solve`, `_predict_chunk`, and `_var_chunk`.
    These models incorporate both values and their gradients into the fit.
    """

    def __init__(
        self,
        x,
        y,
        gradients=None,
        smoothing=1e-4,
        length_scale=None,
        optimize=True,
        **kwargs,
    ):
        """
        Initializes and fits the gradient-enhanced surface model.

        Args:
            x: Training inputs (N, D).
            y: Training values (N,).
            gradients: Training gradients (N, D).
            smoothing: Initial noise/smoothing parameter.
            length_scale: Initial length scale parameter(s).
            optimize: Whether to optimize parameters.
            **kwargs: Additional model-specific parameters.
        """
        self.x = jnp.asarray(x, dtype=jnp.float32)
        y_energies = jnp.asarray(y, dtype=jnp.float32)[:, None]
        grad_vals = (
            jnp.asarray(gradients, dtype=jnp.float32)
            if gradients is not None
            else jnp.zeros_like(self.x)
        )

        self.y_full = jnp.concatenate([y_energies, grad_vals], axis=1)
        self.e_mean = jnp.mean(y_energies)
        self.y_full = self.y_full.at[:, 0].add(-self.e_mean)
        self.y_flat = self.y_full.flatten()
        self.D_plus_1 = self.x.shape[1] + 1

        self._fit(smoothing, length_scale, optimize)
        self._solve()

    def _fit(self, smoothing, length_scale, optimize):
        """Internal method to perform parameter optimization."""
        raise NotImplementedError

    def _solve(self):
        """Internal method to solve the linear system for weights."""
        raise NotImplementedError

    def __call__(self, x_query, chunk_size=500):
        """
        Predict values at query points.

        Args:
            x_query: Query inputs (M, D).
            chunk_size: Number of points to process per batch.

        Returns:
            jnp.ndarray: Predicted values (M,).
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        preds = []
        for i in range(0, x_query.shape[0], chunk_size):
            chunk = x_query[i : i + chunk_size]
            preds.append(self._predict_chunk(chunk))
        return jnp.concatenate(preds, axis=0) + self.e_mean

    def predict_var(self, x_query, chunk_size=500):
        """
        Predict posterior variance at query points.

        Args:
            x_query: Query inputs (M, D).
            chunk_size: Number of points to process per batch.

        Returns:
            jnp.ndarray: Predicted variances (M,).
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        vars_list = []
        for i in range(0, x_query.shape[0], chunk_size):
            chunk = x_query[i : i + chunk_size]
            vars_list.append(self._var_chunk(chunk))
        return jnp.concatenate(vars_list, axis=0)

    def _predict_chunk(self, chunk):
        """Internal method for batch prediction."""
        raise NotImplementedError

    def _var_chunk(self, chunk):
        """Internal method for batch variance."""
        raise NotImplementedError


# ==============================================================================
# 1. TPS IMPLEMENTATION (Optimizable)
# ==============================================================================


@jit
def _tps_kernel_matrix(x):
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K = r**2 * jnp.log(r)
    return K


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

    def __init__(self, x_obs, y_obs, smoothing=1e-3, optimize=True, **kwargs):
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
# 2. MATERN 5/2
# ==============================================================================


@jit
def _matern_kernel_matrix(x, length_scale):
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)

    # Matérn 5/2 Kernel
    # k(r) = (1 + sqrt(5)r/l + 5r^2/3l^2) * exp(-sqrt(5)r/l)
    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    K = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(-sqrt5_r_l)
    return K


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
# 3. GRADIENT-ENHANCED MATERN
# ==============================================================================


def matern_kernel_elem(x1, x2, length_scale=1.0):
    d2 = jnp.sum((x1 - x2) ** 2)
    r = jnp.sqrt(d2 + 1e-12)
    ls = jnp.squeeze(length_scale)
    sqrt5_r_l = jnp.sqrt(5.0) * r / ls
    val = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * ls**2)) * jnp.exp(-sqrt5_r_l)
    return val


def full_covariance_matern(x1, x2, length_scale):
    k_ee = matern_kernel_elem(x1, x2, length_scale)
    k_ed = jax.grad(matern_kernel_elem, argnums=1)(x1, x2, length_scale)
    k_de = jax.grad(matern_kernel_elem, argnums=0)(x1, x2, length_scale)
    k_dd = jax.jacfwd(jax.grad(matern_kernel_elem, argnums=1), argnums=0)(
        x1, x2, length_scale
    )
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_matern_grad_map = vmap(
    vmap(full_covariance_matern, (None, 0, None)), (0, None, None)
)


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
# 4. STANDARD IMQ (Optimizable)
# ==============================================================================


@jit
def _imq_kernel_matrix(x, epsilon):
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    K = 1.0 / jnp.sqrt(d2 + epsilon**2)
    return K


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


# ==============================================================================
# 6. SQUARED EXPONENTIAL (SE) - "The Classic"
# ==============================================================================


def se_kernel_elem(x1, x2, length_scale=1.0):
    d2 = jnp.sum((x1 - x2) ** 2)
    ls = jnp.maximum(length_scale, 1e-5)
    val = jnp.exp(-d2 / (2.0 * ls**2))
    return val


def full_covariance_se(x1, x2, length_scale):
    k_ee = se_kernel_elem(x1, x2, length_scale)
    k_ed = jax.grad(se_kernel_elem, argnums=1)(x1, x2, length_scale)
    k_de = jax.grad(se_kernel_elem, argnums=0)(x1, x2, length_scale)
    k_dd = jax.jacfwd(jax.grad(se_kernel_elem, argnums=1), argnums=0)(
        x1, x2, length_scale
    )
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_se_grad_map = vmap(vmap(full_covariance_se, (None, 0, None)), (0, None, None))


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
# 5. GRADIENT-ENHANCED IMQ (Optimizable)
# ==============================================================================


def imq_kernel_elem(x1, x2, epsilon=1.0):
    d2 = jnp.sum((x1 - x2) ** 2)
    val = 1.0 / jnp.sqrt(d2 + epsilon**2)
    return val


def full_covariance_imq(x1, x2, epsilon):
    k_ee = imq_kernel_elem(x1, x2, epsilon)
    k_ed = jax.grad(imq_kernel_elem, argnums=1)(x1, x2, epsilon)
    k_de = jax.grad(imq_kernel_elem, argnums=0)(x1, x2, epsilon)
    k_dd = jax.jacfwd(jax.grad(imq_kernel_elem, argnums=1), argnums=0)(x1, x2, epsilon)
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_imq_grad_map = vmap(vmap(full_covariance_imq, (None, 0, None)), (0, None, None))


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
# 7. RATIONAL QUADRATIC (RQ)
# ==============================================================================


def rq_kernel_base(x1, x2, length_scale, alpha):
    """Standard RQ Kernel: (1 + r^2 / (2*alpha*l^2))^-alpha"""
    d2 = jnp.sum((x1 - x2) ** 2)
    base = 1.0 + d2 / (2.0 * alpha * (length_scale**2) + 1e-6)
    val = base ** (-alpha)
    return val


def rq_kernel_elem(x1, x2, params):
    length_scale = params[0]
    alpha = params[1]
    k_direct = rq_kernel_base(x1, x2, length_scale, alpha)
    k_mirror = rq_kernel_base(x1[::-1], x2, length_scale, alpha)
    return k_direct + k_mirror


def full_covariance_rq(x1, x2, params):
    k_ee = rq_kernel_elem(x1, x2, params)
    k_ed = jax.grad(rq_kernel_elem, argnums=1)(x1, x2, params)
    k_de = jax.grad(rq_kernel_elem, argnums=0)(x1, x2, params)
    k_dd = jax.jacfwd(jax.grad(rq_kernel_elem, argnums=1), argnums=0)(x1, x2, params)
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_rq_grad_map = vmap(vmap(full_covariance_rq, (None, 0, None)), (0, None, None))


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

    def _fit(self, smoothing, length_scale, optimize):
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
# NYSTRÖM GRADIENT-ENHANCED IMQ
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

    def _fit(self, smoothing, length_scale, optimize):
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


def get_surface_model(name):
    """
    Factory function to retrieve surface model classes by name.

    Args:
        name: Model identifier (e.g., 'grad_matern', 'tps', 'imq').

    Returns:
        type: The model class. Defaults to GradientMatern.
    """
    models = {
        "grad_matern": GradientMatern,
        "grad_rq": GradientRQ,
        "grad_se": GradientSE,
        "grad_imq": GradientIMQ,
        "grad_imq_ny": NystromGradientIMQ,
        "matern": FastMatern,
        "imq": FastIMQ,
        "tps": FastTPS,
        "rbf": FastTPS,
    }
    return models.get(name, GradientMatern)
