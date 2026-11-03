import jax.numpy as jnp

from rgpycrumbs.surfaces._hyperparameters import (
    covariance_roundoff_factor,
    covariance_roundoff_jitter,
)


def safe_cholesky_solve(K, y, noise_scalar):
    """
    Solve with covariance-scaled roundoff jitter.

    .. versionadded:: 1.0.0

    .. versionchanged:: 1.14.0
       The jitter is the Cholesky roundoff bound of ``K``, and the
       ``jitter_steps`` retries are gone.
       A factorization with non-finite entries returns zeros and a NaN
       log-determinant.

    Args:
        K: Covariance matrix.
        y: Observation vector.
        noise_scalar: Initial noise level.

    Returns:
        tuple: (alpha, log_det) where alpha is the solution vector and
               log_det is the log determinant of the jittered matrix.
    """
    N = K.shape[0]
    jitter = noise_scalar + covariance_roundoff_jitter(K)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N, dtype=K.dtype))
    valid = jnp.all(jnp.isfinite(L))
    safe_L = jnp.where(valid, L, jnp.eye(N, dtype=K.dtype))
    alpha = jnp.linalg.solve(safe_L.T, jnp.linalg.solve(safe_L, y))
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(safe_L)))
    return (
        jnp.where(valid, alpha, jnp.zeros_like(y)),
        jnp.where(valid, log_det, jnp.asarray(jnp.nan, dtype=K.dtype)),
    )


def equilibrated_inverse_solve(K, y, noise_scalar):
    """
    Inverse and weights of a gradient-enhanced covariance matrix.

    .. versionadded:: 1.11.1

    The value and gradient rows of ``K`` differ in scale by ``1 / l**2``, so a
    short length scale and repeated inputs leave ``K`` too ill-conditioned for a
    float32 inverse. The matrix is solved as ``K + noise * I + jitter * diag(K)``
    after a symmetric diagonal scaling to unit diagonal, which keeps the jitter
    relative to each row. ``jitter`` is ``covariance_roundoff_factor`` for the
    order of ``K``.

    .. versionchanged:: 1.14.0
       The relative jitter is the Cholesky roundoff factor for the order of
       ``K``, and the ``jitter`` argument is gone.

    Args:
        K: Covariance matrix without noise.
        y: Observation vector.
        noise_scalar: Noise level added to the diagonal.

    Returns:
        tuple: (alpha, K_inv) for the regularized matrix.
    """
    jitter = covariance_roundoff_factor(K.shape[0], K.dtype)
    K_reg = K + jnp.diag(noise_scalar + jitter * jnp.diag(K))
    scale = 1.0 / jnp.sqrt(jnp.diag(K_reg))
    K_unit = scale[:, None] * K_reg * scale[None, :]
    K_inv = scale[:, None] * jnp.linalg.inv(K_unit) * scale[None, :]
    alpha = scale * jnp.linalg.solve(K_unit, scale * y.flatten())
    return alpha, K_inv


def generic_negative_mll(K, y, noise_scalar):
    """
    Calculates the negative Marginal Log-Likelihood (MLL).

    .. versionadded:: 1.0.0

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


class BaseSurface:
    """
    Abstract base class for standard (non-gradient) surface models.

    .. versionadded:: 1.0.0

    Derived classes must implement `_fit`, `_solve`, `_predict_chunk`, and `_var_chunk`.
    """

    def __init__(
        self, x_obs, y_obs, smoothing=1e-3, length_scale=None, optimize=True, **_kwargs
    ):
        """
        Initializes and fits the surface model.

        Args:
            x_obs: Training inputs (N, D).
            y_obs: Training observations (N,).
            smoothing: Noise variance of fixed fits and the start of optimized
                ones. Zero means exact data, and optimized fits then start at
                the largest prior variance.
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

        .. versionadded:: 1.1.0

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

    .. versionadded:: 1.1.0

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
        noise_per_obs=None,
        **_kwargs,
    ):
        """
        Initializes and fits the gradient-enhanced surface model.

        Args:
            x: Training inputs (N, D).
            y: Training values (N,).
            gradients: Training gradients (N, D).
            smoothing: Noise variance of fixed fits and the start of optimized
                ones. Zero means exact data, and optimized fits then start at
                the largest prior variance.
            length_scale: Initial length scale parameter(s).
            optimize: Whether to optimize parameters.
            noise_per_obs: Per-observation noise array (N,) for heteroscedastic
                GP. When provided, each observation gets its own noise level
                instead of the global smoothing parameter.
            **kwargs: Additional model-specific parameters.
        """
        self._noise_per_obs = (
            jnp.asarray(noise_per_obs, dtype=jnp.float32)
            if noise_per_obs is not None
            else None
        )
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

        .. versionadded:: 1.1.0

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
