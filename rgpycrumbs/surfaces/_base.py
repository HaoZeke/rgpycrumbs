import logging

import jax.numpy as jnp


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


class BaseSurface:
    """
    Abstract base class for standard (non-gradient) surface models.

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
        **_kwargs,
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
