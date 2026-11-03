"""Shared numerical policy for surface hyperparameter fits.

Each value below is derived from the data, the floating-point format, or a
cited estimator:

* Length scales start at the median Euclidean distance between distinct
  inputs.
  Garreau, Jitkrittum and Kanagawa analyse this median heuristic, and note
  that some authors use the median distance itself rather than that
  distance over ``sqrt(2)``, https://doi.org/10.48550/arXiv.1707.07269.
  Repeated inputs are left out, so they do not pull the start toward zero.
  The IMQ kernel ``1 / sqrt(r**2 + epsilon**2)`` depends on ``r`` only through
  ``r / epsilon`` up to its amplitude, so ``epsilon`` takes the same start.
  Riabiz et al. set the IMQ length scale to the median distance in the same
  way, https://doi.org/10.1111/rssb.12503.
* ``IMQ_EPSILON_PRIOR_SHAPE`` is the shape of the gamma prior on the IMQ
  ``epsilon``.
  Chung et al. propose shape 2 for a standard deviation because the density
  is zero at zero with a positive slope, so the estimate stays off zero.
  The mode of that prior is ``1 / rate``, which they suggest setting to a
  prior estimate, https://doi.org/10.1007/s11336-013-9328-2.
  The rate here puts the mode at the starting ``epsilon``.
* ``RQ_INITIAL_ALPHA`` starts the rational-quadratic mixture at shape 1.
  The RQ kernel is a gamma scale mixture of squared-exponential kernels over
  inverse squared length scales (Rasmussen and Williams, eq. 4.20).
  Shape 1 makes that mixture exponential, the maximum-entropy distribution on
  the positive reals for a fixed mean (Cover and Thomas, chapter 12,
  https://doi.org/10.1002/047174882X).
* ``minimize_log_parameters`` runs SciPy's L-BFGS-B with its documented
  default tolerances and returns the last iterate its line search accepted.
  The BFGS loop in JAX 0.9.2 moves to the trial point even when the line
  search fails, and ``jax.scipy.optimize.minimize`` ignores ``tol``,
  https://github.com/jax-ml/jax/blob/jax-v0.9.2/jax/_src/scipy/optimize/bfgs.py.
* ``covariance_roundoff_factor`` implements Higham's
  ``gamma_n = n u / (1 - n u)`` with unit roundoff ``u = eps(dtype) / 2``.
  A computed Cholesky factor ``R`` of an order-``n`` matrix ``A`` satisfies
  ``R.T R = A + dA`` with ``|dA| <= gamma_(n+1) |R.T| |R|`` (Theorem 10.3),
  and every entry of ``|R.T| |R|`` is at most the largest diagonal entry of
  ``A``.
  ``covariance_roundoff_jitter`` adds that entrywise bound to the diagonal,
  so a 300-point gradient fit in two dimensions (order 900, float32) gets
  ``gamma_901 = 5.37e-5`` times the largest entry.
  ``safe_cholesky_solve`` reports a factorization that still fails as a NaN
  log-determinant instead of a wrong finite value.
  See Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  chapters 3 and 10, https://doi.org/10.1137/1.9780898718027.
* ``observation_noise`` keeps the caller's ``smoothing`` as the noise
  variance.
  Fits without a caller ``smoothing``, or with zero, treat the observations
  as exact.
  Fixed fits and the IMQ noise caps then set the noise to the roundoff jitter.
  ``starting_noise`` starts optimized exact-data fits at ``covariance_scale``,
  a signal-to-noise ratio of one.
  The largest eigenvalue of a positive semidefinite ``K`` is at most ``n``
  times its largest entry, so ``K + covariance_scale(K) I`` starts the
  optimizer with a condition number of at most ``n + 1``.
* ``NYSTROM_RANDOM_SEED`` only makes the random inducing subset reproducible.
  No fit is tuned to its value, and callers can pass their own ``seed``.
"""

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from rgpycrumbs._aux import ensure_import

IMQ_EPSILON_PRIOR_SHAPE: Final = 2.0
RQ_INITIAL_ALPHA: Final = 1.0
NYSTROM_RANDOM_SEED: Final = 42


def median_pairwise_distance(x: jnp.ndarray) -> jnp.ndarray:
    """Return the median Euclidean distance between distinct input points."""
    if x.ndim != 2 or x.shape[0] < 2:
        msg = "length-scale initialization requires at least two input points"
        raise ValueError(msg)

    rows, columns = jnp.triu_indices(x.shape[0], k=1)
    distances = jnp.sqrt(jnp.sum((x[rows] - x[columns]) ** 2, axis=-1))
    distinct = distances[distances > 0]
    if distinct.size == 0:
        msg = "length-scale initialization requires two distinct input points"
        raise ValueError(msg)
    return jnp.median(distinct)


def minimize_log_parameters(loss_fn, x0) -> jnp.ndarray:
    """Minimize a float32 JAX objective of log-parameters with L-BFGS-B."""
    scipy_optimize = ensure_import("scipy.optimize")
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def objective(log_params):
        value, grad = value_and_grad(jnp.asarray(log_params, dtype=jnp.float32))
        return float(value), np.asarray(grad, dtype=np.float64)

    result = scipy_optimize.minimize(
        objective, np.asarray(x0, dtype=np.float64), jac=True, method="L-BFGS-B"
    )
    return jnp.asarray(result.x, dtype=jnp.float32)


def covariance_roundoff_factor(matrix_order: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Return Higham's ``gamma_(n+1)`` Cholesky factor for a matrix of order ``n``."""
    unit_roundoff = jnp.finfo(dtype).eps / 2
    accumulated = (matrix_order + 1) * unit_roundoff
    return accumulated / (1 - accumulated)


def covariance_scale(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the largest absolute entry, or one for a zero matrix."""
    entry_scale = jnp.max(jnp.abs(matrix))
    return jnp.where(entry_scale > 0, entry_scale, jnp.ones((), dtype=matrix.dtype))


def covariance_roundoff_jitter(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the Cholesky backward-error bound for ``matrix`` as a diagonal shift."""
    order = matrix.shape[0]
    return covariance_roundoff_factor(order, matrix.dtype) * covariance_scale(matrix)


def observation_noise(smoothing, matrix: jnp.ndarray) -> float:
    """Return the caller's smoothing, or the roundoff jitter for exact data."""
    if smoothing is None or smoothing == 0:
        return float(covariance_roundoff_jitter(matrix))
    if smoothing < 0:
        msg = "smoothing must be non-negative, or None for exact observations"
        raise ValueError(msg)
    return float(smoothing)


def starting_noise(smoothing, matrix: jnp.ndarray) -> float:
    """Return the caller's smoothing, or the covariance scale as a noise start."""
    if smoothing is None or smoothing == 0:
        return float(covariance_scale(matrix))
    return observation_noise(smoothing, matrix)


__all__ = [
    "IMQ_EPSILON_PRIOR_SHAPE",
    "NYSTROM_RANDOM_SEED",
    "RQ_INITIAL_ALPHA",
    "covariance_roundoff_factor",
    "covariance_roundoff_jitter",
    "covariance_scale",
    "median_pairwise_distance",
    "minimize_log_parameters",
    "observation_noise",
    "starting_noise",
]
