"""Shared numerical policy for surface hyperparameter fits.

Each value below is derived from the data, the floating-point format, or a
cited estimator:

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
"""

import jax.numpy as jnp


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


__all__ = [
    "covariance_roundoff_factor",
    "covariance_roundoff_jitter",
    "covariance_scale",
]
