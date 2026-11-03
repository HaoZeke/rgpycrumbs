Surface fits with ``optimize=True`` now run SciPy's ``L-BFGS-B`` and keep the last point that its line search accepted.
They used ``jax.scipy.optimize.minimize``, whose ``BFGS`` loop moves to the trial point even when the line search fails.
On the five-point bowl in the tests, ``FastMatern`` stopped that way at a length scale of ``2.9e-6``.
The gradient there was ``0.13``.
That ``minimize`` also ignores ``tol``, so the ``tol=1e-3`` that every model passed had no effect.
Hyperparameter fits now need SciPy, which ``ensure_import`` installs when ``RGPYCRUMBS_AUTO_DEPS`` is set.
