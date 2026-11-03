The surface solvers now add the Cholesky roundoff bound of each covariance matrix to its diagonal.
That bound is Higham's ``gamma_(n+1)`` times the largest entry, so it follows the matrix order and the data scale.
The dense thin-plate, Matérn and inverse multiquadric solves used to add only the smoothing.
With zero smoothing, repeated inputs gave NaN weights.
The gradient and Nystrom solves added fixed shifts of ``1e-6`` and ``1e-4`` instead.
``safe_cholesky_solve`` now spots a failed factorization from its non-finite entries.
``jax.numpy.linalg.cholesky`` returns NaNs there instead of raising, so the old retry loop never retried.
