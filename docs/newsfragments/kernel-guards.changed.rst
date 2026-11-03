The surface kernels take their guards from the float32 machine epsilon ``eps = 2**-23``.
``rgpycrumbs.surfaces._hyperparameters`` names them.
``LENGTH_SCALE_FLOOR`` is ``eps`` and replaces the ``1e-5`` floor of the squared-exponential length scale.
``SQUARED_DISTANCE_GUARD`` is ``eps**2`` and replaces the ``1e-12`` under the thin-plate and Matern square roots.
In float32 the ``1e-12`` was a no-op for every squared distance from ``2**-15`` up.
``RQ_DENOMINATOR_GUARD`` is ``eps**2`` and replaces the ``1e-6`` in the rational-quadratic denominator.
That ``1e-6`` changed the kernel by half a percent at length scale ``0.01`` and by half at ``1e-3``.
