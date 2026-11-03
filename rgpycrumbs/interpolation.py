import numpy as np


def spline_interp(x, y, num=None, knots=3):
    """Interpolate using B-splines.

    Fits a B-spline representation to the data ``(x, y)`` and evaluates it on
    a uniformly spaced grid.

    Args:
        x: Independent variable values (1D array-like, strictly increasing).
        y: Dependent variable values (same length as *x*).
        num: Number of evaluation points on the output grid.
            It defaults to ``(len(x) - 1) * knots + 1``.
            The grid step is then the mean sample spacing divided by
            *knots*, so the grid never has fewer points than the input.
            For evenly spaced *x* the grid passes through every sample.
            With an odd degree, such as the cubic default, each interval
            between samples then gets the ``knots + 1`` grid points that
            fix the spline polynomial on it.
        knots: Degree of the spline (passed as ``k`` to
            ``scipy.interpolate.splrep``).
            The default of 3 is the cubic spline that SciPy recommends.

    Returns:
        tuple: ``(x_fine, y_fine)`` where *x_fine* is a uniform grid of length
        *num* spanning ``[x.min(), x.max()]`` and *y_fine* is the spline
        evaluated on that grid.

    Raises:
        ValueError: If *x* and *y* are not one-dimensional and of equal
            length, if *x* is not strictly increasing, if there are no more
            samples than *knots*, or if *num* is below one.

    .. versionadded:: 1.0.0
    .. versionchanged:: 1.14.0
       Inputs are validated before SciPy sees them, and lists are accepted.
       A repeated *x* used to return all-NaN values.
       *num* defaults to a grid derived from the samples instead of 100
       points, which dropped detail from inputs with more samples than that.
    """
    from rgpycrumbs._aux import ensure_import

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.size != y.size:
        raise ValueError(f"x and y differ in length: {x.size} != {y.size}")
    if x.size <= knots:
        raise ValueError(
            f"a degree {knots} spline needs more than {knots} samples, got {x.size}"
        )
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if num is None:
        num = (x.size - 1) * knots + 1
    elif num < 1:
        raise ValueError(f"num must be at least 1, got {num}")

    scipy_interp = ensure_import("scipy.interpolate")
    spl = scipy_interp.splrep(x, y, k=knots)
    x_fine = np.linspace(x.min(), x.max(), num=num)
    y_fine = scipy_interp.splev(x_fine, spl)
    return x_fine, y_fine
