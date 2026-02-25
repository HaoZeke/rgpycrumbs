import numpy as np


def spline_interp(x, y, num=100, knots=3):
    """Interpolate using B-splines.

    Fits a B-spline representation to the data ``(x, y)`` and evaluates it on
    a uniformly spaced grid.

    Args:
        x: Independent variable values (1D array, must be monotonically
            increasing).
        y: Dependent variable values (same length as *x*).
        num: Number of evaluation points on the output grid.
        knots: Degree of the spline (passed as ``k`` to
            ``scipy.interpolate.splrep``).

    Returns:
        tuple: ``(x_fine, y_fine)`` where *x_fine* is a uniform grid of length
        *num* spanning ``[x.min(), x.max()]`` and *y_fine* is the spline
        evaluated on that grid.

    .. versionadded:: 1.0.0
    """
    from rgpycrumbs._aux import ensure_import

    scipy_interp = ensure_import("scipy.interpolate")
    spl = scipy_interp.splrep(x, y, k=knots)
    x_fine = np.linspace(x.min(), x.max(), num=num)
    y_fine = scipy_interp.splev(x_fine, spl)
    return x_fine, y_fine
