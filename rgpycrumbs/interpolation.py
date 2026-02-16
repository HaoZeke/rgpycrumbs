import numpy as np
from scipy.interpolate import splev, splrep


def spline_interp(x, y, num=100, knots=3):
    spl = splrep(x, y, k=knots)
    x_fine = np.linspace(x.min(), x.max(), num=num)
    y_fine = splev(x_fine, spl)
    return x_fine, y_fine
