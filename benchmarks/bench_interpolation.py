"""ASV benchmarks for rgpycrumbs.interpolation (scipy B-splines)."""

import numpy as np

from rgpycrumbs.interpolation import spline_interp


class TimeSplineInterp:
    """Benchmark B-spline interpolation at varying input sizes."""

    # Output grid is 5x the input size
    params = [100, 1000]
    param_names = ["input_size"]

    def setup(self, input_size):
        self.x = np.linspace(0, 4 * np.pi, input_size)
        self.y = np.sin(self.x)
        self.n_out = input_size * 5

    def time_spline_interp(self, input_size):
        spline_interp(self.x, self.y, num=self.n_out, knots=3)
