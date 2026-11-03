``rgpycrumbs.interpolation.spline_interp`` now derives its default grid from the samples.
Without ``num`` the grid has ``(len(x) - 1) * knots + 1`` points instead of a fixed 100.
It never has fewer points than the input, and it passes through every sample of an evenly spaced *x*.
The fixed 100 points dropped detail from inputs with more samples than that.
Pass ``num`` to keep a fixed grid.
