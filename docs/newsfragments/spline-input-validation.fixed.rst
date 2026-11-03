``rgpycrumbs.interpolation.spline_interp`` now raises ``ValueError`` for inputs it cannot fit.
It checks that *x* is strictly increasing and that *x* and *y* are one-dimensional and of equal length.
It also needs more samples than the spline degree and a ``num`` of at least one.
A repeated *x* used to return NaN values without an error.
The other cases failed inside SciPy with a ``TypeError`` or a message that named no argument.
Lists now work as well as arrays.
