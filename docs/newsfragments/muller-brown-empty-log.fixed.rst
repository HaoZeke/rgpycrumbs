The Muller-Brown plot script ``rgpycrumbs.xts.saddle.mb`` now says when a log has no ``Iteration`` line to plot.
It used to fail inside NumPy with ``need at least one array to concatenate``.
Lines with ``nan`` coordinates are still skipped.
