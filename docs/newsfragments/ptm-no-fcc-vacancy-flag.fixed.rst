``rgpycrumbs eon ptmdisp --no-fcc-vacancy`` is now a flag that turns the vacancy filter off.
Without it, the command runs the filter.
The option used to take a boolean value, which the command passed on inverted.
So the filter ran only with ``--no-fcc-vacancy true``.
