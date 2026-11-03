``rgpycrumbs.basetypes`` defines its records itself and no longer imports chemparseplot.
It used to try ``chemparseplot.basetypes`` first, which only the 0.x chemparseplot releases have.
chemparseplot imports these records from ``rgpycrumbs.basetypes``, as the module docstring now says.
