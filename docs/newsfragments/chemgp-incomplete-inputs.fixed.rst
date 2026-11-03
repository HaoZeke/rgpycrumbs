The ``quality`` subcommand of ``rgpycrumbs chemgp plot-gp`` now names the missing ``grids/gp_mean_N<n>`` grids.
It used to fail inside Matplotlib with ``Number of rows must be a positive integer, not 0``.
For a file without any ``gp_ls<j>_sv<i>`` panel, ``sensitivity`` now fails instead of saving an empty figure.
Without ``points/optimum``, ``nll`` now draws the landscape with no optimum marker.
It used to stop with a ``KeyError`` from h5py.
