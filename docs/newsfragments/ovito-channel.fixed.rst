The ``ptm`` pixi environment installs the Python module of the Open Visualization Tool (OVITO) from ``https://conda.ovito.org``.
The conda-forge ``ovito`` package is a retired desktop build without the Python module.
With it, every ``ptm`` test was skipped.
The environment takes OVITO 3.16.0 or later.
Without a display, the ``ptm`` tests pass with 3.16.0 and every later build, and fail with every earlier one.
The ``test`` feature no longer installs that desktop build.
