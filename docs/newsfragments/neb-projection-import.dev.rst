``rgpycrumbs.eon.plt_neb`` imports ``chemparseplot.parse.projection`` without falling back to ``None``.
Every chemparseplot release without that module also lacks other names that ``plt_neb`` imports.
So the fallback never ran.
The chemparseplot floor of 1.13.0 already covers every name that ``plt_neb`` imports.
