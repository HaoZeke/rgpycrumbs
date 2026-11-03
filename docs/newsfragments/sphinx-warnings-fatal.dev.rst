The ``sphinxbld`` pixi task, and with it the documentation workflow, now fails on any Sphinx warning.
The ``uma`` command pages import their dispatched scripts to render the CLI reference.
The task sets ``RGPYCRUMBS_SUPPRESS_SCRIPT_IMPORT_WARNING=1`` to silence the warning those imports print.
