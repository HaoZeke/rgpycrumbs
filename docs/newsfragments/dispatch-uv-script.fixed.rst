Dispatched scripts run under ``uv run --script``, so an rgpycrumbs installed by
``uvx`` dispatches at all: uv refused the script's directory inside its own
cache as a project. Without ``uv`` on ``PATH`` and without the plot stack in the
active interpreter, the dispatcher now says so before the script fails on its
first import.
