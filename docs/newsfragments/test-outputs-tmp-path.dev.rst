The ``eon`` tests keep their MLflow runs under the ``tmp_path`` of each test.
An autouse fixture in ``tests/conftest.py`` points ``MLFLOW_TRACKING_URI`` there and clears the active experiment.
Those tests used to leave ``mlflow.db`` and ``mlruns/`` in the working directory.
The ``ptm`` tests that render take the image path from a fixture under ``tmp_path``.
