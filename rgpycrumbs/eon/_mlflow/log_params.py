from pathlib import Path

import mlflow

from rgpycrumbs._aux import _import_from_parent_env

eon_config = _import_from_parent_env("eon.config")


def log_config_ini(conf_ini: Path = Path("config.ini"), *, w_artifact: bool = True):
    """Logs EON config.ini parameters to the current MLflow run."""
    econf = eon_config.ConfigClass()
    # EON ConfigClass requires a string path
    econf.init(str(conf_ini.absolute()))

    excluded_keys = {"format", "init", "init_done"}
    # Filter out private methods and internal state
    all_conf_keys = [
        x for x in dir(econf) if not x.startswith("__") and x not in excluded_keys
    ]

    for key in all_conf_keys:
        val = getattr(econf, key)
        # Log each configuration option as a parameter
        mlflow.log_param(key, val)

    if w_artifact:
        mlflow.log_artifact(str(conf_ini.absolute()), "inputs")
