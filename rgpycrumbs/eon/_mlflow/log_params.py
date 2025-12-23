import configparser
from pathlib import Path

import mlflow

from rgpycrumbs._aux import _import_from_parent_env

# Access the legacy EON config module
eon_config = _import_from_parent_env("eon.config")


def log_config_ini(conf_ini: Path = Path("config.ini"), *, w_artifact: bool = True):
    """
    Logs the full EON configuration state, including schema defaults.

    This function reconstructs the hierarchical configuration by combining
    the EON schema with user-defined overrides. It preserves the 'Section/Key'
    provenance required for rigorous simulation tracking.
    """
    # 1. Instantiate the class to load the config.yaml schema
    econf = eon_config.ConfigClass()

    # 2. Build a local parser to hold the hydrated state
    # This emulates the logic inside econf.init but preserves the parser object
    hydrated_parser = configparser.ConfigParser()

    # 3. Apply defaults from the EON schema (provenance of defaults)
    for section in econf.format:
        section_name = section.name
        if not hydrated_parser.has_section(section_name):
            hydrated_parser.add_section(section_name)

        for config_key in section.keys:
            # We use the default value defined in the EON source
            hydrated_parser.set(section_name, config_key.name, str(config_key.default))

    # 4. Apply user overrides from the config.ini file
    if conf_ini.exists():
        hydrated_parser.read(str(conf_ini.absolute()))

    # 5. Log the combined state to MLflow
    # We map 'kind' strings to ConfigParser getters for type-safe logging
    type_getters = {
        "int": hydrated_parser.getint,
        "float": hydrated_parser.getfloat,
        "boolean": hydrated_parser.getboolean,
        "string": hydrated_parser.get,
    }

    for section in econf.format:
        section_name = section.name
        for config_key in section.keys:
            key_name = config_key.name
            getter = type_getters.get(config_key.kind, hydrated_parser.get)

            try:
                # Retrieve the hydrated value (default or user-override)
                val = getter(section_name, key_name)
                mlflow.log_param(f"{section_name}/{key_name}", val)
            except (ValueError, configparser.Error):
                # Fallback to raw string logging upon type mismatch
                mlflow.log_param(
                    f"{section_name}/{key_name}",
                    hydrated_parser.get(section_name, key_name),
                )

    if w_artifact:
        mlflow.log_artifact(str(conf_ini.absolute()), "inputs")
