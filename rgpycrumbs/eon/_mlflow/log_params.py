"""Logging of eOn configuration parameters to MLflow.

Performs a three-way merge between the eOn schema, hydrated defaults, and
user-provided ``config.ini`` overrides, then logs every parameter to MLflow
for full provenance tracking.

.. versionadded:: 1.1.0
"""

import configparser
from pathlib import Path

import mlflow

from rgpycrumbs._aux import _import_from_parent_env

# Access the legacy eOn config module
eon_config = _import_from_parent_env("eon.config")


def log_config_ini(
    conf_ini: Path = Path("config.ini"),
    *,
    w_artifact: bool = True,
    track_overrides: bool = False,
):
    """Log the hydrated eOn configuration to MLflow.

    Performs a three-way merge between the eOn schema, the hydrated defaults,
    and the user-provided ``config.ini``.  Every parameter is logged for full
    provenance; user-provided overrides can optionally be tagged separately.

    Args:
        conf_ini: Path to the eOn ``config.ini`` file.
        w_artifact: Whether to log the config file as an MLflow artifact.
        track_overrides: Whether to separately log parameters that the user
            explicitly set (i.e. values that differ from schema defaults).

    .. versionadded:: 1.1.0
    """
    econf = eon_config.ConfigClass()

    # Build a local parser to hold the hydrated state
    # This emulates the logic inside econf.init but preserves the parser object
    hydrated_parser = configparser.ConfigParser()
    user_parser = configparser.ConfigParser()

    # Populate hydrated state with defaults from schema
    for section in econf.format:
        if not hydrated_parser.has_section(section.name):
            hydrated_parser.add_section(section.name)
        for config_key in section.keys:
            hydrated_parser.set(section.name, config_key.name, str(config_key.default))

    # Read user overrides if they exist
    if conf_ini.exists():
        user_parser.read(str(conf_ini.absolute()))
        hydrated_parser.read(str(conf_ini.absolute()))

    # Map 'kind' strings to ConfigParser getters for type-safe logging
    type_getters = {
        "int": hydrated_parser.getint,
        "float": hydrated_parser.getfloat,
        "boolean": hydrated_parser.getboolean,
        "string": hydrated_parser.get,
    }

    # Log all parameters and focus on overrides
    for section in econf.format:
        section_name = section.name
        for config_key in section.keys:
            key_name = config_key.name
            full_key = f"{section_name}/{key_name}"
            getter = type_getters.get(config_key.kind, hydrated_parser.get)

            try:
                # Always log the hydrated value for the full record
                val = getter(section_name, key_name)
                mlflow.log_param(full_key, val)

                if track_overrides:
                    # Focused logging: If the user explicitly provided this in the INI
                    if user_parser.has_option(section_name, key_name):
                        mlflow.log_param(f"Overrides/{full_key}", val)

            except (ValueError, configparser.Error):
                raw_val = hydrated_parser.get(section_name, key_name)
                mlflow.log_param(full_key, raw_val)
                if track_overrides:
                    if user_parser.has_option(section_name, key_name):
                        mlflow.log_param(f"Overrides/{full_key}", raw_val)

    # Tag the run for easy filtering of specific overrides
    if conf_ini.exists():
        overridden_sections = ", ".join(user_parser.sections())
        mlflow.set_tag("config.overridden_sections", overridden_sections)

    if w_artifact:
        mlflow.log_artifact(str(conf_ini.absolute()), "inputs")
