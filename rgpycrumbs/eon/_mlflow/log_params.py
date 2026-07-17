"""Logging of eOn configuration parameters to MLflow.

Performs a merge between eon-schema L0 catalog defaults and user
``config.ini`` overrides, then logs every parameter to MLflow for full
provenance tracking.

.. versionadded:: 1.1.0
.. versionchanged:: 1.9.x
    Prefer ``eon-schema`` hydrate/catalog; eon-akmc ``ConfigClass`` is a
    one-release fallback only.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import mlflow


def log_config_ini(
    conf_ini: Path = Path("config.ini"),
    *,
    w_artifact: bool = True,
    track_overrides: bool = False,
):
    """Log the hydrated eOn configuration to MLflow.

    Args:
        conf_ini: Path to the eOn ``config.ini`` file.
        w_artifact: Whether to log the config file as an MLflow artifact.
        track_overrides: Whether to separately log parameters that the user
            explicitly set (values present in the INI, not only defaults).

    .. versionadded:: 1.1.0
    """
    conf_ini = Path(conf_ini)
    try:
        _log_via_eon_schema(
            conf_ini, w_artifact=w_artifact, track_overrides=track_overrides
        )
    except ImportError:
        _log_via_eon_config_fallback(
            conf_ini, w_artifact=w_artifact, track_overrides=track_overrides
        )


def _log_via_eon_schema(
    conf_ini: Path,
    *,
    w_artifact: bool,
    track_overrides: bool,
) -> None:
    from eon_schema.config import hydrate_ini, read_ini

    user: dict[str, dict[str, str]] = {}
    if conf_ini.exists():
        user = read_ini(conf_ini)

    hydrated = hydrate_ini(user)
    for section_name, options in hydrated.items():
        for key_name, val in options.items():
            full_key = f"{section_name}/{key_name}"
            mlflow.log_param(full_key, val)
            if track_overrides and conf_ini.exists():
                if key_name in user.get(section_name, {}):
                    mlflow.log_param(f"Overrides/{full_key}", val)

    if conf_ini.exists():
        mlflow.set_tag("config.overridden_sections", ", ".join(user.keys()))
        mlflow.set_tag("config.hydrate_backend", "eon-schema")

    if w_artifact and conf_ini.exists():
        mlflow.log_artifact(str(conf_ini.absolute()), "inputs")


def _log_via_eon_config_fallback(
    conf_ini: Path,
    *,
    w_artifact: bool,
    track_overrides: bool,
) -> None:
    """Legacy path: full eon-akmc ConfigClass (one-release fallback)."""
    from rgpycrumbs._aux import _import_from_parent_env

    eon_config = _import_from_parent_env("eon.config")
    if eon_config is None:
        raise ImportError(
            "log_config_ini needs eon-schema>=0.2 (preferred) or eon-akmc "
            "(eon.config) available for fallback.\n"
            "  pip install 'rgpycrumbs[eon]'\n"
            "  # or: pip install 'eon-schema>=0.2'"
        )

    econf = eon_config.ConfigClass()

    hydrated_parser = configparser.ConfigParser()
    user_parser = configparser.ConfigParser()

    for section in econf.format:
        if not hydrated_parser.has_section(section.name):
            hydrated_parser.add_section(section.name)
        for config_key in section.keys:
            hydrated_parser.set(section.name, config_key.name, str(config_key.default))

    if conf_ini.exists():
        user_parser.read(str(conf_ini.absolute()))
        hydrated_parser.read(str(conf_ini.absolute()))

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
            full_key = f"{section_name}/{key_name}"
            getter = type_getters.get(config_key.kind, hydrated_parser.get)

            try:
                val = getter(section_name, key_name)
                mlflow.log_param(full_key, val)

                if track_overrides:
                    if user_parser.has_option(section_name, key_name):
                        mlflow.log_param(f"Overrides/{full_key}", val)

            except (ValueError, configparser.Error):
                raw_val = hydrated_parser.get(section_name, key_name)
                mlflow.log_param(full_key, raw_val)
                if track_overrides:
                    if user_parser.has_option(section_name, key_name):
                        mlflow.log_param(f"Overrides/{full_key}", raw_val)

    if conf_ini.exists():
        overridden_sections = ", ".join(user_parser.sections())
        mlflow.set_tag("config.overridden_sections", overridden_sections)
        mlflow.set_tag("config.hydrate_backend", "eon-akmc-fallback")

    if w_artifact and conf_ini.exists():
        mlflow.log_artifact(str(conf_ini.absolute()), "inputs")
