# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Declarative TOML plot configuration for eOn CLIs.

Shared style/render knobs live under ``[shared]`` (or at the top level).
Command-specific inputs/outputs live under ``[neb]``, ``[min]``, or ``[saddle]``.

Merge order (later wins for explicit CLI overrides only)::

    shared defaults < command defaults < [shared] + [command] from file
    < Click parameters whose source is not DEFAULT

```{versionadded} 1.8.2
```
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback for library import
    try:
        import tomli as tomllib
    except ImportError as exc:  # pragma: no cover
        msg = "Reading plot config on Python <3.11 requires tomli"
        raise ImportError(msg) from exc

# Keys that are shared across plot entry points (style / render / energy).
SHARED_KEYS: frozenset[str] = frozenset(
    {
        "energy_unit",
        "theme",
        "dpi",
        "figsize",
        "facecolor",
        "fontsize_base",
        "strip_renderer",
        "xyzrender_config",
        "strip_spacing",
        "strip_dividers",
        "strip_zoom",
        "rotation",
        "perspective_tilt",
        "ira_kmax",
        "surface_type",
        "project_path",
        "plot_structures",
        "auto_thin",
        "max_surface_points",
        "verbose",
    }
)

SHARED_DEFAULTS: dict[str, Any] = {
    "energy_unit": "eV",
    "theme": "ruhi",
    "dpi": 200,
    "figsize": (5.37, 5.37),
    "facecolor": None,
    "fontsize_base": None,
    "strip_renderer": "xyzrender",
    "xyzrender_config": "paton",
    "strip_spacing": 1.5,
    "strip_dividers": False,
    "strip_zoom": None,
    "rotation": "auto",
    "perspective_tilt": 0.0,
    "ira_kmax": 14.0,
    "surface_type": "rbf",
    "project_path": True,
    "plot_structures": "none",
    "auto_thin": False,
    "max_surface_points": 64,
    "verbose": False,
}

COMMAND_DEFAULTS: dict[str, dict[str, Any]] = {
    "neb": {
        "plot_type": "profile",
        "title": "NEB Path",
        "source": "eon",
        "landscape_mode": "surface",
        "landscape_path": "all",
        "rc_mode": "path",
        "surface_type": "rbf",
        "show_pts": True,
        "plot_mode": "energy",
        "normalize_rc": False,
        "highlight_last": True,
        "spline_method": "hermite",
        "zoom_ratio": 0.5,
        "show_legend": True,
        "show_evolution": False,
        "force_recompute": False,
        "input_dat_pattern": "neb_*.dat",
        "input_path_pattern": "neb_path_*.con",
        "sp_file": "sp.con",
        "cache_file": ".neb_landscape.parquet",
    },
    "min": {
        "plot_type": "profile",
        "prefix": "minimization",
        "surface_type": "grad_matern",
        "plot_structures": "none",
        "auto_thin": False,
        "max_surface_points": 64,
    },
    "saddle": {
        "plot_type": "profile",
        "surface_type": "grad_matern",
        "plot_structures": "none",
    },
}

# Config keys that should become pathlib.Path (scalars or lists).
_PATH_KEYS: frozenset[str] = frozenset(
    {
        "con_file",
        "sp_file",
        "input_h5",
        "input_traj",
        "output_file",
        "output",
        "cache_file",
        "peak_dir",
        "ref_product",
        "job_dir",
    }
)

_LIST_PATH_KEYS: frozenset[str] = frozenset({"job_dir"})


def load_plot_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML plot config file into a plain dict of sections."""
    config_path = Path(path)
    raw = config_path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        msg = f"Plot config root must be a table, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key == "figsize" and isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if key in _LIST_PATH_KEYS:
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(item) for item in value]
    if key in _PATH_KEYS:
        return Path(value)
    if key == "label" and isinstance(value, str):
        return (value,)
    if key == "label" and isinstance(value, list):
        return tuple(value)
    if key == "job_dir" and isinstance(value, list):
        return tuple(Path(item) for item in value)
    return value


def _flatten_section(section: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_value(key, value) for key, value in section.items()}


def extract_config_layers(
    data: dict[str, Any], command: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a loaded TOML mapping into shared and command-specific layers."""
    shared: dict[str, Any] = {}
    command_layer: dict[str, Any] = {}

    shared_block = data.get("shared")
    if isinstance(shared_block, dict):
        shared.update(_flatten_section(shared_block))

    command_block = data.get(command)
    if isinstance(command_block, dict):
        command_layer.update(_flatten_section(command_block))

    # Flat keys at top level: shared knobs apply globally; others go to command.
    for key, value in data.items():
        if key in {"shared", "neb", "min", "saddle"}:
            continue
        if isinstance(value, dict):
            continue
        coerced = _coerce_value(key, value)
        if key in SHARED_KEYS:
            shared[key] = coerced
        else:
            command_layer[key] = coerced
    return shared, command_layer


def merge_plot_settings(
    command: str,
    *,
    config_path: str | Path | None = None,
    config_data: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
    passthrough: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the resolved settings mapping for a plot command.

    Parameters
    ----------
    command
        One of ``neb``, ``min``, ``saddle``.
    config_path
        Optional TOML file path (ignored when *config_data* is given).
    config_data
        Already-loaded TOML mapping (for tests).
    cli_overrides
        Parameters explicitly set on the CLI (non-default Click source).
    passthrough
        Remaining Click parameter values (defaults) used to fill gaps for
        command-specific path/identity options that are not in the schema.
    """
    if command not in COMMAND_DEFAULTS:
        msg = f"Unknown plot command {command!r}; expected neb|min|saddle"
        raise ValueError(msg)

    settings: dict[str, Any] = {}
    settings.update(SHARED_DEFAULTS)
    settings.update(COMMAND_DEFAULTS[command])

    data = config_data
    if data is None and config_path is not None:
        data = load_plot_config(config_path)
    if data is not None:
        shared_layer, command_layer = extract_config_layers(data, command)
        settings.update(shared_layer)
        settings.update(command_layer)

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None and key not in SHARED_KEYS:
                # Keep explicit None only when meaningful; skip empty overrides.
                continue
            settings[key] = (
                _coerce_value(key, value)
                if key in _PATH_KEYS | _LIST_PATH_KEYS | {"figsize", "label", "job_dir"}
                else value
            )

    if passthrough:
        for key, value in passthrough.items():
            if key in settings:
                continue
            settings[key] = value

    # Normalize job_dir / label sequences for single-ended plotters.
    if "job_dir" in settings and settings["job_dir"] is not None:
        jd = settings["job_dir"]
        if isinstance(jd, (str, Path)):
            settings["job_dir"] = (Path(jd),)
        elif isinstance(jd, list):
            settings["job_dir"] = tuple(Path(p) for p in jd)
        elif isinstance(jd, tuple):
            settings["job_dir"] = tuple(Path(p) for p in jd)

    if "label" in settings and settings["label"] is not None:
        lab = settings["label"]
        if isinstance(lab, str):
            settings["label"] = (lab,)
        elif isinstance(lab, list):
            settings["label"] = tuple(lab)

    return settings


def click_nondefault_overrides(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Return Click parameters whose source is not the decorator default."""
    from click.core import ParameterSource

    overrides: dict[str, Any] = {}
    for name, value in params.items():
        if name == "config":
            continue
        try:
            source = ctx.get_parameter_source(name)
        except Exception:  # pragma: no cover - older click
            continue
        if source is not None and source != ParameterSource.DEFAULT:
            overrides[name] = value
    return overrides


def resolve_from_click(
    command: str,
    ctx: Any,
    *,
    config: str | Path | None,
    **params: Any,
) -> dict[str, Any]:
    """Resolve settings for a Click command callback."""
    overrides = click_nondefault_overrides(ctx, params)
    return merge_plot_settings(
        command,
        config_path=config,
        cli_overrides=overrides,
        passthrough=params,
    )


def run_plot(
    command: str,
    runner: Any,
    *,
    config: str | Path | None = None,
    **overrides: Any,
) -> Any:
    """Merge settings for *command* then call *runner*(settings).

    Shared by the library ``plot_*`` entry points. *runner* is typically
    ``plot_*_from_settings``.
    """
    from rgpycrumbs._aux import enable_library_auto_deps

    enable_library_auto_deps()
    settings = merge_plot_settings(
        command,
        config_path=config,
        cli_overrides=overrides or None,
    )
    return runner(settings)


def run_from_click(
    command: str,
    runner: Any,
    ctx: Any,
    *,
    config: str | Path | None = None,
    **params: Any,
) -> Any:
    """CLI path: ``resolve_from_click`` then *runner*(settings)."""
    return runner(resolve_from_click(command, ctx, config=config, **params))


def library_plot(command: str, runner: Any) -> Any:
    """Build a keyword-only library entry that shares :func:`run_plot`."""

    def plot(*, config: str | Path | None = None, **overrides: Any) -> Any:
        return run_plot(command, runner, config=config, **overrides)

    plot.__name__ = f"plot_{command}"
    plot.__qualname__ = f"plot_{command}"
    plot.__doc__ = (
        f"Library entry for eOn {command} plots (no Click argv).\n\n"
        f"Same pipeline as ``rgpycrumbs eon plt-{command}``."
    )
    return plot


MINIMAL_CONFIG_EXAMPLE = """\
# Minimal rgpkgs eOn plot config (TOML)
# Use: rgpycrumbs eon plt-neb --config plot.toml
#      rgpycrumbs eon plt-min --config plot.toml
#
# Prefer this file for surface-fit knobs instead of growing CLI flags.
# auto_thin defaults to false (opt-in); max_surface_points caps the GP fit set.

[shared]
energy_unit = "eV"
theme = "ruhi"
dpi = 200
figsize = [5.37, 5.37]
strip_renderer = "xyzrender"
xyzrender_config = "paton"
ira_kmax = 14.0
# Surface fit (chemparseplot plot_landscape_surface / single-ended landscapes)
auto_thin = false
max_surface_points = 64

[neb]
con_file = "neb.con"
plot_type = "landscape"
output_file = "neb_landscape.pdf"
title = "NEB path"
plot_structures = "crit_points"
surface_type = "grad_imq"

[min]
job_dir = ["minimization_run"]
plot_type = "landscape"
output = "min_landscape.pdf"
prefix = "minimization"
surface_type = "grad_imq"
# Opt in for dense eOn write_movies force-eval clouds:
# auto_thin = true
# max_surface_points = 64

[saddle]
job_dir = ["saddle_run"]
plot_type = "profile"
output = "saddle_profile.pdf"
"""
