# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Stable public suite API for config, pins/locks, and on-demand imports.

This is the **documented** surface for consumers (chemparseplot, pychum, wailord,
cookbooks). Prefer these names over importing private modules.

Example::

    from rgpycrumbs.api import load_config, load_pypi_pins, ensure_import

    cfg = load_config()
    pins = cfg.merged_package_pins_normalized()
    # optional lock file
    # pins = load_pypi_pins("uv.lock")

    jax = ensure_import("jax")  # needs RGPKGS_AUTO_DEPS=1 or host install

See ``docs/orgmode/explanation/public_api.org`` and suite architecture docs.

.. versionadded:: 1.9.18
"""

from __future__ import annotations

from rgpycrumbs._aux import ensure_import, lazy_import
from rgpycrumbs.config import (
    CONFIG_PATH_ENV,
    load_config,
    resolve_auto_deps_default,
    resolve_force_uv,
    resolve_lock_path_layered,
    user_config_path,
)
from rgpycrumbs.locks import (
    LOCK_PATH_ENV,
    apply_pin_to_spec,
    load_pypi_pins,
    normalize_pypi_name,
    pins_from_env,
    pins_to_constraint_lines,
)

__all__ = [
    "CONFIG_PATH_ENV",
    "LOCK_PATH_ENV",
    "apply_pin_to_spec",
    "ensure_import",
    "lazy_import",
    "load_config",
    "load_pypi_pins",
    "normalize_pypi_name",
    "pins_from_env",
    "pins_to_constraint_lines",
    "resolve_auto_deps_default",
    "resolve_force_uv",
    "resolve_lock_path_layered",
    "suite_pins",
    "user_config_path",
]


def suite_pins() -> dict[str, str]:
    """Merged package pins: ``RGPKGS_LOCK_PINS`` / env pins ∪ layered config.

    Single implementation for the suite. Consumers (chemparseplot, wailord)
    should re-export or call this rather than reimplementing the merge.

    Soft-fails to env-only pins if config load errors (never raises for missing
    config files). Does not install packages and does not require uv.
    """
    pins = dict(pins_from_env())
    try:
        pins.update(load_config().merged_package_pins_normalized())
    except Exception:  # noqa: BLE001 — soft fail for consumers
        pass
    return pins
