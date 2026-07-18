# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Ecosystem-wide layered TOML config (rgpkgs suite).

Shared by **rgpycrumbs**, **chemparseplot**, and siblings — not a
package-private silo. One pin/lock policy for the whole stack.

Search order for config *files* (later merges override earlier):

1. User global: ``$XDG_CONFIG_HOME/rgpkgs/config.toml``
   (default ``~/.config/rgpkgs/config.toml``)
2. Legacy user: ``~/.config/rgpycrumbs/config.toml`` (still read)
3. Project: walk upward for ``rgpkgs.toml`` / ``.rgpkgs.toml``, then legacy
   ``rgpycrumbs.toml`` / ``.rgpycrumbs.toml``
4. ``RGPKGS_CONFIG`` or ``RGPYCRUMBS_CONFIG`` — explicit file (merged last)

Runtime precedence for values (highest first):

* CLI flags (``--lock``, ``--sbom``, ``--dev``, …)
* Environment (``RGPKGS_LOCK`` / ``RGPYCRUMBS_LOCK``, ``RGPKGS_FORCE_UV``, …)
* Merged TOML (project overrides global)
* Built-in defaults

Example ``~/.config/rgpkgs/config.toml``::

    # Shared across the suite (chemparseplot, rgpycrumbs, …)
    [pins]
    lock = "uv.lock"   # or pylock.toml / CycloneDX JSON

    [pins.packages]
    jax = "0.4.31"

    # Tool-specific (optional)
    [rgpycrumbs.dispatch]
    auto_deps = true
    force_uv = false

    # Future: [chemparseplot.plot] theme = "…"

.. versionadded:: 1.9.15
.. versionchanged:: 1.9.16
   Ecosystem identity is **rgpkgs** (not per-package config trees).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

# --- well-known paths (ecosystem first, legacy second) ---
PROJECT_CONFIG_NAMES = (
    "rgpkgs.toml",
    ".rgpkgs.toml",
    "rgpycrumbs.toml",  # legacy
    ".rgpycrumbs.toml",
)
USER_CONFIG_REL = Path("rgpkgs") / "config.toml"
USER_CONFIG_LEGACY_REL = Path("rgpycrumbs") / "config.toml"
CONFIG_PATH_ENV = "RGPKGS_CONFIG"
CONFIG_PATH_ENV_LEGACY = "RGPYCRUMBS_CONFIG"
# lock path envs (also used by locks.py / CLI)
LOCK_PATH_ENVS = ("RGPKGS_LOCK", "RGPYCRUMBS_LOCK", "RGPYCRUMBS_SBOM")
FORCE_UV_ENVS = ("RGPKGS_FORCE_UV", "RGPYCRUMBS_FORCE_UV")
AUTO_DEPS_ENVS = ("RGPKGS_AUTO_DEPS", "RGPYCRUMBS_AUTO_DEPS")
DEV_ENVS = ("RGPKGS_DEV", "RGPYCRUMBS_DEV")


def _xdg_config_home() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg)
    return Path.home() / ".config"


def user_config_path() -> Path:
    """Preferred global config: ``~/.config/rgpkgs/config.toml``."""
    return _xdg_config_home() / USER_CONFIG_REL


def user_config_path_legacy() -> Path:
    """Legacy global: ``~/.config/rgpycrumbs/config.toml``."""
    return _xdg_config_home() / USER_CONFIG_LEGACY_REL


def _is_file(path: Path) -> bool:
    """Existence check via os.path (not Path.is_file) to avoid class mocks."""
    return os.path.isfile(path)


def find_project_config(start: Path | None = None) -> Path | None:
    """Walk from *start* (default CWD) upward for a project config file.

    Prefers ``rgpkgs.toml`` over legacy ``rgpycrumbs.toml`` in the same
    directory.
    """
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        for name in PROJECT_CONFIG_NAMES:
            candidate = directory / name
            if _is_file(candidate):
                return candidate
        if directory.parent == directory:
            break
    return None


def _load_toml_file(path: Path) -> dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        msg = "TOML config requires Python 3.11+ tomllib or tomli"
        raise ImportError(msg)
    if not _is_file(path):
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        msg = f"Invalid TOML config {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Config root must be a table: {path}"
        raise ValueError(msg)
    return data


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off", ""}:
        return False
    return None


def _normalize_packages(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k is None or v is None:
            continue
        if isinstance(v, dict):
            ver = v.get("version") or v.get("pin")
            if ver is None:
                continue
            out[str(k)] = str(ver)
        else:
            out[str(k)] = str(v)
    return out


def _resolve_path(raw: str, base_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _first_env(*names: str) -> str:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


def _env_flag_any(*names: str) -> bool | None:
    """Return True/False if any env is set; None if all unset."""
    for name in names:
        raw = os.environ.get(name, "").strip().lower()
        if raw == "":
            continue
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return None


@dataclass
class RgpycrumbsConfig:
    """Resolved configuration (files + defaults; env/CLI applied separately).

    Shared pin fields apply suite-wide; dispatch fields are rgpycrumbs CLI
    defaults (other tools may ignore them).
    """

    auto_deps: bool | None = None
    force_uv: bool | None = None
    lock_path: Path | None = None
    package_pins: dict[str, str] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)
    # reserved for future tool tables (chemparseplot, …)
    tool_tables: dict[str, dict[str, Any]] = field(default_factory=dict)

    def merged_package_pins_normalized(self) -> dict[str, str]:
        from rgpycrumbs.locks import normalize_pypi_name

        return {normalize_pypi_name(k): v for k, v in self.package_pins.items() if v}


def _extract_dispatch(data: dict[str, Any]) -> dict[str, Any]:
    """Return dispatch table from top-level or [rgpycrumbs.dispatch]."""
    out: dict[str, Any] = {}
    top = data.get("dispatch")
    if isinstance(top, dict):
        out.update(top)
    rgp = data.get("rgpycrumbs")
    if isinstance(rgp, dict):
        nested = rgp.get("dispatch")
        if isinstance(nested, dict):
            out.update(nested)
    return out


def _merge_section(
    cfg: RgpycrumbsConfig,
    data: dict[str, Any],
    *,
    base_dir: Path,
    source: Path,
) -> None:
    dispatch = _extract_dispatch(data)
    if "auto_deps" in dispatch:
        b = _as_bool(dispatch.get("auto_deps"))
        if b is not None:
            cfg.auto_deps = b
    if "force_uv" in dispatch:
        b = _as_bool(dispatch.get("force_uv"))
        if b is not None:
            cfg.force_uv = b

    pins = data.get("pins") or {}
    if isinstance(pins, dict):
        lock = pins.get("lock") or pins.get("sbom") or pins.get("path")
        if lock is not None and str(lock).strip():
            cfg.lock_path = _resolve_path(str(lock).strip(), base_dir)
        packages = pins.get("packages") or {}
        for name, ver in _normalize_packages(packages).items():
            cfg.package_pins[name] = ver

    # top-level convenience aliases
    if "lock" in data and data["lock"] is not None and str(data["lock"]).strip():
        cfg.lock_path = _resolve_path(str(data["lock"]).strip(), base_dir)
    if "force_uv" in data:
        b = _as_bool(data.get("force_uv"))
        if b is not None:
            cfg.force_uv = b
    if "auto_deps" in data:
        b = _as_bool(data.get("auto_deps"))
        if b is not None:
            cfg.auto_deps = b

    # stash other tool tables for future consumers
    for key in ("chemparseplot", "pychum", "readcon"):
        if key in data and isinstance(data[key], dict):
            cfg.tool_tables[key] = data[key]

    cfg.sources.append(source)


def load_config(
    *,
    cwd: Path | None = None,
    explicit_config: Path | str | None = None,
) -> RgpycrumbsConfig:
    """Load and merge global + project + optional explicit config files."""
    cfg = RgpycrumbsConfig()

    # Preferred global, then legacy global
    for path in (user_config_path(), user_config_path_legacy()):
        if _is_file(path):
            _merge_section(
                cfg,
                _load_toml_file(path),
                base_dir=path.parent,
                source=path,
            )

    project_path = find_project_config(cwd)
    if project_path is not None:
        _merge_section(
            cfg,
            _load_toml_file(project_path),
            base_dir=project_path.parent,
            source=project_path,
        )

    exp: Path | None = None
    if explicit_config is not None and str(explicit_config).strip():
        exp = Path(str(explicit_config).strip()).expanduser()
    else:
        env_cfg = _first_env(CONFIG_PATH_ENV, CONFIG_PATH_ENV_LEGACY)
        if env_cfg:
            exp = Path(env_cfg).expanduser()
    if exp is not None:
        if not _is_file(exp):
            msg = f"Config path does not exist: {exp}"
            raise FileNotFoundError(msg)
        _merge_section(
            cfg,
            _load_toml_file(exp),
            base_dir=exp.parent,
            source=exp.resolve(),
        )

    return cfg


def resolve_lock_path_layered(
    *,
    cli_lock: str | None = None,
    cli_sbom: str | None = None,
    config: RgpycrumbsConfig | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """CLI → env (RGPKGS_* / RGPYCRUMBS_*) → config file lock path."""
    for candidate in (cli_lock, cli_sbom):
        if candidate is not None and str(candidate).strip():
            return Path(str(candidate).strip()).expanduser().resolve()

    env_path = _first_env(*LOCK_PATH_ENVS)
    if env_path:
        return Path(env_path).expanduser().resolve()

    cfg = config if config is not None else load_config(cwd=cwd)
    if cfg.lock_path is not None:
        return cfg.lock_path.expanduser().resolve()
    return None


def resolve_force_uv(
    *,
    is_dev: bool,
    config: RgpycrumbsConfig | None = None,
    cwd: Path | None = None,
) -> bool:
    """Whether to force uv isolation (False if --dev / DEV env)."""
    if is_dev:
        return False
    if _env_flag_any(*DEV_ENVS) is True:
        return False
    forced = _env_flag_any(*FORCE_UV_ENVS)
    if forced is not None:
        return forced
    cfg = config if config is not None else load_config(cwd=cwd)
    if cfg.force_uv is not None:
        return cfg.force_uv
    return False


def resolve_auto_deps_default(
    *,
    config: RgpycrumbsConfig | None = None,
    cwd: Path | None = None,
) -> str:
    """Return ``'1'`` or ``'0'`` for default AUTO_DEPS when env unset."""
    for name in AUTO_DEPS_ENVS:
        env = os.environ.get(name, "").strip()
        if env != "":
            return env
    cfg = config if config is not None else load_config(cwd=cwd)
    if cfg.auto_deps is False:
        return "0"
    return "1"
