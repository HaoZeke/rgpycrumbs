# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Layered TOML config for dispatch pins and defaults.

Search order for config *files* (later overrides earlier on a per-key basis
when merged; lock path is last-writer-wins):

1. User global: ``$XDG_CONFIG_HOME/rgpycrumbs/config.toml``
   (default ``~/.config/rgpycrumbs/config.toml``)
2. Project: walk upward from CWD for ``rgpycrumbs.toml`` or ``.rgpycrumbs.toml``
3. ``RGPYCRUMBS_CONFIG`` — explicit file (merged last among files)

Runtime precedence for values (highest first):

* CLI flags (``--lock``, ``--sbom``, ``--dev``, …)
* Environment variables (``RGPYCRUMBS_LOCK``, ``RGPYCRUMBS_FORCE_UV``, …)
* Merged TOML (project overrides global)
* Built-in defaults

Example ``~/.config/rgpycrumbs/config.toml``::

    [dispatch]
    auto_deps = true
    force_uv = false

    [pins]
    # optional default lock (uv.lock | pylock.toml | CycloneDX JSON)
    lock = ""

    [pins.packages]
    # explicit pins always applied (override lock file versions)
    # jax = "0.4.31"

.. versionadded:: 1.9.15
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

# Well-known names
PROJECT_CONFIG_NAMES = ("rgpycrumbs.toml", ".rgpycrumbs.toml")
USER_CONFIG_REL = Path("rgpycrumbs") / "config.toml"
CONFIG_PATH_ENV = "RGPYCRUMBS_CONFIG"


def _xdg_config_home() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg)
    return Path.home() / ".config"


def user_config_path() -> Path:
    """``~/.config/rgpycrumbs/config.toml`` (or ``$XDG_CONFIG_HOME/...``)."""
    return _xdg_config_home() / USER_CONFIG_REL


def find_project_config(start: Path | None = None) -> Path | None:
    """Walk from *start* (default CWD) upward for a project config file."""
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        for name in PROJECT_CONFIG_NAMES:
            candidate = directory / name
            if _is_file(candidate):
                return candidate
        # stop at filesystem root
        if directory.parent == directory:
            break
    return None


def _is_file(path: Path) -> bool:
    """Existence check via os.path (not Path.is_file) to avoid class mocks."""
    return os.path.isfile(path)


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
    except Exception as exc:  # noqa: BLE001
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
        # allow "jax = '0.4.31'" or jax = { version = "..." }
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


@dataclass
class RgpycrumbsConfig:
    """Resolved configuration (files + defaults; env/CLI applied separately)."""

    auto_deps: bool | None = None
    force_uv: bool | None = None
    lock_path: Path | None = None
    # explicit package pins from [pins.packages] (merged global→project)
    package_pins: dict[str, str] = field(default_factory=dict)
    # which config files contributed (for diagnostics)
    sources: list[Path] = field(default_factory=list)

    def merged_package_pins_normalized(self) -> dict[str, str]:
        from rgpycrumbs.locks import normalize_pypi_name

        return {
            normalize_pypi_name(k): v for k, v in self.package_pins.items() if v
        }


def _merge_section(
    cfg: RgpycrumbsConfig,
    data: dict[str, Any],
    *,
    base_dir: Path,
    source: Path,
) -> None:
    dispatch = data.get("dispatch") or {}
    if isinstance(dispatch, dict):
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

    cfg.sources.append(source)


def load_config(
    *,
    cwd: Path | None = None,
    explicit_config: Path | str | None = None,
) -> RgpycrumbsConfig:
    """Load and merge global + project + optional explicit config files."""
    cfg = RgpycrumbsConfig()

    user_path = user_config_path()
    if _is_file(user_path):
        _merge_section(
            cfg,
            _load_toml_file(user_path),
            base_dir=user_path.parent,
            source=user_path,
        )

    project_path = find_project_config(cwd)
    if project_path is not None:
        _merge_section(
            cfg,
            _load_toml_file(project_path),
            base_dir=project_path.parent,
            source=project_path,
        )

    # explicit path: CLI/env RGPYCRUMBS_CONFIG
    exp: Path | None = None
    if explicit_config is not None and str(explicit_config).strip():
        exp = Path(str(explicit_config).strip()).expanduser()
    else:
        env_cfg = os.environ.get(CONFIG_PATH_ENV, "").strip()
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
    """CLI → env → config file lock path."""
    from rgpycrumbs.locks import LOCK_PATH_ENV, SBOM_PATH_ENV

    for candidate in (cli_lock, cli_sbom):
        if candidate is not None and str(candidate).strip():
            return Path(str(candidate).strip()).expanduser().resolve()

    for env_name in (LOCK_PATH_ENV, SBOM_PATH_ENV):
        env_path = os.environ.get(env_name, "").strip()
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
    """Whether to force uv isolation (False if --dev / RGPYCRUMBS_DEV)."""
    if is_dev:
        return False
    if os.environ.get("RGPYCRUMBS_DEV", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    env = os.environ.get("RGPYCRUMBS_FORCE_UV", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    cfg = config if config is not None else load_config(cwd=cwd)
    if cfg.force_uv is not None:
        return cfg.force_uv
    return False  # default: may still prefer in-env when stack ready


def resolve_auto_deps_default(
    *,
    config: RgpycrumbsConfig | None = None,
    cwd: Path | None = None,
) -> str:
    """Return ``'1'`` or ``'0'`` for default AUTO_DEPS when env unset."""
    env = os.environ.get("RGPYCRUMBS_AUTO_DEPS", "").strip()
    if env != "":
        return env
    cfg = config if config is not None else load_config(cwd=cwd)
    if cfg.auto_deps is False:
        return "0"
    return "1"  # default on for CLI dispatch
