# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Consume Python lock files / SBOMs as install pins for dispatch + AUTO_DEPS.

Supported formats (auto-detected by path/content):

* **PEP 751** ``pylock.toml`` / ``pylock.*.toml`` — standards-track lock
* **uv.lock** — uv native TOML lock (``[[package]]``)
* **CycloneDX JSON** — e.g. eb-stack ``--sbom-out`` or ``uv export --format cyclonedx1.5``

rgpycrumbs does **not** generate these files. When a path is provided via
``RGPYCRUMBS_LOCK`` / ``--lock`` (or the SBOM aliases), PyPI packages become
``name==version`` constraints for ``uv run`` and ``ensure_import``.

Non-Python / non-PyPI entries (e.g. eb-stack ``pkg:generic/...``) are skipped.

.. versionadded:: 1.9.13
.. versionchanged:: 1.9.14
   Also accept PEP 751 pylock and uv.lock (not only CycloneDX).
"""

from __future__ import annotations

import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]
        _TOML_IMPORT_ERROR = exc
    else:
        _TOML_IMPORT_ERROR = None

# Primary path env (any supported format). Ecosystem name first.
LOCK_PATH_ENV = "RGPKGS_LOCK"
LOCK_PATH_ENV_LEGACY = "RGPYCRUMBS_LOCK"
# Backward-compatible alias (CycloneDX or any format — same loader).
SBOM_PATH_ENV = "RGPYCRUMBS_SBOM"
# JSON object {normalized_name: version} for child ensure_import / AUTO_DEPS.
PINS_ENV = "RGPKGS_LOCK_PINS"
PINS_ENV_LEGACY = "RGPYCRUMBS_LOCK_PINS"
# Older SBOM-era name still read (dispatch writes all).
SBOM_PINS_ENV = "RGPYCRUMBS_SBOM_PINS"

_VERSION_OPS = re.compile(r"(==|>=|<=|~=|!=|>|<)")
_PYLOCK_NAME = re.compile(r"^pylock(\.[^.]+)?\.toml$", re.IGNORECASE)


class LockFormat(str, Enum):
    PYLOCK = "pylock"
    UV_LOCK = "uv.lock"
    CYCLONEDX = "cyclonedx"
    UNKNOWN = "unknown"


def normalize_pypi_name(name: str) -> str:
    """PEP 503-ish name normalization for pin lookups."""
    return re.sub(r"[-_.]+", "-", name).lower()


def package_name_from_spec(pip_spec: str) -> str:
    """Return the distribution name from a pip requirement string."""
    spec = pip_spec.strip()
    base = spec.split("[", maxsplit=1)[0]
    parts = _VERSION_OPS.split(base, maxsplit=1)
    return parts[0].strip()


def _require_tomllib() -> Any:
    if tomllib is None:  # pragma: no cover
        msg = (
            "Reading TOML lock files requires Python 3.11+ tomllib "
            "or the tomli package"
        )
        raise ImportError(msg) from _TOML_IMPORT_ERROR
    return tomllib


def _pin_from_purl(purl: str) -> tuple[str, str] | None:
    """Parse ``pkg:pypi/name@version`` (optional qualifiers after ``?``)."""
    if not purl.startswith("pkg:pypi/"):
        return None
    rest = purl[len("pkg:pypi/") :]
    rest = rest.split("?", maxsplit=1)[0]
    if "@" not in rest:
        return None
    name, version = rest.rsplit("@", maxsplit=1)
    name = name.strip().replace("%2D", "-").replace("%5F", "_")
    version = version.strip()
    if not name or not version:
        return None
    return name, version


def detect_lock_format(path: str | Path, text: str | None = None) -> LockFormat:
    """Guess lock format from filename and optional file text."""
    p = Path(path)
    name = p.name.lower()
    if name == "uv.lock":
        return LockFormat.UV_LOCK
    if _PYLOCK_NAME.match(name):
        return LockFormat.PYLOCK
    if name.endswith((".cdx.json", ".cdx")) or name.endswith("cyclonedx.json"):
        return LockFormat.CYCLONEDX
    if name.endswith(".json"):
        return LockFormat.CYCLONEDX

    body = text if text is not None else ""
    if not body and p.is_file():
        body = p.read_text(encoding="utf-8")[:4000]
    stripped = body.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(body if text is not None else p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if isinstance(data, dict) and (
            data.get("bomFormat") == "CycloneDX" or "components" in data
        ):
            return LockFormat.CYCLONEDX
    if "lock-version" in body and "[[packages]]" in body:
        return LockFormat.PYLOCK
    if "[[package]]" in body and ("version =" in body or "version=" in body):
        # uv.lock uses [[package]]; pylock uses [[packages]]
        if "[[packages]]" not in body:
            return LockFormat.UV_LOCK
    if name.endswith(".toml"):
        # ambiguous toml: prefer pylock markers
        if "lock-version" in body:
            return LockFormat.PYLOCK
        if "[[package]]" in body:
            return LockFormat.UV_LOCK
    return LockFormat.UNKNOWN


def pypi_pins_from_cyclonedx(doc: dict[str, Any]) -> dict[str, str]:
    """Extract ``{normalized_name: version}`` for PyPI components only."""
    pins: dict[str, str] = {}
    components = doc.get("components") or []
    if not isinstance(components, list):
        return pins

    for comp in components:
        if not isinstance(comp, dict):
            continue
        purl = comp.get("purl") or ""
        if isinstance(purl, str) and purl:
            parsed = _pin_from_purl(purl)
            if parsed is not None:
                name, version = parsed
                pins[normalize_pypi_name(name)] = version
                continue
            if purl.startswith("pkg:") and not purl.startswith("pkg:pypi/"):
                continue

        bom_ref = comp.get("bom-ref") or ""
        if isinstance(bom_ref, str) and bom_ref.startswith("pkg:pypi/"):
            parsed = _pin_from_purl(bom_ref)
            if parsed is not None:
                name, version = parsed
                pins[normalize_pypi_name(name)] = version
                continue

        ctype = (comp.get("type") or "").lower()
        name = comp.get("name")
        version = comp.get("version")
        if ctype in {"pypi", "library"} and name and version:
            props = comp.get("properties") or []
            marked_pypi = False
            if isinstance(props, list):
                for prop in props:
                    if not isinstance(prop, dict):
                        continue
                    if prop.get("name") in {
                        "pypi:package",
                        "rgpycrumbs:ecosystem",
                    } and str(prop.get("value", "")).lower() in {
                        "1",
                        "true",
                        "pypi",
                        "python",
                    }:
                        marked_pypi = True
                        break
            if ctype == "pypi" or marked_pypi:
                pins[normalize_pypi_name(str(name))] = str(version)

    return pins


def pypi_pins_from_pylock(doc: dict[str, Any]) -> dict[str, str]:
    """Extract pins from a PEP 751 pylock.toml document.

    Uses ``[[packages]]`` entries with ``name`` + ``version``. Packages without
    a version (rare) are skipped. Marker diversity is ignored for pin map
    purposes: last wins (constraints still force that version for installs).
    """
    pins: dict[str, str] = {}
    packages = doc.get("packages") or []
    if not isinstance(packages, list):
        return pins
    for pkg in packages:
        if not isinstance(pkg, dict):
            continue
        name = pkg.get("name")
        version = pkg.get("version")
        if not name or version is None or version == "":
            continue
        pins[normalize_pypi_name(str(name))] = str(version)
    return pins


def pypi_pins_from_uv_lock(doc: dict[str, Any]) -> dict[str, str]:
    """Extract pins from a uv.lock TOML document (``[[package]]`` tables)."""
    pins: dict[str, str] = {}
    packages = doc.get("package") or []
    if not isinstance(packages, list):
        return pins
    for pkg in packages:
        if not isinstance(pkg, dict):
            continue
        name = pkg.get("name")
        version = pkg.get("version")
        if not name or version is None or version == "":
            continue
        # Skip virtual path sources without a real version? uv always has version.
        pins[normalize_pypi_name(str(name))] = str(version)
    return pins


def load_cyclonedx(path: str | Path) -> dict[str, Any]:
    """Load and validate a CycloneDX JSON document from *path*."""
    p = Path(path)
    if not p.is_file():
        msg = f"Lock/SBOM path does not exist or is not a file: {p}"
        raise FileNotFoundError(msg)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Lock/SBOM is not valid JSON: {p}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Lock/SBOM root must be a JSON object: {p}"
        raise ValueError(msg)
    bom_format = data.get("bomFormat")
    if bom_format is not None and bom_format != "CycloneDX":
        msg = f"Unsupported bomFormat {bom_format!r} (expected CycloneDX): {p}"
        raise ValueError(msg)
    return data


def load_toml_lock(path: str | Path) -> dict[str, Any]:
    """Load a TOML lock document (pylock or uv.lock)."""
    p = Path(path)
    if not p.is_file():
        msg = f"Lock path does not exist or is not a file: {p}"
        raise FileNotFoundError(msg)
    toml = _require_tomllib()
    try:
        data = toml.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — tomllib raises various
        msg = f"Lock file is not valid TOML: {p}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Lock root must be a TOML table: {p}"
        raise ValueError(msg)
    return data


def load_pypi_pins(path: str | Path) -> dict[str, str]:
    """Load any supported lock/SBOM path and return a PyPI pin map.

    Raises:
        FileNotFoundError: path missing
        ValueError: unreadable / unsupported / empty-of-python when required
    """
    p = Path(path)
    if not p.is_file():
        msg = f"Lock/SBOM path does not exist or is not a file: {p}"
        raise FileNotFoundError(msg)

    text = p.read_text(encoding="utf-8")
    fmt = detect_lock_format(p, text)

    if fmt is LockFormat.CYCLONEDX:
        return pypi_pins_from_cyclonedx(load_cyclonedx(p))

    if fmt is LockFormat.PYLOCK:
        doc = load_toml_lock(p)
        if "lock-version" not in doc and "packages" not in doc:
            msg = f"Not a PEP 751 pylock document (missing lock-version/packages): {p}"
            raise ValueError(msg)
        return pypi_pins_from_pylock(doc)

    if fmt is LockFormat.UV_LOCK:
        doc = load_toml_lock(p)
        if "package" not in doc and "[[package]]" not in text:
            msg = f"Not a uv.lock document (missing [[package]]): {p}"
            raise ValueError(msg)
        return pypi_pins_from_uv_lock(doc)

    # Last-chance content sniff
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return pypi_pins_from_cyclonedx(load_cyclonedx(p))
    if "[[packages]]" in text or "lock-version" in text:
        return pypi_pins_from_pylock(load_toml_lock(p))
    if "[[package]]" in text:
        return pypi_pins_from_uv_lock(load_toml_lock(p))

    msg = (
        f"Unrecognized lock/SBOM format for {p}. "
        "Supported: uv.lock, pylock.toml / pylock.*.toml, CycloneDX JSON."
    )
    raise ValueError(msg)


def pins_to_constraint_lines(pins: dict[str, str]) -> list[str]:
    """Render pin map as pip/uv constraint lines (``name==version``)."""
    return [f"{name}=={pins[name]}" for name in sorted(pins)]


def apply_pin_to_spec(pip_spec: str, pins: dict[str, str]) -> str:
    """If *pip_spec*'s package is in *pins*, return ``name==version``."""
    if not pins:
        return pip_spec
    raw_name = package_name_from_spec(pip_spec)
    key = normalize_pypi_name(raw_name)
    if key not in pins:
        return pip_spec
    return f"{raw_name}=={pins[key]}"


def pins_from_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Read pin map from ``RGPKGS_LOCK_PINS`` / legacy pin envs."""
    import os

    environ = env if env is not None else os.environ
    raw = (
        environ.get(PINS_ENV, "").strip()
        or environ.get(PINS_ENV_LEGACY, "").strip()
        or environ.get(SBOM_PINS_ENV, "").strip()
    )
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if k is None or v is None:
            continue
        out[normalize_pypi_name(str(k))] = str(v)
    return out
