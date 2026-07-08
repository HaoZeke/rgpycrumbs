# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Optional CycloneDX SBOM consumption for pin-constrained installs.

rgpycrumbs does **not** generate SBOMs (use eb-stack or similar). When a path is
provided via ``RGPYCRUMBS_SBOM`` / ``--sbom``, the dispatcher and
:func:`rgpycrumbs._aux.ensure_import` honor PyPI components as ``name==version``
pins. Non-PyPI entries (e.g. eb-stack ``pkg:generic/...`` EasyBuild inventory)
are skipped.

.. versionadded:: 1.9.13
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Environment: path to CycloneDX JSON (set by user / CLI).
SBOM_PATH_ENV = "RGPYCRUMBS_SBOM"
# Environment: JSON object {normalized_name: version} injected by dispatch for
# child processes (ensure_import / AUTO_DEPS).
SBOM_PINS_ENV = "RGPYCRUMBS_SBOM_PINS"

_VERSION_OPS = re.compile(r"(==|>=|<=|~=|!=|>|<)")


def normalize_pypi_name(name: str) -> str:
    """PEP 503-ish name normalization for pin lookups."""
    return re.sub(r"[-_.]+", "-", name).lower()


def package_name_from_spec(pip_spec: str) -> str:
    """Return the distribution name from a pip requirement string."""
    spec = pip_spec.strip()
    # drop extras: name[extra]>=1
    base = spec.split("[", maxsplit=1)[0]
    parts = _VERSION_OPS.split(base, maxsplit=1)
    return parts[0].strip()


def _pin_from_purl(purl: str) -> tuple[str, str] | None:
    """Parse ``pkg:pypi/name@version`` (optional qualifiers after ``?``)."""
    if not purl.startswith("pkg:pypi/"):
        return None
    rest = purl[len("pkg:pypi/") :]
    rest = rest.split("?", maxsplit=1)[0]
    if "@" not in rest:
        return None
    name, version = rest.rsplit("@", maxsplit=1)
    name = name.strip()
    version = version.strip()
    if not name or not version:
        return None
    # purl may percent-encode; keep simple decode for common cases
    name = name.replace("%2D", "-").replace("%5F", "_")
    return name, version


def pypi_pins_from_cyclonedx(doc: dict[str, Any]) -> dict[str, str]:
    """Extract ``{normalized_name: version}`` for PyPI components only.

    Accepts:
    - ``purl`` starting with ``pkg:pypi/``
    - or ``type`` in {library, pypi} with ``name`` + ``version`` when purl is
      absent but a ``properties`` entry marks ``pypi:package`` / bom-ref
      ``pkg:pypi/...``

    Skips ``pkg:generic/...`` and other non-PyPI purls (eb-stack EasyBuild
    inventory). Components without a usable version are skipped, not errors.
    Empty result after filtering is allowed (caller decides policy).
    """
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
            # non-pypi purl → skip (do not fall through to name-only for generic)
            if purl.startswith("pkg:") and not purl.startswith("pkg:pypi/"):
                continue

        # bom-ref may carry pkg:pypi/...
        bom_ref = comp.get("bom-ref") or ""
        if isinstance(bom_ref, str) and bom_ref.startswith("pkg:pypi/"):
            parsed = _pin_from_purl(bom_ref)
            if parsed is not None:
                name, version = parsed
                pins[normalize_pypi_name(name)] = version
                continue

        # Explicit pypi-typed component without purl
        ctype = (comp.get("type") or "").lower()
        name = comp.get("name")
        version = comp.get("version")
        if ctype in {"pypi", "library"} and name and version:
            # Only accept name/version without purl when marked as pypi via property
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


def load_cyclonedx(path: str | Path) -> dict[str, Any]:
    """Load and validate a CycloneDX JSON document from *path*.

    Raises:
        FileNotFoundError: path does not exist
        ValueError: not JSON object / not CycloneDX-shaped
    """
    p = Path(path)
    if not p.is_file():
        msg = f"SBOM path does not exist or is not a file: {p}"
        raise FileNotFoundError(msg)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"SBOM is not valid JSON: {p}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"SBOM root must be a JSON object: {p}"
        raise ValueError(msg)
    bom_format = data.get("bomFormat")
    if bom_format is not None and bom_format != "CycloneDX":
        msg = f"Unsupported bomFormat {bom_format!r} (expected CycloneDX): {p}"
        raise ValueError(msg)
    return data


def load_pypi_pins(path: str | Path) -> dict[str, str]:
    """Load CycloneDX from *path* and return PyPI pin map."""
    return pypi_pins_from_cyclonedx(load_cyclonedx(path))


def pins_to_constraint_lines(pins: dict[str, str]) -> list[str]:
    """Render pin map as pip constraint lines (``name==version``)."""
    lines = []
    for name in sorted(pins):
        lines.append(f"{name}=={pins[name]}")
    return lines


def apply_pin_to_spec(pip_spec: str, pins: dict[str, str]) -> str:
    """If *pip_spec*'s package is in *pins*, return ``name==version``.

    Extras on the original spec are dropped when pinning (pin is exact wheel
    identity). Unknown packages pass through unchanged.
    """
    if not pins:
        return pip_spec
    raw_name = package_name_from_spec(pip_spec)
    key = normalize_pypi_name(raw_name)
    if key not in pins:
        return pip_spec
    return f"{raw_name}=={pins[key]}"


def pins_from_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Read ``RGPYCRUMBS_SBOM_PINS`` JSON map from *env* (default ``os.environ``)."""
    import os

    environ = env if env is not None else os.environ
    raw = environ.get(SBOM_PINS_ENV, "").strip()
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
