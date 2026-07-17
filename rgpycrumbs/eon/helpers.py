# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Shared helpers for eOn job-config authorship.

Cookbook context:
https://atomistic-cookbook.org/examples/eon-pet-neb/eon-pet-neb.html

.. versionchanged:: 1.9.x
    ``write_eon_config`` routes through ``eon-schema`` INI helpers (no eon-akmc).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_eon_config(
    path: str | Path,
    settings: Mapping[str, Mapping[str, Any]],
    *,
    validate: bool = False,
) -> Path:
    """Write a ``config.ini`` for eOn from a nested section dict.

    Uses :func:`eon_schema.config.write_ini` (case-preserving keys, lowercase
    bools). Optional *validate* flags unknown keys on L0-covered sections via
    :func:`eon_schema.config.unknown_ini_keys` (pot-specific sections such as
    ``SocketNWChemPot`` are not flagged).

    Parameters
    ----------
    path:
        File path or directory (writes ``config.ini`` inside a directory).
    settings:
        ``{section: {key: value}}`` map.
    validate:
        If True, raise :class:`ValueError` when covered L0 sections contain
        unknown option names.

    Returns
    -------
    pathlib.Path
        Path to the written ``config.ini``.

    .. versionadded:: 0.1.0
    .. versionchanged:: 1.9.x
        Implemented with eon-schema; added *validate*.
    """
    try:
        from eon_schema.config import unknown_ini_keys, write_ini
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "write_eon_config requires eon-schema>=0.2.\n"
            "  pip install 'rgpycrumbs[eon]'\n"
            "  # or: pip install 'eon-schema>=0.2'\n"
            "  # uv: uv pip install 'eon-schema>=0.2'"
        ) from exc

    if validate:
        bad = unknown_ini_keys(settings, covered_only=True)
        if bad:
            raise ValueError(
                f"unknown INI keys for covered L0 sections: {bad}"
            )

    out = write_ini(path, settings)
    print(f"Wrote eOn config to '{out}'")
    return out
