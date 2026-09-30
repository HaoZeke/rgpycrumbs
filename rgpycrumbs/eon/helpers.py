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

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
            raise ValueError(f"unknown INI keys for covered L0 sections: {bad}")

    out = write_ini(path, settings)
    print(f"Wrote eOn config to '{out}'")
    return out


log = logging.getLogger(__name__)


#: neb.con frame metadata keys, in the column order the legacy neb_*.dat had.
_PROFILE_KEYS = ("reaction_coordinate", "relative_energy", "parallel_force")


def load_profile_rows(path: Path, mass_weighted: bool = False):
    """One NEB step as rows ``[image, rc, energy, parallel_force, eigenvalue]``.

    A ``.con`` band (``neb.con``, ``neb_path_NNN.con``) is read through readcon
    from its frame metadata, where eOn writes the profile, so no column order
    is assumed. With ``mass_weighted`` the coordinate is the frames'
    ``reaction_coordinate_mw``. A legacy ``neb_*.dat`` is read by column, and
    has no mass-weighted coordinate.
    """
    import numpy as np

    path = Path(path)
    if path.suffix != ".con":
        if mass_weighted:
            log.warning(
                "%s has no mass-weighted coordinate; using the path one", path.name
            )
        return np.loadtxt(path, skiprows=1).T
    from readcon import read_con

    rows = []
    for i, frame in enumerate(read_con(str(path))):
        md = frame.metadata
        missing = [k for k in _PROFILE_KEYS if k not in md]
        if missing:
            msg = f"{path.name} frame {i} lacks {', '.join(missing)}"
            raise ValueError(msg)
        rc_key = "reaction_coordinate_mw" if mass_weighted else "reaction_coordinate"
        if rc_key not in md:
            msg = f"{path.name} frame {i} lacks {rc_key}; eOn writes it from 3.4"
            raise ValueError(msg)
        rows.append(
            [
                float(md.get("neb_bead", i)),
                float(md[rc_key]),
                float(md["relative_energy"]),
                float(md["parallel_force"]),
                float(md.get("lowest_eigenvalue", np.nan)),
            ]
        )
    return np.asarray(rows).T
