# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Load structures through ASE or readcon. No local format parsers."""

from __future__ import annotations

from pathlib import Path

_CON_SUFFIXES = {".con", ".convel"}


def load_atoms(path: Path):
    """Return an ASE ``Atoms`` for *path*.

    ``.con`` / ``.convel`` go through readcon (same as ``eon con_splitter``).
    Everything else is ``ase.io.read``. Both come from the PEP 723 / ``uv run``
    environment, or ``ensure_import`` when called as a library.
    """
    from rgpycrumbs._aux import ensure_import

    path = Path(path)
    if path.suffix.lower() in _CON_SUFFIXES:
        readcon = ensure_import("readcon")
        frames = list(readcon.read_con(str(path)))
        if not frames:
            raise ValueError(f"{path} contains no CON frames")
        return frames[0].to_ase()
    ase_io = ensure_import("ase.io")
    return ase_io.read(path)


def atomic_numbers_of(atoms) -> list[int]:
    return [int(z) for z in atoms.get_atomic_numbers()]


def charge_spin_of(atoms, *, charge: int | None = None, spin: int | None = None):
    info = getattr(atoms, "info", {}) or {}
    q = int(charge) if charge is not None else int(info.get("charge", 0))
    s = int(spin) if spin is not None else int(info.get("spin", 1))
    return q, s
