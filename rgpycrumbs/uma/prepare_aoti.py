#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Look up or mint a UMA AOTI package for one system.

Cache key is exact composition + charge + spin + task. Export is
delegated to rgpot ``scripts/export_uma_aoti.py``. Structures are read
with ASE (or readcon for ``.con``) via the PEP 723 ``uv run`` env.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "ase",
#   "readcon>=0.14.5",
#   "rgpycrumbs",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs uma prepare-aoti")

from rgpycrumbs.uma._io import atomic_numbers_of, charge_spin_of, load_atoms
from rgpycrumbs.uma._key import mole_key
from rgpycrumbs.uma._prepare import prepare_uma_aoti


@click.command()
@click.argument("atoms", type=click.Path(exists=True, path_type=Path))
@click.option("--charge", type=int, default=None, help="Override atoms.info charge.")
@click.option("--spin", type=int, default=None, help="Override atoms.info spin.")
@click.option("--task", default="omol", show_default=True)
@click.option("--model", default="uma-s-1p1", show_default=True)
@click.option(
    "--cache",
    type=click.Path(path_type=Path),
    default=None,
    help="Package cache (default: $RGPYCRUMBS_UMA_CACHE or XDG cache).",
)
@click.option(
    "--exporter",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to rgpot scripts/export_uma_aoti.py",
)
@click.option("--force-export", is_flag=True, help="Ignore an existing cache hit.")
def main(
    atoms: Path,
    charge: int | None,
    spin: int | None,
    task: str,
    model: str,
    cache: Path | None,
    exporter: Path | None,
    force_export: bool,
) -> None:
    """Look up or mint a UMA AOTI .pt2 for ATOMS (ASE or CON)."""
    loaded = load_atoms(atoms)
    numbers = atomic_numbers_of(loaded)
    q, s = charge_spin_of(loaded, charge=charge, spin=spin)
    key = mole_key(numbers, charge=q, spin=s, task=task)
    path = prepare_uma_aoti(
        numbers,
        charge=q,
        spin=s,
        task=task,
        cache_dir=cache,
        atoms_path=atoms,
        exporter=exporter,
        model=model,
        force_export=force_export,
    )
    click.echo(f"key={key.slug()}")
    click.echo(str(path))


if __name__ == "__main__":
    main()
