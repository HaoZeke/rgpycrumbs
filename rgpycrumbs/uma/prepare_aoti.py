#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.1",
#   "numpy>=1.24",
#   "rgpycrumbs",
# ]
# ///
"""Look up or mint a merge_mole UMA AOTI package for one system.

Cache key is reduced composition + charge + spin + task. Export is
delegated to rgpot ``scripts/export_uma_aoti.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from rgpycrumbs.uma._cache import find_package, write_sidecar
from rgpycrumbs.uma._key import UmaMoleKey, mole_key
from rgpycrumbs.uma._xyz import read_xyz, write_xyz


def default_cache_dir() -> Path:
    override = os.environ.get("RGPYCRUMBS_UMA_CACHE", "").strip()
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "rgpycrumbs" / "uma"


def resolve_exporter(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``export_uma_aoti.py``.

    Order: ``explicit``, ``RGPOT_EXPORT_UMA``, ``PATH``, then walk parents
    of cwd for ``scripts/export_uma_aoti.py``.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"exporter not found: {path}")
        return path.resolve()
    env = os.environ.get("RGPOT_EXPORT_UMA", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path.resolve()
    which = shutil.which("export_uma_aoti.py")
    if which:
        return Path(which).resolve()
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        cand = parent / "scripts" / "export_uma_aoti.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "export_uma_aoti.py not found. Set RGPOT_EXPORT_UMA or pass --exporter "
        "to the rgpot scripts/export_uma_aoti.py path."
    )


def prepare_uma_aoti(
    atomic_numbers: list[int],
    *,
    charge: int = 0,
    spin: int = 1,
    task: str = "omol",
    cache_dir: Path | None = None,
    atoms_path: Path | None = None,
    positions: list[tuple[float, float, float]] | None = None,
    exporter: Path | None = None,
    model: str = "uma-s-1p1",
    force_export: bool = False,
    python: str | None = None,
) -> Path:
    """Return a ``.pt2`` for this merge_mole key, exporting only on a miss."""
    key = mole_key(atomic_numbers, charge=charge, spin=spin, task=task)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    if not force_export:
        hit = find_package(cache, key)
        if hit is not None:
            return hit
    dest = cache / f"{key.slug()}.pt2"
    script = resolve_exporter(exporter)
    src = Path(atoms_path) if atoms_path is not None else None
    tmp: Path | None = None
    if src is None:
        if positions is None:
            raise ValueError("atoms_path or positions is required to export")
        tmp = Path(tempfile.mkdtemp(prefix="rgpycrumbs-uma-")) / "system.xyz"
        write_xyz(tmp, atomic_numbers, positions)
        src = tmp
    cmd = [
        python or sys.executable,
        str(script),
        "--atoms",
        str(src),
        "--charge",
        str(int(charge)),
        "--spin",
        str(int(spin)),
        "--task",
        str(task),
        "--label",
        key.slug(),
        "--model",
        str(model),
        "--out",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    if not dest.is_file():
        raise RuntimeError(f"exporter did not write {dest}")
    write_sidecar(dest, key, model=model, source=str(src))
    return dest


@click.command()
@click.argument("atoms", type=click.Path(exists=True, path_type=Path))
@click.option("--charge", type=int, default=0, show_default=True)
@click.option("--spin", type=int, default=1, show_default=True)
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
    charge: int,
    spin: int,
    task: str,
    model: str,
    cache: Path | None,
    exporter: Path | None,
    force_export: bool,
) -> None:
    """Look up or mint a UMA AOTI .pt2 for ATOMS (.xyz)."""
    if atoms.suffix.lower() != ".xyz":
        raise click.ClickException("prepare-aoti reads XYZ; convert .con first")
    numbers, positions, _comment = read_xyz(atoms)
    key = mole_key(numbers, charge=charge, spin=spin, task=task)
    path = prepare_uma_aoti(
        numbers,
        charge=charge,
        spin=spin,
        task=task,
        cache_dir=cache,
        atoms_path=atoms,
        positions=positions,
        exporter=exporter,
        model=model,
        force_export=force_export,
    )
    click.echo(f"key={key.slug()}")
    click.echo(str(path))


if __name__ == "__main__":
    main()
