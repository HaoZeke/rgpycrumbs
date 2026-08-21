# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Library prepare path. The Click CLI lives in prepare_aoti.py."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rgpycrumbs.uma._cache import find_package, write_sidecar
from rgpycrumbs.uma._key import mole_key


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
    if atoms_path is None:
        raise ValueError(
            "cache miss: pass atoms_path to a structure ASE/readcon can read"
        )
    dest = cache / f"{key.slug()}.pt2"
    script = resolve_exporter(exporter)
    cmd = [
        python or sys.executable,
        str(script),
        "--atoms",
        str(Path(atoms_path)),
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
    write_sidecar(dest, key, model=model, source=str(atoms_path))
    return dest
