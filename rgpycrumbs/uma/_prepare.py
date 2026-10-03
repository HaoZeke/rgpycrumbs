# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Look up a UMA AOTI package, or export it on a miss.

The Click CLI lives in ``prepare_aoti.py``; this is the library path.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rgpycrumbs.uma._cache import (
    default_cache_dir,
    embedded_metadata,
    embedded_mismatches,
    install,
    key_lock,
    lookup,
    package_path,
    partial_dir,
)
from rgpycrumbs.uma._env import normalize_device, probe_env
from rgpycrumbs.uma._key import AotiKey, ExportEnv, aoti_key

#: Exit codes of rgpot ``scripts/export_uma_aoti.py`` and what they mean.
EXPORTER_EXIT = {
    2: "the eager tensor wrapper does not match the ASE reference",
    3: "the torch.export module does not match the ASE reference",
    4: "the compiled AOTI package does not match the ASE reference",
}


def resolve_exporter(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate rgpot ``scripts/export_uma_aoti.py``.

    Order: ``explicit``, ``$RGPOT_EXPORT_UMA``, ``PATH``, then
    ``scripts/export_uma_aoti.py`` in the working directory or a parent
    (an rgpot checkout).
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"exporter not found: {path}")
        return path.resolve()
    env = os.environ.get("RGPOT_EXPORT_UMA", "").strip()
    if env:
        path = Path(env)
        if not path.is_file():
            raise FileNotFoundError(f"RGPOT_EXPORT_UMA names no file: {path}")
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
        "export_uma_aoti.py not found: pass --exporter or set RGPOT_EXPORT_UMA "
        "to rgpot's scripts/export_uma_aoti.py"
    )


@dataclass(frozen=True)
class Prepared:
    """Outcome of :func:`prepare_uma_aoti`."""

    path: Path
    key: AotiKey
    hit: bool
    #: Wall time of the exporter run; None on a hit or a dry run.
    compile_seconds: float | None = None
    #: Exporter command line; set on a miss (also in a dry run).
    command: list[str] = field(default_factory=list)


def exporter_command(
    key: AotiKey,
    *,
    python: str,
    exporter: Path,
    atoms_path: Path,
    device: str,
    out: Path,
) -> list[str]:
    cmd = [
        python,
        str(exporter),
        "--atoms",
        str(atoms_path),
        "--charge",
        str(key.charge),
        "--spin",
        str(key.spin),
        "--task",
        key.task,
        "--model",
        key.model,
        "--device",
        device,
        "--label",
        key.stem(),
        "--out",
        str(out),
    ]
    if key.molecular_box != "0":
        cmd += ["--molecular-box", key.molecular_box]
    if key.batch_max:
        cmd += ["--batch-max", str(key.batch_max)]
    return cmd


def prepare_uma_aoti(
    atomic_numbers: Iterable[int],
    *,
    charge: int = 0,
    spin: int | None = None,
    task: str = "omol",
    model: str = "uma-s-1p1",
    device: str = "cpu",
    molecular_box: float = 0.0,
    batch_max: int = 0,
    cache_dir: Path | None = None,
    atoms_path: Path | None = None,
    exporter: Path | None = None,
    python: str | None = None,
    env: ExportEnv | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Prepared:
    """Return the package for this system, exporting it on a miss.

    ``spin=None`` derives the minimal multiplicity from electron parity.
    ``python`` is the interpreter that runs the exporter (default: this
    one) and ``env`` its probed compiler side (default: probe it). A
    miss takes the key's lock, so concurrent callers for one key run
    the exporter once and the rest return its package. The finished
    package is checked against the key through its embedded metadata
    before it is installed.
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    device = normalize_device(device)
    python = python or sys.executable
    if env is None:
        env = probe_env(cache, python=python, device=device)
    key = aoti_key(
        atomic_numbers,
        env,
        charge=charge,
        spin=spin,
        task=task,
        model=model,
        molecular_box=molecular_box,
        batch_max=batch_max,
    )
    if not force:
        hit = lookup(cache, key)
        if hit is not None:
            return Prepared(hit, key, hit=True)
    if atoms_path is None:
        raise ValueError(
            "cache miss: atoms_path must name a structure the exporter reads"
        )
    script = resolve_exporter(exporter)
    dest = package_path(cache, key)
    if dry_run:
        cmd = exporter_command(
            key,
            python=python,
            exporter=script,
            atoms_path=atoms_path,
            device=device,
            out=dest,
        )
        return Prepared(dest, key, hit=False, command=cmd)

    with key_lock(cache, key.stem()):
        if not force:
            hit = lookup(cache, key)
            if hit is not None:
                return Prepared(hit, key, hit=True)
        work = partial_dir(cache, key.stem())
        try:
            built = work / dest.name
            cmd = exporter_command(
                key,
                python=python,
                exporter=script,
                atoms_path=Path(atoms_path).resolve(),
                device=device,
                out=built,
            )
            start = time.perf_counter()
            # The exporter's progress goes to stderr; stdout stays the path.
            proc = subprocess.run(cmd, stdout=sys.stderr, check=False)  # noqa: S603
            seconds = time.perf_counter() - start
            if proc.returncode != 0:
                why = EXPORTER_EXIT.get(proc.returncode, "see its output above")
                raise RuntimeError(f"exporter exited {proc.returncode}: {why}")
            if not built.is_file():
                raise RuntimeError(f"exporter exited 0 without writing {built.name}")
            meta = embedded_metadata(built)
            bad = embedded_mismatches(meta, key)
            if bad:
                raise RuntimeError(
                    "exported package does not match its key: " + "; ".join(bad)
                )
            sidecar = {
                "embedded": meta,
                "compile_seconds": round(seconds, 3),
                "created": _dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "source": str(Path(atoms_path).resolve()),
                "exporter": str(script),
                "python": python,
            }
            path = install(cache, key, built, sidecar)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return Prepared(path, key, hit=False, compile_seconds=seconds, command=cmd)
