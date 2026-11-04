"""Draw the energy profile and the landscape of an eOn NEB run with plt-neb.

    python scripts/plot_neb.py [RUN_DIR]

RUN_DIR defaults to the Diels-Alder tutorial run of the rgpycrumbs
repository. The plot stack must be importable here (route 2 in
SKILL.md), so the dispatcher runs plt-neb in this interpreter. Both
figures go to a temporary directory and are checked to be PNG files.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

STACK = ("readcon", "matplotlib", "polars", "ase", "chemparseplot")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TUTORIAL = Path("docs/orgmode/tutorials/data/diels_alder")


def find_tutorial() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / TUTORIAL).is_dir():
            return parent / TUTORIAL
    return None


def run() -> None:
    missing = [name for name in STACK if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(
            "the plot stack is not installed here (missing: "
            + ", ".join(missing)
            + "); see 'Install the plotting stack' in SKILL.md"
        )
    where = find_tutorial() if len(sys.argv) < 2 else Path(sys.argv[1])
    if where is None:
        sys.exit("usage: python scripts/plot_neb.py RUN_DIR")
    where = where.resolve()
    inputs = [
        "--input-dat-pattern",
        str(where / "neb_*.dat"),
        "--input-path-pattern",
        str(where / "neb_path_*.con"),
        "--con-file",
        str(where / "neb.con"),
    ]
    if (where / "sp.con").is_file():
        inputs += ["--sp-file", str(where / "sp.con")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        jobs = {
            "profile.png": ["--plot-type", "profile"],
            "landscape.png": [
                "--plot-type",
                "landscape",
                "--landscape-mode",
                "path",
                "--no-project-path",
                "--cache-file",
                str(out / "landscape.parquet"),
            ],
        }
        for name, options in jobs.items():
            target = out / name
            command = [sys.executable, "-m", "rgpycrumbs.cli", "eon", "plt-neb"]
            command += [*options, *inputs, "-o", str(target)]
            proc = subprocess.run(  # noqa: S603
                command, cwd=out, capture_output=True, text=True, check=False
            )
            if proc.returncode != 0:
                print(proc.stdout + proc.stderr)
                sys.exit(f"plt-neb for {name} exited with status {proc.returncode}")
            if "Failed to process step" in " ".join((proc.stdout + proc.stderr).split()):
                print(proc.stdout + proc.stderr)
                sys.exit(f"plt-neb for {name} left out a step of the band")
            assert "--> Using active interpreter" in proc.stderr, proc.stderr
            assert target.read_bytes()[:8] == PNG_SIGNATURE, name
            print(f"{name}: {target.stat().st_size} bytes")
    for name in ("chemparseplot", "readcon", "matplotlib", "jax", "polars", "ase"):
        print(f"{name} {importlib.metadata.version(name)}")


if __name__ == "__main__":
    run()
