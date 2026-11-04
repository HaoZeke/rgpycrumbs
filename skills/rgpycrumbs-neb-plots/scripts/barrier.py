"""Barrier and reaction energy of an eOn NEB run, from its last neb_NNN.dat.

    python scripts/barrier.py [RUN_DIR]

RUN_DIR defaults to the Diels-Alder tutorial run of the rgpycrumbs
repository. Standard library only.
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

TUTORIAL = Path("docs/orgmode/tutorials/data/diels_alder")


def find_tutorial() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / TUTORIAL).is_dir():
            return parent / TUTORIAL
    return None


def last_band(run: Path) -> tuple[Path, list[dict[str, float]]]:
    steps = sorted(run.glob("neb_[0-9]*.dat"), key=lambda p: int(p.stem[4:]))
    if not steps:
        sys.exit(f"no neb_NNN.dat in {run}")
    lines = steps[-1].read_text(encoding="utf-8").splitlines()
    names = lines[0].split()
    rows = [
        dict(zip(names, map(float, line.split()), strict=True))
        for line in lines[1:]
        if line.strip()
    ]
    return steps[-1], rows


def results(run: Path) -> dict[str, float]:
    """Numeric ``value key`` lines of results.dat, if the run has one."""
    path = run / "results.dat"
    found: dict[str, float] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                with contextlib.suppress(ValueError):
                    found[parts[1]] = float(parts[0])
    return found


def run() -> None:
    tutorial = len(sys.argv) < 2
    where = find_tutorial() if tutorial else Path(sys.argv[1])
    if where is None:
        sys.exit("usage: python scripts/barrier.py RUN_DIR")
    path, rows = last_band(where)
    energy = [row["energy"] for row in rows]
    top = max(range(len(rows)), key=energy.__getitem__)
    forward = energy[top] - energy[0]
    reverse = energy[top] - energy[-1]
    print(f"last step: {path.name} ({len(rows)} images)")
    print(
        f"highest image: {top} at rxn_coord {rows[top]['rxn_coord']:.6f} A, "
        f"energy {energy[top]:.6f} eV"
    )
    print(
        f"forward barrier {forward:.6f} eV, reverse barrier {reverse:.6f} eV, "
        f"reaction energy {energy[-1] - energy[0]:.6f} eV"
    )
    res = results(where)
    extrema = [k for k in res if re.fullmatch(r"extremum\d+_energy", k)]
    if extrema and "energy_reference" in res:
        best = max(extrema, key=res.__getitem__)
        position = res.get(best.replace("_energy", "_position"), float("nan"))
        print(
            f"results.dat: highest extremum {res[best] - res['energy_reference']:.6f} eV "
            f"above image 0, at image position {position:.6f}"
        )
    if tutorial:
        assert path.name == "neb_014.dat", path.name
        assert top == 14, top
        assert f"{forward:.6f}" == "0.787659", forward


if __name__ == "__main__":
    run()
