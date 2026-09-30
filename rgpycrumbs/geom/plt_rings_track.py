#!/usr/bin/env python3
"""Track a query point across primitive rings.

.. versionadded:: 1.12.0

Each frame is labelled by ``chemparseplot.plot.rings``. The label is the
nearest ring centroid or junction midpoint. A block hop is a change of the
terminal ring, and that ring is its atom set. A frame spent on a bridge
keeps the terminal ring. The query coordinate is the caller's Wannier
centre. It is not replaced by the centroid.

Prefer the dispatcher::

    rgpycrumbs geom plt-rings-track molecule.sdf centres.xyz
    rgpycrumbs geom plt-rings-track molecule.xyz centres.xyz --cutoff 1.95

One molecule frame is reused for every centre frame. Several molecule
frames pair with the centre frames one to one. An XYZ molecule has no
bonds, so it needs ``--cutoff`` inside the bond/nonbonded gap. 3.5
angstrom is the ice neighbour list.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
#   "numpy>=1.26",
#   "matplotlib>=3.8.2",
#   "pydseamslib>=2.9.1",
#   "chemparseplot>=1.11",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click
from chemparseplot.plot.rings import (
    format_trajectory,
    load_trajectory,
    write_trajectory_csv,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs geom plt-rings-track")


@click.command()
@click.argument(
    "molecule",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "queries",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Heavy-atom cutoff in angstrom. Required for XYZ, refused for SDF.",
)
@click.option(
    "--max-depth",
    type=int,
    default=6,
    show_default=True,
    help="Largest ring ringNetwork generates.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Per-frame labels, one row per query.",
)
def main(
    molecule: Path,
    queries: Path,
    *,
    cutoff: float | None,
    max_depth: int,
    csv_path: Path | None,
) -> None:
    """Count terminal-ring hops for centres stored in an XYZ trajectory."""
    track = load_trajectory(molecule, queries, cutoff=cutoff, max_depth=max_depth)
    if csv_path is not None:
        write_trajectory_csv(csv_path, track)
    click.echo(format_trajectory(track), nl=False)


if __name__ == "__main__":
    main()
