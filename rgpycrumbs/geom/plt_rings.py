#!/usr/bin/env python3
"""Draw primitive rings and the bridges between them.

.. versionadded:: 1.11.0

The picture is ``chemparseplot.plot.rings``: one xyzrender hull per face
``pydseams.yoda.ringNetwork`` returns, and a dashed stroke on each bridge
whose ends both lie on a returned ring. Query atoms, when given, stay where
the file puts them. They are a label for a Wannier centre, not the coordinate
that enters a mean-square displacement.

Prefer the dispatcher::

    rgpycrumbs geom plt-rings molecule.sdf rings.svg
    rgpycrumbs geom plt-rings molecule.xyz rings.svg --cutoff 1.95

``uv run --script`` on this file is the same isolated environment the
dispatcher builds. An XYZ file has no bonds, so it needs ``--cutoff`` inside
the gap between the longest bond and the shortest nonbonded contact. 3.5
angstrom is the ice neighbour list and is the wrong graph for a thiophene.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
#   "numpy>=1.26",
#   "matplotlib>=3.8.2",
#   "pydseamslib>=2.9.1",
#   "xyzrender>=0.3.8",
#   "chemparseplot>=1.11",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click
from chemparseplot.plot.rings import (
    format_report,
    read_structure,
    render_primitive_rings,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs geom plt-rings")


@click.command()
@click.argument(
    "structure",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
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
    "--queries",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="XYZ of query points, drawn where they sit.",
)
@click.option("--config", default="paton", show_default=True, help="xyzrender preset.")
@click.option("--canvas-size", type=int, default=900, show_default=True)
def main(
    structure: Path,
    output: Path,
    *,
    cutoff: float | None,
    max_depth: int,
    queries: Path | None,
    config: str,
    canvas_size: int,
) -> None:
    """Render primitive rings for an SDF or an XYZ file."""
    symbols, coords, bonds = read_structure(structure)
    query_coords = None
    if queries is not None:
        _qsym, query_coords, _qbonds = read_structure(queries)
    if bonds is not None and cutoff is not None:
        raise SystemExit("this file already has bonds; omit --cutoff")
    if bonds is None and cutoff is None:
        raise SystemExit("XYZ has no bonds; pass --cutoff inside the bond/nonbonded gap")
    report, assignments = render_primitive_rings(
        symbols,
        coords,
        output,
        bonds=bonds,
        cutoff=cutoff,
        max_depth=max_depth,
        queries=query_coords,
        config=config,
        canvas_size=canvas_size,
    )
    click.echo(format_report(report, assignments))


if __name__ == "__main__":
    main()
