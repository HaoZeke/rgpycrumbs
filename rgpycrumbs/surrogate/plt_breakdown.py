#!/usr/bin/env python3
"""Draw wall time per component for each cell and layout.

.. versionadded:: 1.12.0

Reads a per-repetition wall-time CSV (columns cell, ranks, threads,
repetition, stage, seconds) and stacks the median seconds of the chosen
component stages per ranks x threads layout, one figure per cell. The rest of
the ``--total`` stage is drawn as ``other``. Components must not nest (pass
``band_oracle`` and ``band_refit``, not ``band_internal`` with its parts)::

    rgpycrumbs surrogate plt-breakdown strong_scaling_wall.csv -o figs/breakdown --component band_oracle --component band_refit --component dimer_oracle
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1",
#   "numpy>=1.26",
#   "matplotlib>=3.8.2",
#   "pint>=0.22",
#   "h5py>=3.0",
#   "ase>=3.22",
#   "pandas>=2.0",
#   "cmcrameri>=1.7",
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@0bc6e14e8597444ec990b8e3a6158235097ac5a1",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from chemparseplot.parse.surrogate.gpr_optim import parse_breakdown_csv
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import plot_time_breakdown, set_font

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-breakdown")


@click.command()
@click.argument("table", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; '-<cell>-breakdown' and the format are appended.",
)
@click.option(
    "--component",
    "components",
    multiple=True,
    required=True,
    help="Stage to stack (repeatable, bottom to top). Must not nest.",
)
@click.option(
    "--total",
    default="pipeline",
    show_default=True,
    help="Stage holding the whole; 'none' to omit.",
)
@click.option(
    "--cell", "cells", multiple=True, help="Cell to draw (repeatable). Default: all."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option(
    "--font-dir",
    "font_dirs",
    multiple=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Extra directory searched for the --font faces (repeatable).",
)
@click.option(
    "--font",
    default=None,
    help="Font family for text and math text (e.g. Jost); an unresolvable family is an error. Default: the theme font.",
)
def main(*, table, prefix, components, total, cells, fmt, dpi, font_dirs, font):
    """Draw the wall-time breakdown of each cell in TABLE."""
    set_font(font, font_dirs)
    data = parse_breakdown_csv(
        table,
        list(components),
        total=None if total == "none" else total,
        cells=list(cells) or None,
    )
    if not data:
        raise click.ClickException(f"no rows of the requested stages in {table}")
    for comp in components:
        if not any(np.isfinite(parts[comp]).any() for _, parts, _ in data.values()):
            raise click.ClickException(f"stage {comp!r} has no rows in {table}")
    command = " ".join(["rgpycrumbs", "surrogate", "plt-breakdown", *sys.argv[1:]])
    for cell, (layouts, parts, tot) in data.items():
        fig = plot_time_breakdown(layouts, parts, tot, title=cell)
        out = prefix.with_name(f"{prefix.name}-{cell}-breakdown.{fmt}")
        save_with_provenance(
            fig,
            out,
            {},
            command=command,
            hashes={table.name: file_sha256(table)},
            dpi=dpi,
        )
        click.echo(out)


if __name__ == "__main__":
    main()
