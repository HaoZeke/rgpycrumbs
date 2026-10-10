#!/usr/bin/env python3
"""Draw where the wall of one layout goes, per cell, with the Amdahl bound.

.. versionadded:: 1.13.0

TABLE is a tidy CSV with one row per run and component (columns cell,
ranks, threads, repetition, component, seconds); the component named by
``--total`` is the whole wall and the others are non-nesting parts of it.
The figure stacks the median seconds of each component at ``--layout`` for
every cell (the rest of the total as ``unattributed``) and, beside it, the
bound 1/(1 - f) on the whole-run speedup if a component with share f took
no time. The file is ``<prefix>-<layout>-components``::

    rgpycrumbs surrogate plt-components components.csv -o figs/where --layout 4x2 --cell 16_oxirane --cell 11_grignard
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@5daf95d7741b2b5b5e1bd5b5b119ba221fa5b369",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate.gpr_optim import parse_components_csv
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import (
    plot_component_breakdown,
    set_font,
    set_legend_fontsize,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-components")


@click.command()
@click.argument("table", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; '-<layout>-components' and the format are appended.",
)
@click.option(
    "--layout",
    required=True,
    help="ranks x threads layout to draw, e.g. 4x2.",
)
@click.option(
    "--cell",
    "cells",
    multiple=True,
    help="Cell to draw (repeatable, in row order). Default: every cell of TABLE.",
)
@click.option(
    "--total",
    default="total",
    show_default=True,
    help="Component name holding the whole wall of a run.",
)
@click.option("--title", default=None, help="Figure title (default: the layout).")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option(
    "--legend-fontsize",
    type=float,
    default=None,
    help="Legend font size in points (default: each figure's own, 8 or 9).",
)
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
def main(
    *,
    table,
    prefix,
    layout,
    cells,
    total,
    title,
    fmt,
    dpi,
    font_dirs,
    font,
    legend_fontsize,
):
    """Draw the component breakdown of LAYOUT for the cells of TABLE."""
    set_font(font, font_dirs)
    set_legend_fontsize(legend_fontsize)
    try:
        t = parse_components_csv(table, total=total, cells=list(cells) or None)
        fig = plot_component_breakdown(t, layout, title=title)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    command = " ".join(["rgpycrumbs", "surrogate", "plt-components", *sys.argv[1:]])
    out = prefix.with_name(f"{prefix.name}-{layout}-components.{fmt}")
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
