#!/usr/bin/env python3
"""Draw fixed-work speedup and the POP efficiency factors against cores.

.. versionadded:: 1.13.0

TABLE is a CSV with one row per fixed-work run (columns ranks, threads, a
wall column named by ``--time``, one column per factor named by ``--metric``
and optionally cell). Each ``--metric COLUMN[=LABEL]`` is drawn as one line
on a 0 to 1 axis, with the fixed-work efficiency (speedup per core ratio
against the layout with the fewest cores) and the ``--guide`` level. Layouts
sharing a core count are dodged; there is no ideal line. One figure per
cell, ``<prefix>-<cell>-pop`` (``<prefix>-pop`` for a table without cells)::

    rgpycrumbs surrogate plt-pop pop_talp.csv -o figs/pop --metric parallel_eff="parallel efficiency" --metric load_balance="load balance" --metric comm_eff="communication efficiency"
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@738369598da9670ef53f35501f8038ffec657232",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate.gpr_optim import parse_pop_factors, pop_csv_cells
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import (
    plot_pop_factors,
    set_font,
    set_legend_fontsize,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-pop")


def _metric(spec: str) -> tuple[str, str]:
    column, _, label = spec.partition("=")
    if not column:
        raise click.BadParameter(f"{spec!r}: expected COLUMN or COLUMN=LABEL")
    return column, label or column


@click.command()
@click.argument("table", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; '-<cell>-pop' (or '-pop') and the format are appended.",
)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    required=True,
    help="Factor column of TABLE as COLUMN or COLUMN=LABEL (repeatable, in legend order).",
)
@click.option(
    "--time",
    "time_column",
    default="elapsed_s",
    show_default=True,
    help="Column of TABLE holding the wall of each run in seconds.",
)
@click.option(
    "--cell",
    "cells",
    multiple=True,
    help="Cell to draw (repeatable). Default: every cell of TABLE.",
)
@click.option(
    "--guide",
    type=float,
    default=0.8,
    show_default=True,
    help="Efficiency level drawn as a dotted guide; a negative value draws none.",
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
    metrics,
    time_column,
    cells,
    guide,
    fmt,
    dpi,
    font_dirs,
    font,
    legend_fontsize,
):
    """Draw the fixed-work efficiency figure of each cell in TABLE."""
    set_font(font, font_dirs)
    set_legend_fontsize(legend_fontsize)
    labels = dict(_metric(m) for m in metrics)
    have = pop_csv_cells(table)
    if have is None:
        targets = [None]
    else:
        targets = list(cells) or have
        absent = [c for c in targets if c not in have]
        if absent:
            raise click.ClickException(
                f"{table} has no rows of cell(s) {absent}; its cells are {have}"
            )
    command = " ".join(["rgpycrumbs", "surrogate", "plt-pop", *sys.argv[1:]])
    for cell in targets:
        try:
            t = parse_pop_factors(table, labels, time=time_column, cell=cell)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        fig = plot_pop_factors(t, guide=None if guide < 0 else guide, title=cell)
        name = f"{prefix.name}-{cell}-pop" if cell else f"{prefix.name}-pop"
        out = prefix.with_name(f"{name}.{fmt}")
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
