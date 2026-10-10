#!/usr/bin/env python3
"""Draw fixed-work speedup and the POP efficiency factors against cores.

.. versionadded:: 1.13.0

Each TABLE is a CSV with one row per fixed-work run (columns ranks, threads,
a wall column named by ``--time``, one column per factor named by
``--metric`` and optionally cell). Each ``--metric COLUMN[=LABEL]`` is drawn
as one line on a 0 to 1 axis, with the fixed-work efficiency (speedup per
core ratio against the layout with the fewest cores) and the ``--guide``
level. ``--time COLUMN[=QUANTITY]`` names the timed column of the TABLEs in
order (one value serves every table); the quantity titles the row. Layouts
sharing a core count are dodged; there is no ideal line. One figure,
``<prefix>-pop``, with one row per table and cell::

    rgpycrumbs surrogate plt-pop replay.csv fit.csv --time elapsed_s="process wall" --time fit_s=fit -o figs/pop --metric parallel_eff="parallel efficiency" --metric load_balance="load balance" --metric comm_eff="communication efficiency"
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@edd3752ee17dbc576f0d30ded2cafcd99547e9bf",
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
@click.argument(
    "tables",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; '-pop' and the format are appended.",
)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    required=True,
    help="Factor column as COLUMN or COLUMN=LABEL (repeatable, in legend order).",
)
@click.option(
    "--time",
    "times",
    multiple=True,
    default=("elapsed_s=process wall",),
    show_default=True,
    help="Timed column per TABLE as COLUMN or COLUMN=QUANTITY (repeatable, in TABLE order; one value serves all).",
)
@click.option(
    "--cell",
    "cells",
    multiple=True,
    help="Cell to draw (repeatable). Default: every cell of every TABLE.",
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
    tables,
    prefix,
    metrics,
    times,
    cells,
    guide,
    fmt,
    dpi,
    font_dirs,
    font,
    legend_fontsize,
):
    """Draw the fixed-work efficiency figure of the TABLEs."""
    set_font(font, font_dirs)
    set_legend_fontsize(legend_fontsize)
    labels = dict(_metric(m) for m in metrics)
    if len(times) == 1:
        times = times * len(tables)
    if len(times) != len(tables):
        raise click.ClickException(
            f"{len(times)} --time values for {len(tables)} tables; give one or one per table"
        )
    series = []
    seen: set[str] = set()
    for table, spec in zip(tables, times, strict=True):
        column, _, quantity = spec.partition("=")
        have = pop_csv_cells(table)
        targets = [None] if have is None else [c for c in have if not cells or c in cells]
        seen.update(have or [])
        for cell in targets:
            try:
                series.append(
                    parse_pop_factors(
                        table,
                        labels,
                        time=column,
                        quantity=quantity or column,
                        cell=cell,
                    )
                )
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
    absent = [c for c in cells if c not in seen]
    if absent:
        raise click.ClickException(
            f"no table has rows of cell(s) {absent}; the tables hold {sorted(seen)}"
        )
    if not series:
        raise click.ClickException("no series to draw")
    fig = plot_pop_factors(series, guide=None if guide < 0 else guide)
    command = " ".join(["rgpycrumbs", "surrogate", "plt-pop", *sys.argv[1:]])
    out = prefix.with_name(f"{prefix.name}-pop.{fmt}")
    save_with_provenance(
        fig,
        out,
        {},
        command=command,
        hashes={t.name: file_sha256(t) for t in tables},
        dpi=dpi,
    )
    click.echo(out)


if __name__ == "__main__":
    main()
