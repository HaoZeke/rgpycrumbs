#!/usr/bin/env python3
"""Draw strong-scaling speedup and a parallel-efficiency table.

.. versionadded:: 1.12.0

TABLE is a CSV of per-repetition wall times (columns cell, ranks, threads,
repetition, stage, seconds, calls; ``--counts`` adds the passed column) or JSON ``{"<series>": {"workers": [...], "time_s": [...]}}``. Speedup
is relative to each series' own first point, or to the first point of the
series ``--reference`` names. Files are ``<prefix>-speedup``
and ``<prefix>-efficiency``::

    rgpycrumbs surrogate plt-scaling scaling.json -o figs/scaling
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@1a6993eb7942b734425549302bb9c66e2b8eb38b",
# ]
# ///

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import ScalingTable
from chemparseplot.parse.surrogate.gpr_optim import parse_pop_csv, parse_scaling_csv
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import (
    plot_efficiency_table,
    plot_pop_efficiencies,
    plot_scaling,
    plot_strong_scaling,
    set_font,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-scaling")


@click.command()
@click.argument("table", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; the panel name and format are appended.",
)
@click.option("--reference", default=None, help="Series that defines speedup 1.")
@click.option(
    "--counts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV with a passed column per run (CSV input); false marks a capped run.",
)
@click.option(
    "--stage", default="pipeline", show_default=True, help="Stage of a CSV TABLE."
)
@click.option("--cell", "cells", multiple=True, help="Cell of a CSV TABLE (repeatable).")
@click.option(
    "--pop",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Fixed-work CSV (ranks, threads, optional cell, efficiency columns).",
)
@click.option(
    "--pop-metric",
    "pop_metrics",
    multiple=True,
    help="Column of --pop to draw as an efficiency (repeatable).",
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
def main(
    *,
    table,
    prefix,
    reference,
    counts,
    stage,
    cells,
    pop,
    pop_metrics,
    fmt,
    dpi,
    font,
    font_dirs,
):
    """Draw scaling figures from TABLE."""
    set_font(font, font_dirs)
    hashes = {table.name: file_sha256(table)}
    if counts is not None:
        hashes[counts.name] = file_sha256(counts)
    if table.suffix == ".csv":
        t = parse_scaling_csv(table, counts, stage=stage, cells=list(cells) or None)
        series = t.series
        if not series:
            raise click.ClickException(f"no rows of stage {stage!r} in {table}")
        t.reference = reference
    else:
        raw = json.loads(table.read_text())
        try:
            series = {k: (v["workers"], v["time_s"]) for k, v in raw.items()}
        except (KeyError, TypeError, AttributeError) as exc:
            raise click.ClickException(
                'TABLE must be {"series": {"workers": [...], "time_s": [...]}}'
            ) from exc
        t = ScalingTable(series, reference)
    if reference is not None and reference not in series:
        raise click.ClickException(
            f"--reference {reference!r} is not a series of {table}"
        )
    figs = {"speedup": plot_scaling(t), "efficiency": plot_efficiency_table(t)}
    if table.suffix == ".csv":
        for name in t.series:
            figs[f"{name}-strong"] = plot_strong_scaling(t, name)
    if pop is not None:
        if not pop_metrics:
            raise click.ClickException("--pop needs at least one --pop-metric COLUMN")
        hashes[pop.name] = file_sha256(pop)
        for name in list(t.series) if table.suffix == ".csv" else [None]:
            layouts, metrics = parse_pop_csv(pop, list(pop_metrics), cell=name)
            if layouts:
                figs[f"{name}-pop" if name else "pop"] = plot_pop_efficiencies(
                    layouts, metrics, title=name
                )
    command = " ".join(["rgpycrumbs", "surrogate", "plt-scaling", *sys.argv[1:]])
    for panel, fig in figs.items():
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=hashes, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
