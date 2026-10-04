#!/usr/bin/env python3
"""Draw per-case oracle calls and wall time, one panel per benchmark board.

.. versionadded:: 1.12.0

TABLE is a tidy CSV with one row per case: ``board, case, label,
search_calls, validation_calls, pipeline_wall_s, certified`` and optionally
``reference_pipeline_wall_s``. Boards and rows keep the order of the file.
``--baseline FILE`` adds comparison methods from a long CSV ``board, case,
method, calls, converged``. Files are ``<prefix>-calls`` (stacked search and
independent-validation bars, linear axis) and ``<prefix>-wall`` (log axis, a
tick at the reference time)::

    rgpycrumbs surrogate plt-cases cases.csv -o figs/cases --font Jost --reference-label "earlier record"
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@f0aa00d81c90ce795a8542f5fb244c828af6d092",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import attach_baselines, parse_cases_csv
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import plot_cases_calls, plot_cases_wall, set_font

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-cases")


@click.command()
@click.argument("table", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; '-calls', '-wall' and the format are appended.",
)
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Long CSV board, case, method, calls, converged.",
)
@click.option(
    "--reference-label",
    default="reference",
    show_default=True,
    help="Legend text of the reference-time tick.",
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
def main(*, table, prefix, baseline, reference_label, fmt, dpi, font_dirs, font):
    """Draw the per-case calls and wall-time figures of TABLE."""
    set_font(font, font_dirs)
    try:
        boards = parse_cases_csv(table)
        hashes = {table.name: file_sha256(table)}
        if baseline is not None:
            attach_baselines(boards, baseline)
            hashes[baseline.name] = file_sha256(baseline)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not boards:
        raise click.ClickException(f"{table} has no rows")
    command = " ".join(["rgpycrumbs", "surrogate", "plt-cases", *sys.argv[1:]])
    figs = {
        "calls": plot_cases_calls(boards),
        "wall": plot_cases_wall(boards, reference_label=reference_label),
    }
    for panel, fig in figs.items():
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=hashes, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
