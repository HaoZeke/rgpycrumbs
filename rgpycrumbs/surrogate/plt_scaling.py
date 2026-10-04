#!/usr/bin/env python3
"""Draw strong-scaling speedup and a parallel-efficiency table.

.. versionadded:: 1.12.0

TABLE is JSON ``{"<series>": {"workers": [...], "time_s": [...]}}``. Speedup
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@fe1c161871b3bc337a60dd18c3fc43f4340e5a52",
# ]
# ///

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import ScalingTable
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import plot_efficiency_table, plot_scaling

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
    "--format",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
def main(table, prefix, reference, fmt, dpi):
    """Draw scaling figures from TABLE."""
    raw = json.loads(table.read_text())
    try:
        series = {k: (v["workers"], v["time_s"]) for k, v in raw.items()}
    except (KeyError, TypeError, AttributeError) as exc:
        raise click.ClickException(
            'TABLE must be {"series": {"workers": [...], "time_s": [...]}}'
        ) from exc
    if reference is not None and reference not in series:
        raise click.ClickException(
            f"--reference {reference!r} is not a series of {table}"
        )
    t = ScalingTable(series, reference)
    command = " ".join(["rgpycrumbs", "surrogate", "plt-scaling", *sys.argv[1:]])
    for panel, fig in (
        ("speedup", plot_scaling(t)),
        ("efficiency", plot_efficiency_table(t)),
    ):
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
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
