#!/usr/bin/env python3
"""Draw a landfold FES CSV through chemparseplot.

Landfold writes ``# x y F rho`` via ``landfold fes --csv``. This script
does not invert the histogram; it only loads that grid and calls
``chemparseplot.plot.landfold.plot_fes``.

.. versionadded:: 1.10.11
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "chemparseplot>=1.9.17,<2",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs landfold plot-fes")


@click.command()
@click.option(
    "--input",
    "inp",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV from `landfold fes --csv` (columns x y F rho).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output figure path (.png or .svg).",
)
@click.option("--kt", default=1.0, show_default=True, type=float, help="kT used for the invert.")
@click.option("--fmax", default=None, type=float, help="Clip finite F to [0, fmax].")
@click.option("--clabel", default=None, help="Colorbar label (default F/kT when kt=1).")
@click.option("--xlabel", default=r"$s_1$", show_default=True)
@click.option("--ylabel", default=r"$s_2$", show_default=True)
def main(inp, output, kt, fmax, clabel, xlabel, ylabel) -> None:
    from chemparseplot.parse.landfold import load_fes_csv
    from chemparseplot.plot.landfold import plot_fes

    fes = load_fes_csv(inp, kt=kt)
    fig = plot_fes(fes, xlabel=xlabel, ylabel=ylabel, clabel=clabel, fmax=fmax)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    click.echo(str(output))


if __name__ == "__main__":
    main()
