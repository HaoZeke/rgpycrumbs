#!/usr/bin/env python3
"""Draw curvature and force of a surrogate-assisted single-ended search.

.. versionadded:: 1.12.0

Curvature along the dimer mode (surrogate value per step, measured value
where one was taken) and true force against oracle calls, with batches of
oracle calls spent on a curvature spectrum marked on both panels::

    rgpycrumbs surrogate plt-dimer rec/birkholz/16_oxirane -o figs/oxirane
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@ef29c8b4451cbe764e8ae3024726f3df9b1bd3ef",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import read_search
from chemparseplot.plot.provenance import save_with_provenance
from chemparseplot.plot.surrogate import plot_single_ended

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-dimer")


@click.command()
@click.argument("search", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; '-dimer' and the format are appended.",
)
@click.option(
    "--energy-unit",
    default="eV",
    show_default=True,
    help="eV, kcal/mol, kJ/mol or hartree (curvature axis).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
def main(search, prefix, energy_unit, fmt, dpi):
    """Draw the single-ended history of SEARCH (a gpr_optim dimer cell)."""
    s = read_search(search, "gpr_optim")
    if s.single_ended is None:
        raise click.ClickException(
            f"{search} holds no dimer history (no saddle MLflow run)"
        )
    command = " ".join(["rgpycrumbs", "surrogate", "plt-dimer", *sys.argv[1:]])
    fig = plot_single_ended(s.single_ended, energy_unit=energy_unit, title=s.label)
    out = prefix.with_name(f"{prefix.name}-dimer.{fmt}")
    save_with_provenance(fig, out, {}, command=command, hashes=s.provenance, dpi=dpi)
    click.echo(out)


if __name__ == "__main__":
    main()
