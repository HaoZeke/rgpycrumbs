#!/usr/bin/env python3
"""Draw the band of a surrogate-assisted NEB: profile, evolution, landscape.

.. versionadded:: 1.12.0

Reads one search directory with a chemparseplot parser and draws

- ``profile``: surrogate mean along the path, uncertainty ribbon when the
  producer recorded one, true evaluations, retained observations;
- ``evolution``: the band over outer iterations (snapshots), or the record of
  which image the oracle was called on when only the final band exists.

Prefer the dispatcher::

    rgpycrumbs surrogate plt-band results/rec/baker/25_hcnh2 -o figs/hcnh2
    rgpycrumbs surrogate plt-band run/ --producer ml-neb --panel profile

Files are ``<prefix>-profile.<fmt>`` and ``<prefix>-evolution.<fmt>``. Output
is byte-reproducible; each file embeds the SHA-256 of the inputs and the tool
versions, and a ``.provenance.json`` sits beside it.
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@4af4172c2fe7d43fae5630e3d2694ae62452c170",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import read_search
from chemparseplot.plot.provenance import save_with_provenance
from chemparseplot.plot.surrogate import plot_band_evolution, plot_band_profile

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-band")


@click.command()
@click.argument("search", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; the panel name and format are appended.",
)
@click.option(
    "--producer",
    type=click.Choice(["gpr_optim", "ml-neb"]),
    default="gpr_optim",
    show_default=True,
    help="Which code wrote SEARCH.",
)
@click.option(
    "--panel",
    "panels",
    multiple=True,
    type=click.Choice(["profile", "evolution"]),
    help="Panel to draw (repeatable). Default: both.",
)
@click.option(
    "--energy-unit",
    default="eV",
    show_default=True,
    help="eV, kcal/mol, kJ/mol or hartree.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
def main(*, search, prefix, producer, panels, energy_unit, fmt, dpi):
    """Draw the band profile and its evolution for SEARCH."""
    s = read_search(search, producer)
    if s.band is None:
        raise click.ClickException(f"{search} holds no band (no band.h5 or trajectories)")
    command = " ".join(["rgpycrumbs", "surrogate", "plt-band", *sys.argv[1:]])
    for panel in panels or ("profile", "evolution"):
        if panel == "profile":
            fig = plot_band_profile(s.band, energy_unit=energy_unit, title=s.label)
        else:
            fig = plot_band_evolution(s.band, energy_unit=energy_unit)
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=s.provenance, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
