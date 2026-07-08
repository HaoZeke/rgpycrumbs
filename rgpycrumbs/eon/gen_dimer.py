#!/usr/bin/env python3
"""Seed eOn dimer saddle searches from every peak of a converged NEB band.

.. versionadded:: 1.8.0

Reads a converged NEB band (``neb.con``), finds all interior local maxima of
the energy profile, and writes one ``job=saddle_search`` seed directory per
peak under ``out-dir/peak_<NN>/`` (pos.con, direction.dat, displacement.con,
config.ini). See :mod:`rgpycrumbs.eon.seed_dimers` for the library API.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "scipy",
#   "rich",
#   "ase",
#   "readcon>=0.7.0",
#   "rgpycrumbs>=1.9.13",
# ]
# ///

import logging
from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - direct script execution without package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs eon gen-dimer")

try:
    from .seed_dimers import seed_dimers_from_peaks
except ImportError:  # pragma: no cover - direct script execution
    from rgpycrumbs.eon.seed_dimers import seed_dimers_from_peaks

logging.basicConfig(level=logging.INFO, format="%(message)s")


@click.command()
@click.option(
    "--neb-con",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Converged NEB band (neb.con) with per-image energies.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Parent dir; each peak gets out-dir/peak_<NN>/.",
)
@click.option(
    "--settings",
    "settings_name",
    required=True,
    help="Name of the NWChem .nwi settings fragment referenced by config.ini.",
)
@click.option("--socket", required=True, help="Unix socket path for [SocketNWChemPot].")
@click.option(
    "--peak-files-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Dir with eOn peak{MM}_pos.con + peak{MM}_mode.dat to seed direction.dat.",
)
@click.option(
    "--prominence",
    type=float,
    default=0.02,
    show_default=True,
    help="scipy.signal.find_peaks prominence on the energy profile.",
)
@click.option(
    "--rmsd-tol",
    type=float,
    default=0.1,
    show_default=True,
    help="Max RMSD (A) for an eOn peak geometry to match a NEB peak frame.",
)
def main(
    neb_con: Path,
    out_dir: Path,
    settings_name: str,
    socket: str,
    peak_files_dir: Path | None,
    prominence: float,
    rmsd_tol: float,
):
    """Seed one eOn dimer saddle search per interior NEB peak."""
    seed_dirs, summary = seed_dimers_from_peaks(
        neb_con=neb_con,
        out_dir=out_dir,
        settings_name=settings_name,
        socket=socket,
        peak_files_dir=peak_files_dir,
        prominence=prominence,
        rmsd_tol=rmsd_tol,
    )
    if not seed_dirs:
        click.echo("No interior peaks found; no seed directories written.", err=True)
        return
    click.echo(f"Created {len(seed_dirs)} dimer seed dir(s) under {out_dir}:")
    for seed in summary:
        click.echo(
            f"  peak_{seed.peak_index:02d}  image={seed.image_index}  "
            f"E={seed.energy:.4f}  dE={seed.energy_vs_reactant:.4f}  "
            f"mode={seed.mode_source}  -> {seed.seed_dir}"
        )


if __name__ == "__main__":
    main()
