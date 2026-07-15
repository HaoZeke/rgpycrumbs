#!/usr/bin/env python3
"""Plot a general readcon CON/convel trajectory (energy / force overview).

Unlike ``plt-min`` / ``plt-neb``, this does not require an eOn job directory
or sidecar ``.dat`` files — only a CON movie (or multi-image path file).

.. versionadded:: 1.9.x
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "polars",
#   "ase",
#   "rich",
#   "chemparseplot[neb,plot]>=1.9.12,<2",
#   "readcon>=0.13.1",
#   "rgpycrumbs>=1.9.13",
# ]
# ///

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs eon plt-con")

from chemparseplot.api import load_con_trajectory, plot_con_overview
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)
log = logging.getLogger("rich")


@click.command()
@click.argument("con_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output figure path (default: <con_stem>_profile.pdf).",
)
@click.option(
    "--energy-unit",
    type=click.Choice(["eV", "kcal/mol", "kJ/mol"]),
    default="eV",
    show_default=True,
)
@click.option(
    "--absolute/--relative",
    default=False,
    help="Plot absolute energies instead of ΔE from the first frame.",
)
@click.option(
    "--no-forces",
    is_flag=True,
    default=False,
    help="Skip force panel even when CON frames carry forces.",
)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option("--title", type=str, default=None)
def main(con_file, output, energy_unit, absolute, no_forces, dpi, title):
    """Plot energy (and force) profiles for a general CON trajectory."""
    traj = load_con_trajectory(con_file)
    log.info(
        "Loaded %s: %d frames, energy finite=%d, forces=%s",
        con_file.name,
        traj.n_frames,
        int(sum(1 for e in traj.energies if e == e)),  # NaN-safe count
        "yes" if (traj.table is not None and "fmax" in traj.table.columns) else "no",
    )
    fig = plot_con_overview(
        traj,
        energy_unit=energy_unit,
        relative=not absolute,
        show_forces=not no_forces,
        title=title or con_file.name,
    )
    out = output or con_file.with_name(f"{con_file.stem}_profile.pdf")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    log.info("Wrote %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
