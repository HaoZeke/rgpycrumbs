#!/usr/bin/env python3
"""Plots minimization optimization trajectories.

.. versionadded:: 1.3.0

Visualizes minimization runs using the generalized (s, d) reaction
valley projection. Supports:

1. **Energy Profile:** Energy vs iteration.
2. **2D Optimization Landscape:** Projects trajectory into (progress,
   deviation) coordinates relative to (initial, minimum).
3. **Convergence Panel:** Force norm and step size vs iteration.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "jax",
#   "cmcrameri",
#   "rich",
#   "ase",
#   "polars",
#   "chemparseplot",
# ]
# ///

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler

from chemparseplot.parse.eon.min_trajectory import load_min_trajectory
from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
)
from chemparseplot.plot.optimization import (
    plot_convergence_panel,
    plot_optimization_landscape,
    plot_optimization_profile,
)
from chemparseplot.plot.theme import get_theme, setup_global_theme

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)
log = logging.getLogger("rich")

IRA_KMAX_DEFAULT = 1.8


@click.command()
@click.option(
    "--job-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the eOn minimization output directory.",
)
@click.option(
    "--prefix",
    type=str,
    default="min",
    help="Movie file prefix (default 'min').",
)
@click.option(
    "--plot-type",
    type=click.Choice(["profile", "landscape", "convergence"]),
    default="profile",
    help="Type of plot to generate.",
)
@click.option(
    "--project-path/--no-project-path",
    is_flag=True,
    default=True,
    help="Project landscape into (s, d) coordinates.",
)
@click.option(
    "--surface-type",
    type=click.Choice(["grad_matern", "grad_imq", "rbf"]),
    default="grad_matern",
    help="Surface fitting method for landscape plot.",
)
@click.option(
    "--ira-kmax",
    type=float,
    default=IRA_KMAX_DEFAULT,
    help="IRA kmax parameter for RMSD calculation.",
)
@click.option(
    "--theme",
    type=str,
    default="ruhi",
    help="Plot theme name.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path. Defaults to min_{plot_type}.pdf.",
)
@click.option(
    "--dpi",
    type=int,
    default=200,
    help="Output resolution.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    job_dir,
    prefix,
    plot_type,
    project_path,
    surface_type,
    ira_kmax,
    theme,
    output,
    dpi,
    verbose,
):
    """Plot minimization trajectory visualization."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if output is None:
        output = Path(f"min_{plot_type}.pdf")

    log.info("Loading minimization trajectory from %s", job_dir)
    traj = load_min_trajectory(job_dir, prefix=prefix)
    log.info("Loaded %d frames", len(traj.atoms_list))

    setup_global_theme(get_theme(theme))

    if plot_type == "profile":
        _plot_profile(traj, output, dpi)
    elif plot_type == "landscape":
        _plot_landscape(
            traj, output, dpi,
            project_path=project_path,
            surface_type=surface_type,
            ira_kmax=ira_kmax,
        )
    elif plot_type == "convergence":
        _plot_convergence(traj, output, dpi)

    log.info("Saved %s", output)


def _plot_profile(traj, output, dpi):
    dat = traj.dat_df
    fig, ax = plt.subplots(figsize=(5.37, 4), dpi=dpi)

    iters = dat["iteration"].to_numpy()
    energies = dat["energy"].to_numpy()

    plot_optimization_profile(ax, iters, energies)
    ax.set_title("Minimization Energy Profile")

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_landscape(traj, output, dpi, *, project_path, surface_type, ira_kmax):
    try:
        from rgpycrumbs._aux import _import_from_parent_env

        ira_mod = _import_from_parent_env("ira_mod")
    except ImportError:
        ira_mod = None

    ira_instance = ira_mod.IRA() if ira_mod else None

    rmsd_a, rmsd_b = calculate_landscape_coords(
        traj.atoms_list,
        ira_instance,
        ira_kmax,
        ref_a=traj.initial_atoms,
        ref_b=traj.final_atoms,
    )

    energies = traj.dat_df["energy"].to_numpy()
    n = min(len(rmsd_a), len(energies))
    rmsd_a, rmsd_b, energies = rmsd_a[:n], rmsd_b[:n], energies[:n]

    f_para = -np.gradient(energies)
    grad_a, grad_b = compute_synthetic_gradients(rmsd_a, rmsd_b, f_para)

    fig, ax = plt.subplots(figsize=(5.37, 5.37), dpi=dpi)

    plot_optimization_landscape(
        ax,
        rmsd_a,
        rmsd_b,
        grad_a,
        grad_b,
        energies,
        project_path=project_path,
        method=surface_type,
        label_mode="optimization",
    )

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(traj, output, dpi):
    fig, (ax_force, ax_step) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    plot_convergence_panel(ax_force, ax_step, traj.dat_df)
    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
