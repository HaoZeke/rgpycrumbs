#!/usr/bin/env python3
"""Plots dimer/saddle search optimization trajectories.

.. versionadded:: 1.3.0

Visualizes single-ended saddle point searches (dimer, Lanczos, GPRD)
using the generalized (s, d) reaction valley projection. Supports:

1. **2D Optimization Landscape:** Projects the optimization trajectory
   into (progress, deviation) coordinates relative to (initial, saddle).
2. **Energy/Eigenvalue Profile:** Energy and curvature vs iteration.
3. **Convergence Panel:** Force norm and step size vs iteration.
4. **Mode Evolution:** Alignment of dimer mode with final mode.
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

from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory
from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
)
from chemparseplot.plot.optimization import (
    plot_convergence_panel,
    plot_dimer_mode_evolution,
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
    help="Path to the eOn saddle search output directory.",
)
@click.option(
    "--plot-type",
    type=click.Choice(["profile", "landscape", "convergence", "mode-evolution"]),
    default="profile",
    help="Type of plot to generate.",
)
@click.option(
    "--ref-product",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional product .con file to use as reference B instead of saddle.",
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
    help="Output file path. Defaults to {plot_type}.pdf.",
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
    plot_type,
    ref_product,
    project_path,
    surface_type,
    ira_kmax,
    theme,
    output,
    dpi,
    verbose,
):
    """Plot dimer/saddle search trajectory visualization."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if output is None:
        output = Path(f"saddle_{plot_type}.pdf")

    log.info("Loading trajectory from %s", job_dir)
    traj = load_dimer_trajectory(job_dir)
    log.info(
        "Loaded %d frames, saddle=%s",
        len(traj.atoms_list),
        "yes" if traj.saddle_atoms is not None else "no",
    )

    setup_global_theme(get_theme(theme))

    if plot_type == "profile":
        _plot_profile(traj, output, dpi)
    elif plot_type == "landscape":
        _plot_landscape(
            traj,
            output,
            dpi,
            ref_product=ref_product,
            project_path=project_path,
            surface_type=surface_type,
            ira_kmax=ira_kmax,
        )
    elif plot_type == "convergence":
        _plot_convergence(traj, output, dpi)
    elif plot_type == "mode-evolution":
        _plot_mode_evolution(traj, output, dpi)

    log.info("Saved %s", output)


def _plot_profile(traj, output, dpi):
    dat = traj.dat_df
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

    iters = dat["iteration"].to_numpy()
    energies = dat["delta_e"].to_numpy()

    eigenvalues = dat["eigenvalue"].to_numpy() if "eigenvalue" in dat.columns else None

    plot_optimization_profile(
        axes[0],
        iters,
        energies,
        eigenvalues=eigenvalues,
        ax_eigen=axes[1] if eigenvalues is not None else None,
    )
    axes[0].set_title("Energy vs Iteration")

    if eigenvalues is not None:
        axes[1].set_title("Eigenvalue vs Iteration")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_landscape(traj, output, dpi, *, ref_product, project_path, surface_type, ira_kmax):
    from ase.io import read as ase_read

    try:
        from rgpycrumbs._aux import _import_from_parent_env

        ira_mod = _import_from_parent_env("ira_mod")
    except ImportError:
        ira_mod = None

    ira_instance = ira_mod.IRA() if ira_mod else None

    # Determine reference B
    if ref_product is not None:
        ref_b = ase_read(str(ref_product), format="eon")
    elif traj.saddle_atoms is not None:
        ref_b = traj.saddle_atoms
    else:
        ref_b = traj.atoms_list[-1]

    rmsd_a, rmsd_b = calculate_landscape_coords(
        traj.atoms_list,
        ira_instance,
        ira_kmax,
        ref_a=traj.initial_atoms,
        ref_b=ref_b,
    )

    # Synthetic gradients from energy differences
    energies = traj.dat_df["delta_e"].to_numpy()
    if len(energies) == len(rmsd_a):
        f_para = -np.gradient(energies)
        grad_a, grad_b = compute_synthetic_gradients(rmsd_a, rmsd_b, f_para)
    else:
        # Mismatch (movie may include initial frame not in dat)
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


def _plot_mode_evolution(traj, output, dpi):
    if traj.mode_vector is None:
        log.warning("No mode.dat found; cannot plot mode evolution")
        return

    # Mode evolution requires per-iteration mode vectors.
    # Currently only the final mode is available from mode.dat.
    # When per-iteration modes are saved, this will use them.
    log.warning(
        "Per-iteration mode vectors not yet available from eOn output. "
        "Showing final mode only."
    )
    fig, ax = plt.subplots(figsize=(5.37, 4), dpi=dpi)
    ax.text(
        0.5,
        0.5,
        "Per-iteration mode vectors\nnot yet available",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
