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
#   "xyzrender>=0.1.3",
# ]
# ///

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from chemparseplot.parse.eon.min_trajectory import load_min_trajectory
from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
)
from chemparseplot.parse.projection import compute_projection_basis, project_to_sd
from chemparseplot.plot.neb import plot_structure_strip
from chemparseplot.plot.optimization import (
    plot_convergence_panel,
    plot_optimization_landscape,
)
from chemparseplot.plot.theme import apply_axis_theme, get_theme, setup_global_theme
from matplotlib.gridspec import GridSpec
from rich.logging import RichHandler

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
    multiple=True,
    help="Path to eOn minimization output directory. Repeat for overlay.",
)
@click.option(
    "--label",
    type=str,
    multiple=True,
    help="Label for each job-dir (e.g. FIRE, LBFGS). Must match --job-dir count.",
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
    "--plot-structures",
    type=click.Choice(["none", "endpoints"]),
    default="none",
    help="Show structure strip below landscape.",
)
@click.option(
    "--strip-renderer",
    type=click.Choice(["xyzrender", "ase", "solvis", "ovito"]),
    default="xyzrender",
    help="Rendering backend for structure strip.",
)
@click.option(
    "--xyzrender-config", type=str, default="paton", show_default=True,
    help="xyzrender preset (paton, bubble, flat, tube, wire, skeletal).",
)
@click.option("--strip-spacing", type=float, default=1.5, help="Column spacing in strip.")
@click.option(
    "--strip-dividers",
    is_flag=True,
    default=False,
    help="Show dividers between structures.",
)
@click.option(
    "--rotation", type=str, default="auto", help="Viewing angle for structure rendering."
)
@click.option(
    "--perspective-tilt",
    type=float,
    default=0.0,
    help="Off-axis perspective tilt in degrees.",
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
    label,
    prefix,
    plot_type,
    project_path,
    surface_type,
    ira_kmax,
    theme,
    plot_structures,
    strip_renderer,
    xyzrender_config,
    strip_spacing,
    strip_dividers,
    rotation,
    perspective_tilt,
    output,
    dpi,
    verbose,
):
    """Plot minimization trajectory visualization.

    Use --job-dir multiple times to overlay trajectories from different
    optimizers (e.g. FIRE, LBFGS, SD) on the same landscape or profile.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if output is None:
        output = Path(f"min_{plot_type}.pdf")

    trajs = []
    for jd in job_dir:
        log.info("Loading minimization trajectory from %s", jd)
        t = load_min_trajectory(jd, prefix=prefix)
        log.info("  %d frames", len(t.atoms_list))
        trajs.append(t)

    labels = list(label) if label else [Path(jd).name for jd in job_dir]
    if len(labels) < len(trajs):
        labels.extend([f"Run {i + 1}" for i in range(len(labels), len(trajs))])

    trajs[0]

    active_theme = get_theme(theme)
    setup_global_theme(active_theme)

    if plot_type == "profile":
        _plot_profile(trajs, labels, output, dpi)
    elif plot_type == "landscape":
        _plot_landscape(
            trajs,
            labels,
            output,
            dpi,
            project_path=project_path,
            surface_type=surface_type,
            ira_kmax=ira_kmax,
            cmap=active_theme.cmap_landscape,
            plot_structures=plot_structures,
            strip_renderer=strip_renderer,
            xyzrender_config=xyzrender_config,
            strip_spacing=strip_spacing,
            strip_dividers=strip_dividers,
            rotation=rotation,
            perspective_tilt=perspective_tilt,
            theme=active_theme,
        )
    elif plot_type == "convergence":
        _plot_convergence(trajs, labels, output, dpi)

    log.info("Saved %s", output)


_OVERLAY_COLORS = ["#004D40", "#FF655D", "#3F51B5", "#FF9800", "#9C27B0", "#009688"]


def _plot_profile(trajs, labels, output, dpi):
    fig, ax = plt.subplots(figsize=(5.37, 4), dpi=dpi)

    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        dat = traj.dat_df
        color = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]
        iters = dat["iteration"].to_numpy()
        energies = dat["energy"].to_numpy()
        ax.plot(
            iters, energies, "o-", color=color, markersize=4, linewidth=1.5, label=lbl
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Minimization Energy Profile")
    if len(trajs) > 1:
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_landscape(
    trajs,
    labels,
    output,
    dpi,
    *,
    project_path,
    surface_type,
    ira_kmax,
    cmap="viridis",
    plot_structures="none",
    strip_renderer="xyzrender",
    xyzrender_config="paton",
    strip_spacing=1.5,
    strip_dividers=False,
    rotation="auto",
    perspective_tilt=0.0,
    theme=None,
):
    try:
        from rgpycrumbs._aux import _import_from_parent_env

        ira_mod = _import_from_parent_env("ira_mod")
    except ImportError:
        ira_mod = None

    ira_instance = ira_mod.IRA() if ira_mod else None
    traj = trajs[0]

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

    has_strip = plot_structures == "endpoints"
    fig = plt.figure(figsize=(5.37, 5.37 + (1.5 if has_strip else 0)), dpi=dpi)

    if has_strip:
        gs = GridSpec(2, 1, height_ratios=[1, 0.25], hspace=0.3, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1])
        if theme:
            apply_axis_theme(ax_strip, theme)
    else:
        ax = fig.add_subplot(111)
        ax_strip = None

    if theme:
        apply_axis_theme(ax, theme)

    plot_optimization_landscape(
        ax,
        rmsd_a,
        rmsd_b,
        grad_a,
        grad_b,
        energies,
        project_path=project_path,
        method=surface_type,
        cmap=cmap,
        label_mode="optimization",
    )

    # Overlay paths from all trajectories
    for idx, (t, lbl) in enumerate(zip(trajs, labels, strict=False)):
        ra, rb = calculate_landscape_coords(
            t.atoms_list,
            ira_instance,
            ira_kmax,
            ref_a=traj.initial_atoms,
            ref_b=traj.final_atoms,
        )
        m = min(len(ra), len(t.dat_df))
        ra, rb = ra[:m], rb[:m]

        if project_path:
            basis = compute_projection_basis(rmsd_a, rmsd_b)
            px, py = project_to_sd(ra, rb, basis)
        else:
            px, py = ra, rb

        color = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]
        if len(trajs) > 1:
            ax.plot(
                px,
                py,
                "o-",
                color=color,
                markersize=3,
                linewidth=1.5,
                alpha=0.8,
                zorder=55,
                label=lbl,
            )

    # Annotate endpoints
    if project_path:
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        s_all, d_all = project_to_sd(rmsd_a, rmsd_b, basis)
        sx, sy = float(s_all[0]), float(d_all[0])
        ex, ey = float(s_all[-1]), float(d_all[-1])
    else:
        sx, sy = float(rmsd_a[0]), float(rmsd_b[0])
        ex, ey = float(rmsd_a[-1]), float(rmsd_b[-1])

    ax.annotate(
        "Init",
        (sx, sy),
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=60,
    )
    ax.annotate(
        "Min",
        (ex, ey),
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=60,
    )

    if len(trajs) > 1:
        ax.legend(frameon=True, framealpha=0.9, loc="best")

    if has_strip and ax_strip is not None:
        structs = [traj.initial_atoms]
        strip_labels = ["Init"]
        if traj.final_atoms is not None:
            structs.append(traj.final_atoms)
            strip_labels.append("Min")
        else:
            structs.append(traj.atoms_list[-1])
            strip_labels.append("Min")

        plot_structure_strip(
            ax_strip,
            structs,
            strip_labels,
            zoom=0.8,
            rotation=rotation,
            theme_color=theme.textcolor if theme else "black",
            renderer=strip_renderer,
            col_spacing=strip_spacing,
            show_dividers=strip_dividers,
            perspective_tilt=perspective_tilt,
            xyzrender_config=xyzrender_config,
        )

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(trajs, labels, output, dpi):
    fig, (ax_force, ax_step) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        color = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]
        plot_convergence_panel(ax_force, ax_step, traj.dat_df, color=color)
        ax_force.plot([], [], color=color, label=lbl)
    if len(trajs) > 1:
        ax_force.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
