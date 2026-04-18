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
#   "xyzrender>=0.1.3",
#   "readcon>=0.7.0",
# ]
# ///

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory
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

IRA_KMAX_DEFAULT = 14.0


@click.command()
@click.option(
    "--job-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    multiple=True,
    help="Path to eOn saddle search output directory. Repeat for overlay.",
)
@click.option(
    "--label",
    type=str,
    multiple=True,
    help="Label for each job-dir (e.g. FIRE, LBFGS). Must match --job-dir count.",
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
    "--plot-structures",
    type=click.Choice(["none", "endpoints"]),
    default="none",
    help="Show structure strip below landscape.",
)
@click.option(
    "--strip-renderer",
    type=click.Choice(["xyzrender", "ase", "solvis", "ovito"]),
    default="xyzrender",
    help=(
        "Rendering backend for structure strip. "
        "xyzrender/ase work with the default dispatcher setup; "
        "solvis and ovito require separate heavy installs."
    ),
)
@click.option(
    "--xyzrender-config",
    type=str,
    default="paton",
    show_default=True,
    help="xyzrender preset (paton, bubble, flat, tube, wire, skeletal).",
)
@click.option("--strip-spacing", type=float, default=1.5, help="Column spacing in strip.")
@click.option(
    "--strip-zoom",
    type=float,
    default=None,
    help="Strip image zoom (default: auto-scaled by atom count).",
)
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
    label,
    plot_type,
    ref_product,
    project_path,
    surface_type,
    ira_kmax,
    theme,
    plot_structures,
    strip_renderer,
    xyzrender_config,
    strip_spacing,
    strip_zoom,
    strip_dividers,
    rotation,
    perspective_tilt,
    output,
    dpi,
    verbose,
):
    """Plot dimer/saddle search trajectory visualization.

    Use --job-dir multiple times to overlay trajectories from different
    optimizers (e.g. FIRE, LBFGS, SD) on the same landscape or profile.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if output is None:
        output = Path(f"saddle_{plot_type}.pdf")

    # Load all trajectories
    trajs = []
    for jd in job_dir:
        log.info("Loading trajectory from %s", jd)
        t = load_dimer_trajectory(jd)
        log.info(
            "  %d frames, saddle=%s",
            len(t.atoms_list),
            "yes" if t.saddle_atoms is not None else "no",
        )
        trajs.append(t)

    # Generate labels
    labels = list(label) if label else [Path(jd).name for jd in job_dir]
    if len(labels) < len(trajs):
        labels.extend([f"Run {i + 1}" for i in range(len(labels), len(trajs))])

    traj = trajs[0]  # primary trajectory for single-traj plot types

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
            ref_product=ref_product,
            project_path=project_path,
            surface_type=surface_type,
            ira_kmax=ira_kmax,
            cmap=active_theme.cmap_landscape,
            plot_structures=plot_structures,
            strip_renderer=strip_renderer,
            xyzrender_config=xyzrender_config,
            strip_spacing=strip_spacing,
            strip_zoom=strip_zoom,
            strip_dividers=strip_dividers,
            rotation=rotation,
            perspective_tilt=perspective_tilt,
            theme=active_theme,
        )
    elif plot_type == "convergence":
        _plot_convergence(trajs, labels, output, dpi)
    elif plot_type == "mode-evolution":
        _plot_mode_evolution(traj, output, dpi)

    log.info("Saved %s", output)


_OVERLAY_COLORS = ["#004D40", "#FF655D", "#3F51B5", "#FF9800", "#9C27B0", "#009688"]


def _plot_profile(trajs, labels, output, dpi):
    has_eigen = any("eigenvalue" in t.dat_df.columns for t in trajs)
    fig, axes = plt.subplots(
        1, 2 if has_eigen else 1, figsize=(10 if has_eigen else 5.37, 4), dpi=dpi
    )
    if not has_eigen:
        axes = [axes]

    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        dat = traj.dat_df
        color = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]
        iters = dat["iteration"].to_numpy()
        energies = dat["delta_e"].to_numpy()
        axes[0].plot(
            iters, energies, "o-", color=color, markersize=4, linewidth=1.5, label=lbl
        )

        if has_eigen and "eigenvalue" in dat.columns:
            eigenvalues = dat["eigenvalue"].to_numpy()
            axes[1].plot(
                iters,
                eigenvalues,
                "s-",
                color=color,
                markersize=3,
                linewidth=1.2,
                label=lbl,
            )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("Energy vs Iteration")
    if len(trajs) > 1:
        axes[0].legend(frameon=False)

    if has_eigen:
        axes[1].axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Eigenvalue (eV/$\\AA^2$)")
        axes[1].set_title("Eigenvalue vs Iteration")
        if len(trajs) > 1:
            axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_landscape(
    trajs,
    labels,
    output,
    dpi,
    *,
    ref_product,
    project_path,
    surface_type,
    ira_kmax,
    cmap="viridis",
    plot_structures="none",
    strip_renderer="xyzrender",
    xyzrender_config="paton",
    strip_spacing=1.5,
    strip_zoom=None,
    strip_dividers=False,
    rotation="auto",
    perspective_tilt=0.0,
    theme=None,
):
    import readcon

    try:
        from rgpycrumbs._aux import _import_from_parent_env

        ira_mod = _import_from_parent_env("ira_mod")
    except ImportError:
        ira_mod = None

    ira_instance = ira_mod.IRA() if ira_mod else None
    traj = trajs[0]

    # Determine reference B from primary trajectory
    if ref_product is not None:
        ref_b = readcon.read_con_as_ase(str(ref_product))[0]
    elif traj.saddle_atoms is not None:
        ref_b = traj.saddle_atoms
    else:
        ref_b = traj.atoms_list[-1]

    has_strip = plot_structures == "endpoints"
    fig = plt.figure(figsize=(5.37, 5.37 + (1.5 if has_strip else 0)), dpi=dpi)

    if has_strip:
        gs = GridSpec(2, 1, height_ratios=[1, 0.3], hspace=0.15, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1])
        if theme:
            apply_axis_theme(ax_strip, theme)
    else:
        ax = fig.add_subplot(111)
        ax_strip = None

    if theme:
        apply_axis_theme(ax, theme)

    # Plot surface from primary trajectory
    rmsd_a, rmsd_b = calculate_landscape_coords(
        traj.atoms_list,
        ira_instance,
        ira_kmax,
        ref_a=traj.initial_atoms,
        ref_b=ref_b,
    )
    energies = traj.dat_df["delta_e"].to_numpy()
    n = min(len(rmsd_a), len(energies))
    rmsd_a, rmsd_b, energies = rmsd_a[:n], rmsd_b[:n], energies[:n]
    f_para = -np.gradient(energies)
    grad_a, grad_b = compute_synthetic_gradients(rmsd_a, rmsd_b, f_para)

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
            ref_b=ref_b,
        )
        e = t.dat_df["delta_e"].to_numpy()
        m = min(len(ra), len(e))
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
        ax.annotate(
            "R",
            (float(px[0]), float(py[0])),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=60,
        )

    # Label final point of primary trajectory
    if project_path:
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        s_all, d_all = project_to_sd(rmsd_a, rmsd_b, basis)
        ex, ey = float(s_all[-1]), float(d_all[-1])
    else:
        ex, ey = float(rmsd_a[-1]), float(rmsd_b[-1])

    label_b = "SP" if traj.saddle_atoms is not None else "End"
    ax.annotate(
        label_b,
        (ex, ey),
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=60,
    )

    if len(trajs) > 1:
        ax.legend(frameon=True, framealpha=0.9, loc="best")

    # Structure strip
    if has_strip and ax_strip is not None:
        structs = [traj.initial_atoms]
        strip_labels = ["R"]
        if traj.saddle_atoms is not None:
            structs.append(traj.saddle_atoms)
            strip_labels.append("SP")
        else:
            structs.append(traj.atoms_list[-1])
            strip_labels.append("End")

        if strip_zoom is None:
            max_atoms = max(len(s) for s in structs) if structs else 10
            strip_zoom = max(0.25, 0.8 * (20 / max(max_atoms, 20)) ** 0.3)
        plot_structure_strip(
            ax_strip,
            structs,
            strip_labels,
            zoom=strip_zoom,
            rotation=rotation,
            theme_color=theme.textcolor if theme else "black",
            renderer=strip_renderer,
            col_spacing=strip_spacing,
            xyzrender_config=xyzrender_config,
            show_dividers=strip_dividers,
            perspective_tilt=perspective_tilt,
        )

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(trajs, labels, output, dpi):
    fig, (ax_force, ax_step) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        color = _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]
        plot_convergence_panel(ax_force, ax_step, traj.dat_df, color=color)
        # Add legend entry via invisible line
        ax_force.plot([], [], color=color, label=lbl)
    if len(trajs) > 1:
        ax_force.legend(frameon=False)
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
