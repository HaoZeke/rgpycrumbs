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
#   "chemparseplot>=1.8.0",
#   "xyzrender>=0.1.3",
#   "readcon>=0.7.0",
# ]
# ///

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._render_cli import add_render_options
    from ._single_ended_cli import default_output_path, load_trajectories, overlay_labels
except ImportError:  # pragma: no cover - direct script execution
    from rgpycrumbs.eon._render_cli import add_render_options
    from rgpycrumbs.eon._single_ended_cli import (
        default_output_path,
        load_trajectories,
        overlay_labels,
    )
from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory
from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
)
from chemparseplot.plot.optimization import (
    OVERLAY_COLORS,
    annotate_endpoint,
    create_landscape_axes,
    plot_optimization_landscape,
    plot_single_ended_convergence,
    plot_single_ended_profile,
    project_landscape_path,
    render_endpoint_strip,
    save_landscape_figure,
)
from chemparseplot.plot.structs import (
    convert_energy,
)
from chemparseplot.plot.theme import get_theme, setup_global_theme
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
    "--energy-unit",
    type=click.Choice(["eV", "kcal/mol", "kJ/mol"]),
    default="eV",
    show_default=True,
    help="Presentation unit for energy axes and color scales.",
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
@add_render_options
@click.option(
    "--strip-zoom",
    type=float,
    default=None,
    help="Strip image zoom (default: auto-scaled by atom count).",
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
    energy_unit,
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

    output = default_output_path("saddle", plot_type, output)
    trajs = load_trajectories(
        job_dir,
        load_dimer_trajectory,
        log_info=log.info,
        noun="trajectory",
        detail=lambda traj: (
            f"{len(traj.atoms_list)} frames, saddle="
            f"{'yes' if traj.saddle_atoms is not None else 'no'}"
        ),
    )
    labels = overlay_labels(job_dir, label)

    traj = trajs[0]  # primary trajectory for single-traj plot types

    active_theme = get_theme(theme)
    setup_global_theme(active_theme)

    if plot_type == "profile":
        _plot_profile(trajs, labels, output, dpi, energy_unit=energy_unit)
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
            energy_unit=energy_unit,
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


def _plot_profile(trajs, labels, output, dpi, *, energy_unit):
    plot_single_ended_profile(
        trajs,
        labels,
        output,
        dpi,
        energy_unit=energy_unit,
        energy_column="delta_e",
        title="Energy vs Iteration",
        eigen_column="eigenvalue",
    )


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
    energy_unit,
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
    fig, ax, ax_strip = create_landscape_axes(dpi=dpi, has_strip=has_strip, theme=theme)

    # Plot surface from primary trajectory
    rmsd_a, rmsd_b = calculate_landscape_coords(
        traj.atoms_list,
        ira_instance,
        ira_kmax,
        ref_a=traj.initial_atoms,
        ref_b=ref_b,
    )
    energies = convert_energy(traj.dat_df["delta_e"].to_numpy(), energy_unit)
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
        energy_unit=energy_unit,
    )

    # Overlay paths from all trajectories
    basis = None
    if project_path:
        _, _, basis = project_landscape_path(rmsd_a, rmsd_b, project_path=True)
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

        px, py, _ = project_landscape_path(ra, rb, project_path=project_path, basis=basis)

        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
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
        annotate_endpoint(ax, float(px[0]), float(py[0]), "R", boxed=False)

    # Label final point of primary trajectory
    plot_x, plot_y, _ = project_landscape_path(
        rmsd_a, rmsd_b, project_path=project_path, basis=basis
    )
    ex, ey = float(plot_x[-1]), float(plot_y[-1])

    label_b = "SP" if traj.saddle_atoms is not None else "End"
    annotate_endpoint(ax, ex, ey, label_b, boxed=False)

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

        render_endpoint_strip(
            ax_strip,
            structs,
            strip_labels,
            strip_zoom=strip_zoom,
            rotation=rotation,
            theme=theme,
            strip_renderer=strip_renderer,
            strip_spacing=strip_spacing,
            strip_dividers=strip_dividers,
            perspective_tilt=perspective_tilt,
            xyzrender_config=xyzrender_config,
        )

    save_landscape_figure(fig, output, dpi=dpi, has_strip=has_strip)


def _plot_convergence(trajs, labels, output, dpi):
    plot_single_ended_convergence(trajs, labels, output, dpi)


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
