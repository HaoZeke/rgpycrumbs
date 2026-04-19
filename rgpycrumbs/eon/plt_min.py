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
#   "readcon>=0.7.0",
# ]
# ///

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from ._render_cli import add_render_options
from ._single_ended_plot import (
    annotate_endpoint,
    create_landscape_axes,
    plot_single_ended_convergence,
    plot_single_ended_profile,
    project_landscape_path,
    render_endpoint_strip,
    save_landscape_figure,
)
from chemparseplot.parse.eon.min_trajectory import load_min_trajectory
from chemparseplot.parse.neb_utils import (
    calculate_landscape_coords,
    compute_synthetic_gradients,
)
from chemparseplot.plot.optimization import plot_optimization_landscape
from chemparseplot.plot.structs import convert_energy
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
    default="minimization",
    help="Movie file prefix (default 'minimization').",
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

    log.info("Saved %s", output)


def _plot_profile(trajs, labels, output, dpi, *, energy_unit):
    plot_single_ended_profile(
        trajs,
        labels,
        output,
        dpi,
        energy_unit=energy_unit,
        energy_column="energy",
        title="Minimization Energy Profile",
    )


def _plot_landscape(
    trajs,
    labels,
    output,
    dpi,
    *,
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
    energies = convert_energy(traj.dat_df["energy"].to_numpy(), energy_unit)
    n = min(len(rmsd_a), len(energies))
    rmsd_a, rmsd_b, energies = rmsd_a[:n], rmsd_b[:n], energies[:n]
    f_para = -np.gradient(energies)
    grad_a, grad_b = compute_synthetic_gradients(rmsd_a, rmsd_b, f_para)

    has_strip = plot_structures == "endpoints"
    fig, ax, ax_strip = create_landscape_axes(dpi=dpi, has_strip=has_strip, theme=theme)

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
            ref_b=traj.final_atoms,
        )
        m = min(len(ra), len(t.dat_df))
        ra, rb = ra[:m], rb[:m]

        px, py, _ = project_landscape_path(ra, rb, project_path=project_path, basis=basis)

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
    plot_x, plot_y, _ = project_landscape_path(
        rmsd_a, rmsd_b, project_path=project_path, basis=basis
    )
    annotate_endpoint(ax, float(plot_x[0]), float(plot_y[0]), "Init", boxed=True)
    annotate_endpoint(ax, float(plot_x[-1]), float(plot_y[-1]), "Min", boxed=True)

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


if __name__ == "__main__":
    main()
