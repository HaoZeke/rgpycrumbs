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
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - direct script execution without package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs eon plt-saddle")

try:
    from ._render_cli import add_config_option, add_render_options
    from ._single_ended_cli import default_output_path, load_trajectories, overlay_labels
    from .plot_config import resolve_from_click
except ImportError:  # pragma: no cover - direct script execution
    from rgpycrumbs.eon._render_cli import add_config_option, add_render_options
    from rgpycrumbs.eon._single_ended_cli import (
        default_output_path,
        load_trajectories,
        overlay_labels,
    )
    from rgpycrumbs.eon.plot_config import resolve_from_click
from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory
from chemparseplot.plot.optimization import (
    plot_single_ended_convergence,
    plot_single_ended_profile,
    render_single_ended_landscape,
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
@click.pass_context
@add_config_option
@click.option(
    "--job-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
    multiple=True,
    help="Path to eOn saddle search output directory. Repeat for overlay. "
    "Optional when [saddle].job_dir is set in --config.",
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
    ctx,
    config,
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
    settings = resolve_from_click(
        "saddle",
        ctx,
        config=config,
        job_dir=job_dir,
        label=label,
        plot_type=plot_type,
        ref_product=ref_product,
        project_path=project_path,
        surface_type=surface_type,
        ira_kmax=ira_kmax,
        energy_unit=energy_unit,
        theme=theme,
        plot_structures=plot_structures,
        strip_renderer=strip_renderer,
        xyzrender_config=xyzrender_config,
        strip_spacing=strip_spacing,
        strip_zoom=strip_zoom,
        strip_dividers=strip_dividers,
        rotation=rotation,
        perspective_tilt=perspective_tilt,
        output=output,
        dpi=dpi,
        verbose=verbose,
    )
    job_dir = settings.get("job_dir") or ()
    if not job_dir:
        raise click.UsageError(
            "Provide --job-dir and/or set [saddle].job_dir in --config"
        )
    label = settings.get("label") or ()
    plot_type = settings["plot_type"]
    ref_product = settings.get("ref_product")
    project_path = settings["project_path"]
    surface_type = settings["surface_type"]
    ira_kmax = settings["ira_kmax"]
    energy_unit = settings["energy_unit"]
    theme = settings["theme"]
    plot_structures = settings["plot_structures"]
    strip_renderer = settings["strip_renderer"]
    xyzrender_config = settings["xyzrender_config"]
    strip_spacing = settings["strip_spacing"]
    strip_zoom = settings.get("strip_zoom")
    strip_dividers = settings["strip_dividers"]
    rotation = settings["rotation"]
    perspective_tilt = settings["perspective_tilt"]
    output = settings.get("output")
    dpi = settings["dpi"]
    verbose = settings["verbose"]
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

    if ref_product is not None:
        ref_b = readcon.read_con_as_ase(str(ref_product))[0]
    elif traj.saddle_atoms is not None:
        ref_b = traj.saddle_atoms
    else:
        ref_b = traj.atoms_list[-1]

    end_label = "SP" if traj.saddle_atoms is not None else "End"
    strip_structs = None
    strip_labels = None
    if plot_structures == "endpoints":
        strip_structs = [traj.initial_atoms, ref_b]
        strip_labels = ["R", end_label]

    render_single_ended_landscape(
        atoms_list=traj.atoms_list,
        energies_eV=traj.dat_df["delta_e"].to_numpy(),
        ref_a=traj.initial_atoms,
        ref_b=ref_b,
        overlay_atom_lists=[t.atoms_list for t in trajs],
        overlay_labels=labels,
        ira_instance=ira_instance,
        ira_kmax=ira_kmax,
        project_path=project_path,
        surface_type=surface_type,
        energy_unit=energy_unit,
        cmap=cmap,
        output=output,
        dpi=dpi,
        theme=theme,
        plot_structures=plot_structures,
        strip_structs=strip_structs,
        strip_labels=strip_labels,
        endpoint_start_label="R",
        endpoint_end_label=end_label,
        endpoint_boxed=False,
        annotate_overlay_starts=True,
        overlay_start_label="R",
        strip_renderer=strip_renderer,
        xyzrender_config=xyzrender_config,
        strip_spacing=strip_spacing,
        strip_zoom=strip_zoom,
        strip_dividers=strip_dividers,
        rotation=rotation,
        perspective_tilt=perspective_tilt,
    )


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
