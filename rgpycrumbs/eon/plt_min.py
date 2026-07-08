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
#   "chemparseplot[neb,plot]>=1.8.0,<2",
#   "xyzrender>=0.1.3",
#   "readcon>=0.13.1",
#   "rgpycrumbs>=1.9.13",
# ]
# ///
# Optional deps (jax for landscapes) via uv PEP 723 or RGPYCRUMBS_AUTO_DEPS=1.

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
    warn_on_direct_script_import(__name__, "rgpycrumbs eon plt-min")

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
from chemparseplot.parse.eon.min_trajectory import load_min_trajectory
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
    help="Path to eOn minimization output directory. Repeat for overlay. "
    "Optional when [min].job_dir is set in --config.",
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
    "--energy-cap",
    type=float,
    default=None,
    help="Clip trajectory energies at this absolute ceiling (in --energy-unit) "
    "before the surface fit and color scale, so a few high-energy frames (e.g. "
    "repulsive starts) do not flatten the colormap. Overrides --energy-cap-window.",
)
@click.option(
    "--energy-cap-window",
    type=float,
    default=None,
    help="Clip trajectory energies to this window above the minimum energy "
    "(in --energy-unit). Ignored when --energy-cap is given.",
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
def main(  # noqa: PLR0913
    ctx,
    config,
    job_dir,
    label,
    prefix,
    plot_type,
    project_path,
    surface_type,
    ira_kmax,
    energy_unit,
    energy_cap,
    energy_cap_window,
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
        "min",
        ctx,
        config=config,
        job_dir=job_dir,
        label=label,
        prefix=prefix,
        plot_type=plot_type,
        project_path=project_path,
        surface_type=surface_type,
        ira_kmax=ira_kmax,
        energy_unit=energy_unit,
        energy_cap=energy_cap,
        energy_cap_window=energy_cap_window,
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
            "Provide --job-dir and/or set [min].job_dir in --config"
        )
    label = settings.get("label") or ()
    prefix = settings["prefix"]
    plot_type = settings["plot_type"]
    project_path = settings["project_path"]
    surface_type = settings["surface_type"]
    ira_kmax = settings["ira_kmax"]
    energy_unit = settings["energy_unit"]
    energy_cap = settings.get("energy_cap")
    energy_cap_window = settings.get("energy_cap_window")
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
    """Plot minimization trajectory visualization.

    Use --job-dir multiple times to overlay trajectories from different
    optimizers (e.g. FIRE, LBFGS, SD) on the same landscape or profile.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output = default_output_path("min", plot_type, output)
    trajs = load_trajectories(
        job_dir,
        lambda jd: load_min_trajectory(jd, prefix=prefix),
        log_info=log.info,
        noun="minimization trajectory",
        detail=lambda traj: f"{len(traj.atoms_list)} frames",
    )
    labels = overlay_labels(job_dir, label)

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
            energy_cap=energy_cap,
            energy_cap_window=energy_cap_window,
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
    energy_cap=None,
    energy_cap_window=None,
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
    ref_b = traj.final_atoms if traj.final_atoms is not None else traj.atoms_list[-1]
    # Primary label for this figure (single-job landscapes are the usual case).
    job_name = (labels[0] if labels else "minimization").strip() or "minimization"
    job_title = job_name[:1].upper() + job_name[1:]
    strip_structs = None
    strip_labels = None
    if plot_structures == "endpoints":
        strip_structs = [traj.initial_atoms, ref_b]
        # Short strip captions (job name is in the figure title already).
        strip_labels = ["initial", "minimized"]

    render_single_ended_landscape(
        atoms_list=traj.atoms_list,
        energies_eV=traj.dat_df["energy"].to_numpy(),
        ref_a=traj.initial_atoms,
        ref_b=ref_b,
        # Single-trajectory figures: do not re-plot the primary path as an overlay
        # (avoids a useless legend and double-drawing). Multi-job overlays still
        # pass every trajectory.
        overlay_atom_lists=[t.atoms_list for t in trajs] if len(trajs) > 1 else None,
        overlay_labels=labels if len(trajs) > 1 else None,
        ira_instance=ira_instance,
        ira_kmax=ira_kmax,
        project_path=project_path,
        surface_type=surface_type,
        energy_unit=energy_unit,
        energy_cap=energy_cap,
        energy_cap_window=energy_cap_window,
        relative_energy=True,
        title=f"{job_title} minimization",
        cmap=cmap,
        output=output,
        dpi=dpi,
        theme=theme,
        plot_structures=plot_structures,
        strip_structs=strip_structs,
        strip_labels=strip_labels,
        endpoint_start_label="initial",
        endpoint_end_label="minimized",
        endpoint_boxed=True,
        strip_renderer=strip_renderer,
        xyzrender_config=xyzrender_config,
        strip_spacing=max(strip_spacing, 2.2),
        strip_zoom=strip_zoom,
        strip_dividers=strip_dividers,
        rotation=rotation,
        perspective_tilt=perspective_tilt,
    )


def _plot_convergence(trajs, labels, output, dpi):
    plot_single_ended_convergence(trajs, labels, output, dpi)


if __name__ == "__main__":
    main()
