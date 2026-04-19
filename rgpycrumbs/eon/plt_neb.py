#!/usr/bin/env python3
"""Plots Nudged Elastic Band (NEB) reaction paths and landscapes.

.. versionadded:: 0.0.2

This script provides a command-line interface (CLI) to visualize data
generated from NEB calculations. It can plot:

1.  **Energy/Eigenvalue Profiles:** Shows the evolution of the energy or
    lowest eigenvalue along the reaction coordinate. It can overlay multiple
    paths (e.g., from different optimization steps) and use a
    physically-motivated Hermite spline interpolation using force data.

2.  **2D Reaction Landscapes:** Plots the path on a 2D coordinate system
    defined by the Root Mean Square Deviation (RMSD) from the reactant
    and product structures. This requires the 'ira_mod' library.
    It can also interpolate and display the 2D energy/eigenvalue surface.

The script can also render atomic structures from a .con file as insets
on the plots for key points (reactant, saddle, product).

This script follows the guidelines laid out here:
https://realpython.com/python-script-structure/
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "jax",
#   "adjustText",
#   "cmcrameri",
#   "rich",
#   "ase",
#   "polars",
#   "h5py",
#   "chemparseplot",
#   "xyzrender>=0.1.3",
#   "readcon>=0.7.0",
# ]
# ///

import logging
import sys
from pathlib import Path

import click
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from adjustText import adjust_text
from chemparseplot.parse.eon.neb import (
    aggregate_neb_landscape_data,
    compute_profile_rmsd,
    estimate_rbf_smoothing,
    load_structures_and_calculate_additional_rmsd,
)

# --- Library Imports ---
from chemparseplot.parse.file_ import find_file_paths
from chemparseplot.parse.trajectory.hdf5 import (
    history_to_landscape_df as hdf5_history_to_landscape_df,
)
from chemparseplot.parse.trajectory.hdf5 import (
    history_to_profile_dats,
    result_to_atoms_list,
)
from chemparseplot.parse.trajectory.hdf5 import (
    result_to_profile_dat as hdf5_result_to_profile_dat,
)
from chemparseplot.parse.trajectory.neb import (
    load_trajectory,
    trajectory_to_landscape_df,
    trajectory_to_profile_dat,
)
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ArrowStyle
from rich.logging import RichHandler

try:
    from chemparseplot.parse.projection import (
        compute_projection_basis,
        project_to_sd,
    )
except ImportError:
    compute_projection_basis = None
    project_to_sd = None
from chemparseplot.plot.neb import (
    plot_energy_path,
    plot_landscape_path_overlay,
    plot_landscape_surface,
    plot_mmf_peaks_overlay,
    plot_neb_evolution,
    plot_structure_inset,
    plot_structure_strip,
)
from chemparseplot.plot.structs import (
    convert_energy,
    convert_energy_curvature,
    eigenvalue_axis_label,
    energy_axis_label,
)
from chemparseplot.plot.theme import (
    apply_axis_theme,
    get_theme,
    setup_global_theme,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)
log = logging.getLogger("rich")


# --- Constants ---
DEFAULT_INPUT_PATTERN = "neb_*.dat"
DEFAULT_PATH_PATTERN = "neb_path_*.con"
IRA_KMAX_DEFAULT = 14.0


def _convert_neb_values(values, plot_mode: str, energy_unit: str):
    """Convert NEB data for the active plotted quantity."""

    if plot_mode == "energy":
        return convert_energy(values, energy_unit)
    return convert_energy_curvature(values, energy_unit)


def _default_neb_ylabel(plot_mode: str, energy_unit: str) -> str:
    """Return the canonical label for NEB energy-like axes."""

    if plot_mode == "energy":
        return energy_axis_label(energy_unit, label="Relative Energy")
    return eigenvalue_axis_label(energy_unit, label="Lowest Eigenvalue")


def _landscape_projection_basis(global_basis, final_r, final_p):
    """Reuse the full-dataset basis whenever projected overlays need one."""

    if global_basis is not None:
        return global_basis
    return compute_projection_basis(final_r, final_p)


def _landscape_half_span(x_limits, final_r, final_p, additional_atoms_data, global_basis):
    """Expand the symmetric landscape half-span to keep extra markers visible."""

    half_span = (x_limits[1] - x_limits[0]) / 2
    if not additional_atoms_data:
        return half_span

    basis = _landscape_projection_basis(global_basis, final_r, final_p)
    for _, add_r, add_p, _ in additional_atoms_data:
        _, add_d = project_to_sd(np.array([add_r]), np.array([add_p]), basis)
        half_span = max(half_span, abs(float(add_d[0])) * 1.15)
    return half_span


def _save_plot(output_file, dpi, *, has_strip):
    """Save plots without tight-bbox strip overflow."""

    save_kwargs = {"transparent": False, "pad_inches": 0.1, "dpi": dpi}
    if not has_strip:
        save_kwargs["bbox_inches"] = "tight"
    plt.savefig(output_file, **save_kwargs)


def _profile_structure_indices(atoms_list, y_values, plot_structures, plot_mode):
    """Select profile structures to render as a strip payload."""

    if plot_structures == "all":
        return list(range(len(atoms_list)))
    saddle_idx = (
        int(np.argmax(y_values[1:-1]) + 1)
        if plot_mode == "energy"
        else int(np.argmin(y_values))
    )
    return sorted({0, saddle_idx, len(atoms_list) - 1})


def _profile_strip_payload(atoms_list, x_values, y_values, plot_structures, plot_mode):
    """Build an ordered strip payload for profile plots."""

    payload = []
    for index in _profile_structure_indices(
        atoms_list, y_values, plot_structures, plot_mode
    ):
        if plot_structures == "all":
            label = str(index)
        elif index == 0:
            label = "R"
        elif index == len(atoms_list) - 1:
            label = "P"
        else:
            label = "SP"
        payload.append(
            {"atoms": atoms_list[index], "x": float(x_values[index]), "label": label}
        )
    return payload


# --- CLI ---
@click.command()
@click.option(
    "--input-dat-pattern",
    default=DEFAULT_INPUT_PATTERN,
    help="Glob pattern for input data files.",
)
@click.option(
    "--input-path-pattern",
    default=DEFAULT_PATH_PATTERN,
    help="Glob pattern for input path files.",
)
@click.option(
    "--con-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to .con trajectory file.",
)
@click.option(
    "--additional-con",
    type=(
        click.Path(exists=True, dir_okay=False, path_type=Path),
        str,
    ),  # Takes (Path, Label)
    multiple=True,
    default=None,
    help="Path(s) to additional .con file(s) and label.",
)
@click.option(
    "--augment-dat",
    type=str,
    default=None,
    help="Glob pattern for extra .dat files for surface fitting.",
)
@click.option(
    "--augment-con",
    type=str,
    default=None,
    help="Glob pattern for extra .con files for surface fitting.",
)
@click.option(
    "--sp-file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("sp.con"),
    help="Path to explicit saddle point file (eOn sp.con).",
)
@click.option(
    "--source",
    type=click.Choice(["eon", "traj", "hdf5"]),
    default="eon",
    help="Data source: 'eon' for .dat/.con pairs, 'traj' for extxyz, 'hdf5' for ChemGP HDF5.",
)
@click.option(
    "--input-h5",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to ChemGP NEB HDF5 file (result or history).",
)
@click.option(
    "--input-traj",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to extxyz trajectory file (used with --source traj).",
)
@click.option(
    "--plot-type",
    type=click.Choice(["profile", "landscape"]),
    default="profile",
    help="Type of plot to generate.",
)
@click.option(
    "--rbf-smoothing",
    type=float,
    default=None,
    show_default=True,
    help="Smoothing term for 2D RBF.",
)
@click.option(
    "--landscape-mode",
    type=click.Choice(["path", "surface"]),
    default="surface",
    help="For landscape plot: 'path' or 'surface'.",
)
@click.option(
    "--landscape-path",
    type=click.Choice(["last", "all"]),
    default="all",
    help="Last uses an interpolation only on the last path, otherwise use all points.",
)
@click.option(
    "--project-path/--no-project-path",
    is_flag=True,
    default=True,
    help="Project landscape coordinates into the reaction valley (s, d).",
)
@click.option(
    "--rc-mode",
    type=click.Choice(["path", "rmsd", "index"]),
    default="path",
    help="Reaction coordinate for profile plot.",
)
@click.option(
    "--plot-structures",
    type=click.Choice(["none", "all", "crit_points"]),
    default="none",
    help="Structures to render on the path. Requires --con-file.",
)
@click.option(
    "--surface-type",
    type=click.Choice(
        [
            "grid",
            "rbf",
            "grad_matern",
            "grad_imq",
            "grad_imq_ny",
            "matern",
            "imq",
            "grad_rq",
            "grad_se",
        ]
    ),
    default="rbf",
    help="Interpolation method for the 2D surface.",
)
@click.option(
    "--n-inducing",
    type=int,
    default=None,
    help="Number of inducing points for Nystrom or RFF features. Defaults to 300 (Nystrom) or 500 (RFF).",
)
@click.option(
    "--show-pts/--no-show-pts",
    default=True,
    help="Show all paths from the optimization on the RMSD 2D plot.",
)
@click.option(
    "--plot-mode",
    type=click.Choice(["energy", "eigenvalue"]),
    default="energy",
    help="Quantity to plot.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output image filename.",
)
@click.option(
    "--start", type=int, default=None, help="Start file index for profile plot."
)
@click.option("--end", type=int, default=None, help="End file index for profile plot.")
@click.option(
    "--normalize-rc", is_flag=True, default=False, help="Normalize reaction coordinate."
)
@click.option("--title", default="NEB Path", help="Plot title.")
@click.option("--xlabel", default=None, help="X-axis label.")
@click.option("--ylabel", default=None, help="Y-axis label.")
@click.option(
    "--energy-unit",
    type=click.Choice(["eV", "kcal/mol", "kJ/mol"]),
    default="eV",
    show_default=True,
    help="Presentation unit for energy-like axes and color scales.",
)
# --- Theme and Override Options ---
@click.option(
    "--theme",
    default="ruhi",
    help="The plotting theme to use.",
)
@click.option("--cmap-profile", default=None, help="Colormap for profile plot.")
@click.option("--cmap-landscape", default=None, help="Colormap for landscape plot.")
@click.option("--facecolor", type=str, default=None, help="Background color.")
@click.option("--fontsize-base", type=int, default=None, help="Base font size.")
# --- Figure and Inset Options ---
@click.option(
    "--figsize",
    nargs=2,
    type=(float, float),
    default=(5.37, 5.37),
    show_default=True,
    help="Figure width, height in inches.",
)
@click.option(
    "--fig-height",
    type=float,
    default=None,
    help="Figure height in inches.",
)
@click.option(
    "--aspect-ratio",
    type=float,
    default=None,
    help="Figure aspect ratio.",
)
@click.option(
    "--dpi",
    type=int,
    default=200,
    show_default=True,
    help="Resolution in Dots Per Inch.",
)
@click.option(
    "--zoom-ratio",
    type=float,
    default=0.4,
    show_default=True,
    help="Scale the inset image.",
)
@click.option(
    "--rotation",
    "ase_rotation",
    type=str,
    default="auto",
    show_default=True,
    help="Viewing rotation. 'auto' lets xyzrender auto-orient (default). ASE-style string e.g. '0x,90y,0z' for manual control.",
)
@click.option(
    "--perspective-tilt",
    type=float,
    default=0.0,
    show_default=True,
    help="Small off-axis tilt (degrees) to reveal occluded atoms. 5-10 is typical.",
)
@click.option(
    "--strip-renderer",
    type=click.Choice(["ase", "xyzrender", "solvis", "ovito"]),
    default="xyzrender",
    show_default=True,
    help=(
        "Rendering backend for structure images. "
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
@click.option(
    "--strip-spacing",
    type=float,
    default=1.5,
    show_default=True,
    help="Horizontal spacing between structure images.",
)
@click.option(
    "--strip-dividers/--no-strip-dividers",
    is_flag=True,
    default=False,
    help="Draw vertical divider lines between structures.",
)
@click.option(
    "--arrow-head-length",
    type=float,
    default=0.2,
    show_default=True,
    help="Arrow head length.",
)
@click.option(
    "--arrow-head-width",
    type=float,
    default=0.3,
    show_default=True,
    help="Arrow head width.",
)
@click.option(
    "--arrow-tail-width",
    type=float,
    default=0.1,
    show_default=True,
    help="Arrow tail width.",
)
# --- Path/Spline Options ---
@click.option(
    "--highlight-last/--no-highlight-last",
    is_flag=True,
    default=True,
    help="Highlight last path.",
)
@click.option(
    "--spline-method",
    type=click.Choice(["hermite", "spline"]),
    default="hermite",
    help="Spline interpolation method.",
)
# --- Inset Position Options ---
@click.option(
    "--draw-reactant",
    type=(float, float, float),
    nargs=3,
    default=(15, 60, 0.1),
    show_default=True,
    help="Reactant inset pos (x, y, rad).",
)
@click.option(
    "--draw-saddle",
    type=(float, float, float),
    nargs=3,
    default=(15, 60, 0.1),
    show_default=True,
    help="Saddle inset pos (x, y, rad).",
)
@click.option(
    "--draw-product",
    type=(float, float, float),
    nargs=3,
    default=(15, 60, 0.1),
    show_default=True,
    help="Product inset pos (x, y, rad).",
)
@click.option(
    "--cache-file",
    type=click.Path(path_type=Path),
    default=Path(".neb_landscape.parquet"),
    help="Parquet cache file.",
)
@click.option(
    "--force-recompute",
    is_flag=True,
    default=False,
    help="Force re-calculation of RMSD.",
)
@click.option(
    "--mmf-peaks/--no-mmf-peaks",
    is_flag=True,
    default=None,
    help="Overlay MMF peak positions on landscape (auto-detected if peak files exist).",
)
@click.option(
    "--peak-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing peak{NN}_pos.con files for MMF overlay.",
)
@click.option(
    "--show-evolution",
    is_flag=True,
    default=False,
    help="Show band evolution across iterations (requires write_movies data).",
)
@click.option(
    "--show-legend/--no-legend",
    default=True,
    help="Show the legends.",
)
@click.option(
    "--ira-kmax",
    default=IRA_KMAX_DEFAULT,
    help="kmax factor for IRA.",
)
def main(
    # --- Input Files ---
    input_dat_pattern,
    input_path_pattern,
    con_file,
    additional_con,
    # --- Data Source ---
    source,
    input_traj,
    input_h5,
    # --- Plot Behavior ---
    plot_type,
    landscape_mode,
    landscape_path,
    project_path,
    rc_mode,
    plot_structures,
    rbf_smoothing,
    show_pts,
    plot_mode,
    surface_type,
    n_inducing,
    # --- Output & Slicing ---
    output_file,
    start,
    end,
    # --- Plot Aesthetics ---
    normalize_rc,
    title,
    xlabel,
    ylabel,
    energy_unit,
    highlight_last,
    # --- Theme ---
    theme,
    cmap_profile,
    cmap_landscape,
    facecolor,
    fontsize_base,
    # --- Figure & Inset ---
    figsize,
    fig_height,
    aspect_ratio,
    dpi,
    zoom_ratio,
    ase_rotation,
    perspective_tilt,
    strip_renderer,
    xyzrender_config,
    strip_spacing,
    strip_dividers,
    arrow_head_length,
    arrow_head_width,
    arrow_tail_width,
    # --- Spline ---
    spline_method,
    # --- Inset Positions ---
    draw_reactant,
    draw_saddle,
    draw_product,
    # --- OCI-NEB/RONEB ---
    mmf_peaks,
    peak_dir,
    show_evolution,
    show_legend,
    # Caching
    cache_file,
    force_recompute,
    ira_kmax,
    sp_file,
    augment_dat,
    augment_con,
):
    """Main entry point for NEB plot script."""

    # 1. Setup Theme
    active_theme = get_theme(
        theme,
        cmap_profile=cmap_profile,
        cmap_landscape=cmap_landscape,
        font_size=fontsize_base,
        facecolor=facecolor,
    )
    setup_global_theme(active_theme)

    if fig_height and aspect_ratio:
        figsize = (fig_height * aspect_ratio, fig_height)
    elif fig_height or aspect_ratio:
        log.error(
            "Both --fig-height and --aspect-ratio must be provided together. Using default figsize."
        )

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Layout Logic
    has_strip = plot_structures in ["all", "crit_points"] and plot_type in {
        "landscape",
        "profile",
    }

    if has_strip:
        # Heuristic layout adjustment
        n_expected = (3 if plot_structures == "crit_points" else 10) + len(
            additional_con or []
        )
        max_cols = 6
        n_rows = (n_expected + max_cols - 1) // max_cols
        calc_hspace = 0.8 if n_rows > 1 else 0.3

        gs = GridSpec(2, 1, height_ratios=[1, 0.25], hspace=calc_hspace, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1])
        apply_axis_theme(ax_strip, active_theme)
    else:
        ax = fig.add_subplot(111)
        ax_strip = None

    apply_axis_theme(ax, active_theme)

    atoms_list = None
    additional_atoms_data = []
    sp_data = None

    # Only attempt to load structures if specifically requested or needed for the plot type
    if con_file:
        try:
            atoms_list, additional_atoms_data, sp_data = (
                load_structures_and_calculate_additional_rmsd(
                    con_file, additional_con, ira_kmax, sp_file
                )
            )
        except Exception as e:
            log.error(f"Error loading structures: {e}")
            # Critical failure for landscape/RMSD modes
            if plot_type == "landscape" or rc_mode == "rmsd":
                log.critical("Cannot proceed without structures. Exiting.")
                sys.exit(1)

    # --- Trajectory source: load once if applicable ---
    traj_atoms_list = None
    if source == "traj":
        if not input_traj:
            log.critical("--input-traj is required when --source traj is used.")
            sys.exit(1)
        traj_atoms_list = load_trajectory(str(input_traj))

    if plot_type == "landscape":
        # --- Landscape Plot ---
        z_label = _default_neb_ylabel(plot_mode, energy_unit)

        if source == "traj":
            df = trajectory_to_landscape_df(traj_atoms_list, ira_kmax=ira_kmax)
            # Use traj structures for con_file features when not provided
            if atoms_list is None:
                atoms_list = traj_atoms_list
        elif source == "hdf5":
            if not input_h5:
                log.critical("--input-h5 is required when --source hdf5 is used.")
                sys.exit(1)
            h5_str = str(input_h5)
            # Prefer history file for multi-step landscape
            try:
                df = hdf5_history_to_landscape_df(h5_str, ira_kmax=ira_kmax)
            except Exception:
                log.warning("History read failed, falling back to single-step result.")
                from chemparseplot.parse.trajectory.hdf5 import (
                    result_to_atoms_list as _r2a,
                )
                from chemparseplot.parse.trajectory.neb import (
                    trajectory_to_landscape_df as _traj_ldf,
                )

                hdf5_atoms = _r2a(h5_str)
                df = _traj_ldf(hdf5_atoms, ira_kmax=ira_kmax)
            if atoms_list is None:
                atoms_list = result_to_atoms_list(h5_str)
        else:
            dat_paths = find_file_paths(input_dat_pattern)
            con_paths = find_file_paths(str(input_path_pattern))

            if not dat_paths:
                log.critical(f"No data files found for pattern: {input_dat_pattern}")
                sys.exit(1)

            # Fallback if no path files found but main file exists
            if not con_paths and con_file:
                con_paths = [con_file]

            y_col = 2 if plot_mode == "energy" else 4

            df = aggregate_neb_landscape_data(
                dat_paths,
                con_paths,
                y_col,
                None,
                cache_file=cache_file,
                force_recompute=force_recompute,
                ira_kmax=ira_kmax,
                augment_dat=augment_dat,
                augment_con=augment_con,
                ref_atoms=atoms_list[0] if atoms_list else None,  # main reactant
                prod_atoms=atoms_list[-1] if atoms_list else None,  # main product
            )

        # Compute a SINGLE projection basis from the full dataset's endpoints.
        # This basis is used consistently for: surface grid, path overlay,
        # additional-con overlay, and viewport calculation.
        r_full = df["r"].to_numpy()
        p_full = df["p"].to_numpy()
        global_basis = compute_projection_basis(r_full, p_full) if project_path else None

        # Surface Generation
        if landscape_mode == "surface":
            if landscape_path == "last":
                max_step = df["step"].max()
                df_surface = df.filter(pl.col("step") == max_step)
            else:
                df_surface = df

            # Prepare arrays
            r_all = df_surface["r"].to_numpy()
            p_all = df_surface["p"].to_numpy()
            z_all = _convert_neb_values(
                df_surface["z"].to_numpy(), plot_mode, energy_unit
            )
            gr_all = _convert_neb_values(
                df_surface["grad_r"].to_numpy(), plot_mode, energy_unit
            )
            gp_all = _convert_neb_values(
                df_surface["grad_p"].to_numpy(), plot_mode, energy_unit
            )
            step_all = df_surface["step"].to_numpy()

            # Heuristic for RBF smoothing if missing
            if rbf_smoothing is None:
                rbf_smoothing = estimate_rbf_smoothing(df)
                log.info(f"Calculated heuristic RBF smoothing: {rbf_smoothing:.4f}")

            extra_pts = []
            if sp_data:
                extra_pts.append([sp_data["r"], sp_data["p"]])
            for _, add_r, add_p, _ in additional_atoms_data:
                extra_pts.append([add_r, add_p])
            extra_pts_arr = np.array(extra_pts) if extra_pts else None

            # Pre-compute viewport from FULL data (not filtered surface data)
            vp_xlim = vp_ylim = None
            if project_path and global_basis is not None:
                _s, _d = project_to_sd(r_full, p_full, global_basis)
                _s_pad = (_s.max() - _s.min()) * 0.1
                vp_xlim = (float(_s.min() - _s_pad), float(_s.max() + _s_pad))
                _half = (vp_xlim[1] - vp_xlim[0]) / 2
                # Make sure the trajectory itself fits inside the symmetric
                # d window before considering additional structures, so the
                # converged path and saddle marker do not get clipped at
                # the top/bottom of the viewport.
                _half = max(
                    _half, abs(float(_d.max())) * 1.15, abs(float(_d.min())) * 1.15
                )
                for _, add_r, add_p, _ in additional_atoms_data:
                    _, _ad = project_to_sd(
                        np.array([add_r]), np.array([add_p]), global_basis
                    )
                    _half = max(_half, abs(float(_ad[0])) * 1.15)
                vp_ylim = (-_half, _half)
                # Expand X to match Y so equal-aspect produces a square plot
                x_span = vp_xlim[1] - vp_xlim[0]
                if 2 * _half > x_span:
                    x_center = (vp_xlim[0] + vp_xlim[1]) / 2
                    vp_xlim = (x_center - _half, x_center + _half)

            plot_landscape_surface(
                ax,
                r_all,
                p_all,
                gr_all,
                gp_all,
                z_all,
                step_data=step_all,
                method=surface_type,
                rbf_smooth=rbf_smoothing,
                cmap=active_theme.cmap_landscape,
                show_pts=show_pts,
                # so we always show 5% and 95%, this is the user defined additional one
                # TODO(rg): just be a user parameter..
                variance_threshold=0.5,  # 50% uncertainty
                project_path=project_path,
                extra_points=extra_pts_arr,
                n_inducing=n_inducing,
                xlim=vp_xlim,
                ylim=vp_ylim,
                basis=global_basis,
            )

        # Path Overlay (Final Step)
        max_step = df["step"].max()
        df_final = df.filter(pl.col("step") == max_step)
        final_r = df_final["r"].to_numpy()
        final_p = df_final["p"].to_numpy()
        final_z = _convert_neb_values(df_final["z"].to_numpy(), plot_mode, energy_unit)

        # Pass all-iteration data for triangulated background when no GP surface
        bg_kwargs = {}
        if landscape_mode != "surface":
            bg_kwargs = {
                "all_r": df["r"].to_numpy(),
                "all_p": df["p"].to_numpy(),
                "all_z": _convert_neb_values(df["z"].to_numpy(), plot_mode, energy_unit),
            }

        plot_landscape_path_overlay(
            ax,
            final_r,
            final_p,
            final_z,
            active_theme.cmap_landscape,
            z_label,
            project_path=project_path,
            basis=global_basis,
            **bg_kwargs,
        )

        # --- OCI-NEB/RONEB: MMF Peaks Overlay ---
        _show_mmf = mmf_peaks
        _peak_search_dir = peak_dir or Path(".")
        if _show_mmf is None:
            # Auto-detect: check if peak files exist
            _show_mmf = len(list(_peak_search_dir.glob("peak*_pos.con"))) > 0
        if _show_mmf:
            peak_files = sorted(_peak_search_dir.glob("peak*_pos.con"))
            if peak_files:
                import readcon

                from rgpycrumbs.geom.api.alignment import calculate_rmsd_from_ref

                try:
                    from rgpycrumbs._aux import _import_from_parent_env

                    _ira_mod = _import_from_parent_env("ira_mod")
                    ira_instance = _ira_mod.IRA()
                except (ImportError, AttributeError):
                    ira_instance = None
                peak_atoms = [readcon.read_con_as_ase(str(pf))[0] for pf in peak_files]
                # Use same references as the main path
                ref_r = atoms_list[0] if atoms_list else None
                ref_p = atoms_list[-1] if atoms_list else None
                if ref_r is not None and ref_p is not None:
                    peak_rmsd_r = calculate_rmsd_from_ref(
                        peak_atoms, ira_instance, ref_atom=ref_r, ira_kmax=ira_kmax
                    )
                    peak_rmsd_p = calculate_rmsd_from_ref(
                        peak_atoms, ira_instance, ref_atom=ref_p, ira_kmax=ira_kmax
                    )
                    # Energies from peak structures (if available)
                    peak_e = convert_energy(
                        np.array(
                            [
                                a.get_potential_energy() if a.calc else 0.0
                                for a in peak_atoms
                            ]
                        ),
                        energy_unit,
                    )
                    plot_mmf_peaks_overlay(
                        ax,
                        peak_rmsd_r,
                        peak_rmsd_p,
                        peak_e,
                        project_path=project_path,
                        path_rmsd_r=final_r,
                        path_rmsd_p=final_p,
                    )
                    log.info("Plotted %d MMF peak(s)", len(peak_files))

        # --- OCI-NEB/RONEB: Band Evolution ---
        if show_evolution:
            unique_steps = sorted(df["step"].unique().to_list())
            if len(unique_steps) > 1:
                step_r_list = []
                step_p_list = []
                for step in unique_steps:
                    step_df = df.filter(pl.col("step") == step)
                    step_r_list.append(step_df["r"].to_numpy())
                    step_p_list.append(step_df["p"].to_numpy())
                plot_neb_evolution(
                    ax,
                    step_r_list,
                    step_p_list,
                    project_path=project_path,
                )
                log.info("Plotted band evolution (%d steps)", len(unique_steps))

        # Saddle Point Marker
        if sp_data:
            # Use explicit SP coordinates
            sp_x_raw, sp_y_raw = sp_data["r"], sp_data["p"]
            log.info(f"Plotting explicit SP at R={sp_x_raw:.3f}, P={sp_y_raw:.3f}")
        else:
            # Fallback to heuristic
            if plot_mode == "energy":
                saddle_idx = np.argmax(final_z[1:-1]) + 1
            else:
                saddle_idx = np.argmin(final_z)
            sp_x_raw, sp_y_raw = final_r[saddle_idx], final_p[saddle_idx]

        # Apply projection to saddle point if enabled
        if project_path:
            _sp_basis = _landscape_projection_basis(global_basis, final_r, final_p)
            sp_sd = project_to_sd(np.array([sp_x_raw]), np.array([sp_y_raw]), _sp_basis)
            sp_x, sp_y = float(sp_sd[0][0]), float(sp_sd[1][0])
        else:
            sp_x, sp_y = sp_x_raw, sp_y_raw

        ax.scatter(
            sp_x,
            sp_y,
            marker="*",
            s=int(active_theme.font_size**2 * 1.5),
            c="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=100,
            label="SP",
        )

        if additional_atoms_data:
            marker_cmap = mpl.colormaps.get_cmap("tab10")
            for i, (_, add_r, add_p, add_label) in enumerate(additional_atoms_data):
                color = marker_cmap(i % 10)

                if project_path:
                    _add_basis = _landscape_projection_basis(
                        global_basis, final_r, final_p
                    )
                    _s, _d = project_to_sd(
                        np.array([add_r]), np.array([add_p]), _add_basis
                    )
                    plot_add_r, plot_add_p = float(_s[0]), float(_d[0])
                else:
                    plot_add_r, plot_add_p = add_r, add_p

                ax.plot(
                    plot_add_r,
                    plot_add_p,
                    marker="*",
                    markersize=int(active_theme.font_size * 1.1),
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    linestyle="None",
                    zorder=102,
                    label=add_label,
                )

        if has_strip and atoms_list:
            strip_payload = []

            # Helper to calculate projected coordinates for labels
            def get_projected_coords(r_val, p_val):
                if project_path:
                    _pc_basis = _landscape_projection_basis(
                        global_basis, final_r, final_p
                    )
                    _s, _d = project_to_sd(
                        np.array([r_val]), np.array([p_val]), _pc_basis
                    )
                    return float(_s[0]), float(_d[0])
                return r_val, p_val

            # Add Reactant
            rx, ry = get_projected_coords(final_r[0], final_p[0])
            strip_payload.append({"atoms": atoms_list[0], "x": rx, "y": ry, "label": "R"})

            # Add Saddle (Explicit or Heuristic)
            if sp_data:
                sx, sy = get_projected_coords(sp_data["r"], sp_data["p"])
                strip_payload.append(
                    {"atoms": sp_data["atoms"], "x": sx, "y": sy, "label": "SP"}
                )
            else:
                s_idx = (
                    (np.argmax(final_z[1:-1]) + 1)
                    if plot_mode == "energy"
                    else np.argmin(final_z)
                )
                sx, sy = get_projected_coords(final_r[s_idx], final_p[s_idx])
                strip_payload.append(
                    {"atoms": atoms_list[s_idx], "x": sx, "y": sy, "label": "SP"}
                )

            # Add Product
            px, py = get_projected_coords(final_r[-1], final_p[-1])
            strip_payload.append(
                {"atoms": atoms_list[-1], "x": px, "y": py, "label": "P"}
            )

            # Add intermediate points if 'all' requested
            if plot_structures == "all":
                for i in range(1, len(atoms_list) - 1):
                    ix, iy = get_projected_coords(final_r[i], final_p[i])
                    strip_payload.append(
                        {"atoms": atoms_list[i], "x": ix, "y": iy, "label": str(i)}
                    )

            # Add additional structures
            for add_atoms, add_r, add_p, add_label in additional_atoms_data:
                ax_r, ax_p = get_projected_coords(add_r, add_p)
                strip_payload.append(
                    {
                        "atoms": add_atoms,
                        "x": ax_r,
                        "y": ax_p,
                        "label": add_label,
                    }
                )

            strip_payload.sort(key=lambda d: d["x"])
            labels = [d["label"] for d in strip_payload]
            structs = [d["atoms"] for d in strip_payload]

            plot_structure_strip(
                ax_strip,
                structs,
                labels,
                zoom=zoom_ratio,
                rotation=ase_rotation,
                theme_color=active_theme.textcolor,
                renderer=strip_renderer,
                xyzrender_config=xyzrender_config,
                col_spacing=strip_spacing,
                show_dividers=strip_dividers,
                perspective_tilt=perspective_tilt,
            )

            # Annotate Main Plot -- only label R, SP, P (not additional con;
            # those are identified by the legend markers instead)
            main_plot_texts = []
            main_labels = {"R", "SP", "P"}
            for d in strip_payload:
                if d["label"] not in main_labels:
                    continue
                t = ax.text(
                    d["x"],
                    d["y"],
                    d["label"],
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    ha="center",
                    va="bottom",
                    zorder=102,
                )
                t.set_path_effects(
                    [path_effects.withStroke(linewidth=2.5, foreground="black")]
                )
                main_plot_texts.append(t)

            if main_plot_texts:
                adjust_text(
                    main_plot_texts,
                    ax=ax,
                    arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.0},
                    expand_points=(2.0, 2.0),
                    force_text=(1.0, 2.0),
                    force_points=(1.0, 2.0),
                )

        # Labels
        if project_path:
            final_xlabel = xlabel or r"Reaction progress $s$ ($\AA$)"
            final_ylabel = ylabel or r"Orthogonal deviation $d$ ($\AA$)"
            final_title = "Reaction Valley Projection" if title == "NEB Path" else title
        else:
            final_xlabel = xlabel or r"RMSD from Reactant ($\AA$)"
            final_ylabel = ylabel or r"RMSD from Product ($\AA$)"
            final_title = "RMSD(R,P) projection" if title == "NEB Path" else title

    else:
        # --- Profile Plot ---
        profile_strip_payload = []
        if source == "hdf5":
            if not input_h5:
                log.critical("--input-h5 is required when --source hdf5 is used.")
                sys.exit(1)
            h5_str = str(input_h5)
            # Use history final step if available, else result
            try:
                dats = history_to_profile_dats(h5_str)
                data = dats[-1]
            except Exception:
                data = hdf5_result_to_profile_dat(h5_str)
            if atoms_list is None:
                atoms_list = result_to_atoms_list(h5_str)

            if rc_mode == "index":
                data[1] = np.arange(data.shape[1])
            elif normalize_rc:
                data[1] = data[1] / data[1].max() if data[1].max() > 0 else data[1]

            y_col = 2 if plot_mode == "energy" else 4
            data[y_col] = _convert_neb_values(data[y_col], plot_mode, energy_unit)
            if plot_mode == "energy":
                data[3] = convert_energy(data[3], energy_unit)
            color = active_theme.highlight_color
            plot_energy_path(
                ax,
                data[1],
                data[y_col],
                data[3],
                color,
                1.0,
                20,
                method=spline_method,
            )
        elif source == "traj":
            # Trajectory source: single extxyz file -> one profile
            data = trajectory_to_profile_dat(traj_atoms_list)
            if atoms_list is None:
                atoms_list = traj_atoms_list

            if rc_mode == "index":
                data[1] = np.arange(data.shape[1])
            elif normalize_rc:
                data[1] = data[1] / data[1].max() if data[1].max() > 0 else data[1]

            y_col = 2 if plot_mode == "energy" else 4
            data[y_col] = _convert_neb_values(data[y_col], plot_mode, energy_unit)
            if plot_mode == "energy":
                data[3] = convert_energy(data[3], energy_unit)
            color = active_theme.highlight_color
            plot_energy_path(
                ax,
                data[1],
                data[y_col],
                data[3],
                color,
                1.0,
                20,
                method=spline_method,
            )

            if atoms_list and plot_structures != "none":
                if has_strip:
                    profile_strip_payload.extend(
                        _profile_strip_payload(
                            atoms_list,
                            data[1],
                            data[y_col],
                            plot_structures,
                            plot_mode,
                        )
                    )
                else:
                    indices = _profile_structure_indices(
                        atoms_list, data[y_col], plot_structures, plot_mode
                    )
                    for i in indices:
                        if i == 0:
                            xybox, rad = draw_reactant[:2], draw_reactant[2]
                        elif i == len(atoms_list) - 1:
                            xybox, rad = draw_product[:2], draw_product[2]
                        else:
                            xybox, rad = draw_saddle[:2], draw_saddle[2]

                        if plot_structures == "all":
                            xybox = (15.0, 60.0 if i % 2 == 0 else -60.0)
                            rad = 0.1 if i % 2 == 0 else -0.1

                        plot_structure_inset(
                            ax,
                            atoms_list[i],
                            data[1][i],
                            data[y_col][i],
                            xybox,
                            rad,
                            zoom=zoom_ratio,
                            rotation=ase_rotation,
                            renderer=strip_renderer,
                            xyzrender_config=xyzrender_config,
                            perspective_tilt=perspective_tilt,
                        )
        else:
            # eOn source: multiple .dat files
            dat_paths = find_file_paths(input_dat_pattern)
            file_paths_to_plot = dat_paths[start:end]

            if not file_paths_to_plot:
                log.error("No files found in range.")
                sys.exit(1)

            # Optional: Load RMSD for X-axis
            rmsd_rc = None
            if rc_mode == "rmsd" and atoms_list:
                df_rmsd = compute_profile_rmsd(
                    atoms_list,
                    cache_file=cache_file,
                    force_recompute=force_recompute,
                    ira_kmax=ira_kmax,
                )
                rmsd_rc = df_rmsd["r"].to_numpy()

            # Plot Loop
            cm = plt.get_cmap(active_theme.cmap_profile)
            color_divisor = (
                len(file_paths_to_plot) - 1 if len(file_paths_to_plot) > 1 else 1.0
            )

            y_col = 2 if plot_mode == "energy" else 4

            for idx, fpath in enumerate(file_paths_to_plot):
                try:
                    data = np.loadtxt(fpath, skiprows=1).T
                except Exception as ex:
                    log.error(ex)
                    continue

                # X-Axis Logic
                if rc_mode == "rmsd" and rmsd_rc is not None:
                    if len(rmsd_rc) == data.shape[1]:
                        data[1] = rmsd_rc
                elif rc_mode == "index":
                    data[1] = np.arange(data.shape[1])
                elif normalize_rc:
                    data[1] = data[1] / data[1].max() if data[1].max() > 0 else data[1]

                # Style Logic
                is_last = idx == len(file_paths_to_plot) - 1
                if highlight_last and is_last:
                    color, alpha, zorder = active_theme.highlight_color, 1.0, 20
                    step_label = f"Step {idx + 1} (final)"
                else:
                    color = cm(idx / color_divisor)
                    alpha = 1.0 if idx == 0 else 0.5
                    zorder = 10 if idx == 0 else 5
                    step_label = f"Step {idx + 1}" if idx == 0 else None

                # Plot
                data[y_col] = _convert_neb_values(data[y_col], plot_mode, energy_unit)
                if plot_mode == "energy":
                    data[3] = convert_energy(data[3], energy_unit)
                plot_energy_path(
                    ax,
                    data[1],
                    data[y_col],
                    data[3],  # Forces
                    color,
                    alpha,
                    zorder,
                    method=spline_method,
                    label=step_label,
                )

                if (
                    highlight_last
                    and is_last
                    and atoms_list
                    and plot_structures != "none"
                ):
                    if has_strip:
                        profile_strip_payload.extend(
                            _profile_strip_payload(
                                atoms_list,
                                data[1],
                                data[y_col],
                                plot_structures,
                                plot_mode,
                            )
                        )
                    else:
                        indices = _profile_structure_indices(
                            atoms_list, data[y_col], plot_structures, plot_mode
                        )
                        for i in indices:
                            if i == 0:
                                xybox, rad = draw_reactant[:2], draw_reactant[2]
                            elif i == len(atoms_list) - 1:
                                xybox, rad = draw_product[:2], draw_product[2]
                            else:
                                xybox, rad = draw_saddle[:2], draw_saddle[2]

                            if plot_structures == "all":
                                xybox = (15.0, 60.0 if i % 2 == 0 else -60.0)
                                rad = 0.1 if i % 2 == 0 else -0.1

                            # Call library function
                            plot_structure_inset(
                                ax,
                                atoms_list[i],
                                data[1][i],
                                data[y_col][i],
                                xybox,
                                rad,
                                zoom=zoom_ratio,
                                rotation=ase_rotation,
                                renderer=strip_renderer,
                                xyzrender_config=xyzrender_config,
                            )

        # --- Profile Additional Structures ---
        if additional_atoms_data and rc_mode == "rmsd":
            for i, (add_atoms, add_r, _) in enumerate(additional_atoms_data):
                ax.axvline(
                    add_r,
                    color=active_theme.gridcolor,
                    linestyle=":",
                    linewidth=2,
                    zorder=90,
                )
                if has_strip:
                    profile_strip_payload.append(
                        {"atoms": add_atoms, "x": float(add_r), "label": f"Add {i + 1}"}
                    )
                elif plot_structures != "none":
                    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_pos = ax.get_ylim()[0] + 0.9 * y_span
                    plot_structure_inset(
                        ax,
                        add_atoms,
                        add_r,
                        y_pos,
                        xybox=(
                            draw_saddle[0] + (i * 15),
                            draw_saddle[1],
                        ),  # Stagger slightly
                        rad=draw_saddle[2],
                        zoom=zoom_ratio,
                        rotation=ase_rotation,
                        renderer=strip_renderer,
                        xyzrender_config=xyzrender_config,
                        arrow_props={
                            "arrowstyle": ArrowStyle.Fancy(
                                head_length=arrow_head_length,
                                head_width=arrow_head_width,
                                tail_width=arrow_tail_width,
                            ),
                            "connectionstyle": f"arc3,rad={rad}",
                            "linestyle": "-",
                            "alpha": 0.8,
                            "color": "black",
                            "linewidth": 1.2,
                        },
                    )

        if has_strip and profile_strip_payload:
            deduped_payload = []
            seen = set()
            for entry in sorted(profile_strip_payload, key=lambda d: d["x"]):
                key = (entry["label"], round(entry["x"], 8))
                if key in seen:
                    continue
                seen.add(key)
                deduped_payload.append(entry)

            plot_structure_strip(
                ax_strip,
                [entry["atoms"] for entry in deduped_payload],
                [entry["label"] for entry in deduped_payload],
                zoom=zoom_ratio,
                rotation=ase_rotation,
                theme_color=active_theme.textcolor,
                renderer=strip_renderer,
                xyzrender_config=xyzrender_config,
                col_spacing=strip_spacing,
                show_dividers=strip_dividers,
                perspective_tilt=perspective_tilt,
            )

        # Profile Labels
        final_xlabel = xlabel or (
            r"RMSD ($\AA$)" if rc_mode == "rmsd" else r"Reaction Coordinate ($\AA$)"
        )
        final_ylabel = ylabel or _default_neb_ylabel(plot_mode, energy_unit)
        final_title = title

    # Final Aesthetics
    ax.set_xlabel(final_xlabel, weight="bold")
    ax.set_ylabel(final_ylabel, weight="bold")
    ax.set_title(final_title)
    ax.minorticks_on()

    if plot_type == "landscape" and not aspect_ratio:
        ax.set_aspect("equal")
        if project_path:
            # Force Y-axis to be symmetric and match the X-axis span,
            # but expand if additional structures fall outside
            x_min, x_max = ax.get_xlim()
            half_span = _landscape_half_span(
                (x_min, x_max),
                final_r,
                final_p,
                additional_atoms_data,
                global_basis,
            )
            ax.set_ylim(-half_span, half_span)
            log.info(f"Set symmetric Y-axis limits: [-{half_span:.2f}, {half_span:.2f}]")

    if show_legend:
        ax.legend(
            # In (s,d) space markers can appear anywhere, so let
            # matplotlib pick the least-overlapping corner.
            # In raw RMSD-RMSD the lower left is always empty.
            loc="best" if project_path else "lower left",
            borderaxespad=0.5,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            fontsize=int(active_theme.font_size * 0.8),
        ).set_zorder(101)

    if not ax_strip:
        plt.tight_layout(pad=0.5)

    if ax_strip:
        pos_main = ax.get_position()
        pos_strip = ax_strip.get_position()

        # Force strip to match the main plot's Left and Width exactly,
        # and push it down slightly to avoid overlapping the xlabel
        strip_y = pos_strip.y0 - 0.02
        ax_strip.set_position([pos_main.x0, strip_y, pos_main.width, pos_strip.height])
        # Clip AnnotationBbox images to prevent bbox_inches="tight" from
        # expanding the figure, but leave Text labels unclipped.
        from matplotlib.offsetbox import AnnotationBbox

        for artist in ax_strip.get_children():
            if isinstance(artist, AnnotationBbox):
                artist.set_clip_on(True)

    if output_file:
        _save_plot(output_file, dpi, has_strip=ax_strip is not None)
    else:
        plt.show()


if __name__ == "__main__":
    main()
