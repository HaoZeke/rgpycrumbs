#!/usr/bin/env python3
"""
Plots Nudged Elastic Band (NEB) reaction paths and landscapes.

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
#   "chemparseplot @ /home/rgoswami/Git/Github/Python/chemparseplot",
#   "rgpycrumbs",
# ]
# ///

import io
import logging
from pathlib import Path

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import polars as pl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ArrowStyle
from rich.logging import RichHandler

from adjustText import adjust_text

# --- Library Imports ---
from chemparseplot.parse.file_ import find_file_paths
from chemparseplot.parse.eon.neb import (
    aggregate_neb_landscape_data,
    compute_profile_rmsd,
    load_structures_and_calculate_additional_rmsd,
    load_augmenting_neb_data,
    estimate_rbf_smoothing,
)
from chemparseplot.plot.neb import (
    InsetImagePos,
    plot_energy_path,
    plot_landscape_path_overlay,
    plot_landscape_surface,
    plot_structure_inset,
    plot_structure_strip,
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
ROUNDING_DF = 3
IRA_KMAX_DEFAULT = 1.8


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
    "--rounding",
    type=int,
    default=ROUNDING_DF,
    show_default=True,
    help="Data rounding term for 2D plots.",
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
    type=click.Choice(["grid", "rbf", "grad_matern", "grad_imq", "matern", "imq", "grad_rq", "grad_se"]),
    default="rbf",
    help="Interpolation method for the 2D surface.",
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
    "--ase-rotation",
    type=str,
    default="0x, 90y, 0z",
    show_default=True,
    help="ASE rotation string.",
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
@click.option(
    "--savgol-window",
    type=int,
    default=5,
    help="Savitzky-Golay filter window.",
)
@click.option(
    "--savgol-order",
    type=int,
    default=2,
    help="Savitzky-Golay filter order.",
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
    "--show-legend",
    is_flag=True,
    default=False,
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
    # --- Plot Behavior ---
    plot_type,
    landscape_mode,
    landscape_path,
    project_path,
    rc_mode,
    plot_structures,
    rbf_smoothing,
    rounding,
    show_pts,
    plot_mode,
    surface_type,
    # --- Output & Slicing ---
    output_file,
    start,
    end,
    # --- Plot Aesthetics ---
    normalize_rc,
    title,
    xlabel,
    ylabel,
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
    arrow_head_length,
    arrow_head_width,
    arrow_tail_width,
    # --- Spline ---
    spline_method,
    savgol_window,
    savgol_order,
    # --- Inset Positions ---
    draw_reactant,
    draw_saddle,
    draw_product,
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
    has_strip = plot_structures in ["all", "crit_points"] and plot_type == "landscape"

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
                exit(1)

    if plot_type == "landscape":
        # --- Landscape Plot ---
        dat_paths = find_file_paths(input_dat_pattern)
        con_paths = find_file_paths(str(input_path_pattern))

        if not dat_paths:
            log.critical(f"No data files found for pattern: {input_dat_pattern}")
            exit(1)

        # Fallback if no path files found but main file exists
        if not con_paths and con_file:
            con_paths = [con_file]

        y_col = 2 if plot_mode == "energy" else 4
        z_label = (
            "Relative Energy (eV)"
            if plot_mode == "energy"
            else r"Lowest Eigenvalue (eV/$\AA^2$)"
        )

        df = aggregate_neb_landscape_data(
            dat_paths, 
            con_paths, 
            y_col, 
            None, 
            cache_file, 
            force_recompute, 
            ira_kmax,
            augment_dat=augment_dat,
            augment_con=augment_con,
            ref_atoms=atoms_list[0] if atoms_list else None, # main reactant
            prod_atoms=atoms_list[-1] if atoms_list else None, # main product
        )

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
            z_all = df_surface["z"].to_numpy()
            gr_all = df_surface["grad_r"].to_numpy()
            gp_all = df_surface["grad_p"].to_numpy()
            step_all = df_surface["step"].to_numpy()

            # Heuristic for RBF smoothing if missing
            if rbf_smoothing is None:
                rbf_smoothing = estimate_rbf_smoothing(df)
                log.info(f"Calculated heuristic RBF smoothing: {rbf_smoothing:.4f}")

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
                variance_threshold=0.5, # 50% uncertainity
                project_path=project_path,
            )

        # Path Overlay (Final Step)
        max_step = df["step"].max()
        df_final = df.filter(pl.col("step") == max_step)
        final_r = df_final["r"].to_numpy()
        final_p = df_final["p"].to_numpy()
        final_z = df_final["z"].to_numpy()

        plot_landscape_path_overlay(
            ax, final_r, final_p, final_z, active_theme.cmap_landscape, z_label, project_path=project_path
        )

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
            r_start, p_start = final_r[0], final_p[0]
            r_end, p_end = final_r[-1], final_p[-1]
            vec_r = r_end - r_start
            vec_p = p_end - p_start
            path_norm = np.hypot(vec_r, vec_p)
            u_r, u_p = vec_r / path_norm, vec_p / path_norm
            v_r, v_p = -u_p, u_r
            
            sp_x = (sp_x_raw - r_start) * u_r + (sp_y_raw - p_start) * u_p
            sp_y = (sp_x_raw - r_start) * v_r + (sp_y_raw - p_start) * v_p
        else:
            sp_x, sp_y = sp_x_raw, sp_y_raw

        ax.scatter(
            sp_x,
            sp_y,
            marker="s",
            s=int(active_theme.font_size * 2),
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
                    plot_add_r = (add_r - r_start) * u_r + (add_p - p_start) * u_p
                    plot_add_p = (add_r - r_start) * v_r + (add_p - p_start) * v_p
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
                    s_val = (r_val - r_start) * u_r + (p_val - p_start) * u_p
                    d_val = (r_val - r_start) * v_r + (p_val - p_start) * v_p
                    return s_val, d_val
                return r_val, p_val
            
            # Add Reactant
            rx, ry = get_projected_coords(final_r[0], final_p[0])
            strip_payload.append({
                "atoms": atoms_list[0],
                "x": rx,
                "y": ry,
                "label": "R"
            })

            # Add Saddle (Explicit or Heuristic)
            if sp_data:
                sx, sy = get_projected_coords(sp_data["r"], sp_data["p"])
                strip_payload.append({
                    "atoms": sp_data["atoms"],
                    "x": sx,
                    "y": sy,
                    "label": "SP"
                })
            else:
                s_idx = (np.argmax(final_z[1:-1]) + 1) if plot_mode == "energy" else np.argmin(final_z)
                sx, sy = get_projected_coords(final_r[s_idx], final_p[s_idx])
                strip_payload.append({
                    "atoms": atoms_list[s_idx],
                    "x": sx,
                    "y": sy,
                    "label": "SP"
                })

            # Add Product
            px, py = get_projected_coords(final_r[-1], final_p[-1])
            strip_payload.append({
                "atoms": atoms_list[-1],
                "x": px,
                "y": py,
                "label": "P"
            })

            # Add intermediate points if 'all' requested
            if plot_structures == "all":
                for i in range(1, len(atoms_list) - 1):
                    ix, iy = get_projected_coords(final_r[i], final_p[i])
                    strip_payload.append({
                        "atoms": atoms_list[i],
                        "x": ix,
                        "y": iy,
                        "label": str(i)
                    })

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
            )

            # Annotate Main Plot
            main_plot_texts = []
            for d in strip_payload:
                t = ax.text(
                    d["x"],
                    d["y"],
                    d["label"],
                    fontsize=12,
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
                    arrowprops=dict(arrowstyle="-", color="white", lw=1.5),
                    expand_points=(1.5, 1.5),
                    force_text=0.5,
                )

        # Labels
        if project_path:
            final_xlabel = xlabel or r"Reaction Progress, $s$ ($\AA$)"
            final_ylabel = ylabel or r"Orthogonal Distance, $d$ ($\AA$)"
            final_title = "Reaction Valley Projection" if title == "NEB Path" else title
        else:
            final_xlabel = xlabel or r"RMSD from Reactant ($\AA$)"
            final_ylabel = ylabel or r"RMSD from Product ($\AA$)"
            final_title = "RMSD(R,P) projection" if title == "NEB Path" else title

    else:
        # --- Profile Plot ---
        dat_paths = find_file_paths(input_dat_pattern)
        file_paths_to_plot = dat_paths[start:end]

        if not file_paths_to_plot:
            log.error("No files found in range.")
            exit(1)

        # Optional: Load RMSD for X-axis
        rmsd_rc = None
        if rc_mode == "rmsd" and atoms_list:
            df_rmsd = compute_profile_rmsd(
                atoms_list, cache_file, force_recompute, ira_kmax
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
            except Exception:
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
            else:
                color = cm(idx / color_divisor)
                alpha = 1.0 if idx == 0 else 0.5
                zorder = 10 if idx == 0 else 5

            # Plot
            plot_energy_path(
                ax,
                data[1],
                data[y_col],
                data[3],  # Forces
                color,
                alpha,
                zorder,
                method=spline_method,
            )

            if highlight_last and is_last and atoms_list and plot_structures != "none":
                indices = (
                    list(range(len(atoms_list)))
                    if plot_structures == "all"
                    else sorted(
                        list(
                            {
                                0,
                                np.argmax(data[y_col][1:-1]) + 1
                                if plot_mode == "energy"
                                else np.argmin(data[y_col]),
                                len(atoms_list) - 1,
                            }
                        )
                    )
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
                    )

        # --- Profile Additional Structures ---
        if additional_atoms_data and rc_mode == "rmsd":
            for i, (add_atoms, add_r, _, add_label) in enumerate(additional_atoms_data):
                ax.axvline(
                    add_r,
                    color=active_theme.gridcolor,
                    linestyle=":",
                    linewidth=2,
                    zorder=90,
                )
                if plot_structures != "none":
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

        # Profile Labels
        final_xlabel = xlabel or (
            r"RMSD ($\AA$)" if rc_mode == "rmsd" else r"Reaction Coordinate ($\AA$)"
        )
        final_ylabel = ylabel or "Relative Energy (eV)"
        final_title = title

    # Final Aesthetics
    ax.set_xlabel(final_xlabel, weight="bold")
    ax.set_ylabel(final_ylabel, weight="bold")
    ax.set_title(final_title)
    ax.minorticks_on()

    if plot_type == "landscape" and not aspect_ratio:
            if project_path:
                ax.set_aspect('auto')
            else:
                ax.set_aspect('equal')

    if show_legend:
        ax.legend(
            loc="lower left",
            borderaxespad=0.5,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            fontsize=int(active_theme.font_size * 0.8),
        ).set_zorder(101)

    if ax_strip:
        pos_main = ax.get_position()
        pos_strip = ax_strip.get_position()

        # Force strip to match the main plot's Left and Width exactly
        ax_strip.set_position(
            [pos_main.x0, pos_strip.y0, pos_main.width, pos_strip.height]
        )

    plt.tight_layout(pad=0.5)

    if output_file:
        plt.savefig(output_file, transparent=False, bbox_inches="tight", dpi=dpi)
    else:
        plt.show()


if __name__ == "__main__":
    main()
