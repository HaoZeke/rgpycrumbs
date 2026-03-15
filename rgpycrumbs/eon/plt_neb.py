#!/usr/bin/env python3
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
#   "ase",
#   "polars",
#   "h5py",
#   "chemparseplot",
#   "rgpycrumbs",
# ]
# ///
"""EON CLI - NEB plotting from .dat/.con files or HDF5 trajectories.

Thin CLI wrapper that delegates to chemparseplot for parsing and plotting.
All heavy lifting done by chemparseplot modules.

The default ``grad_imq`` surface method uses gradient-enhanced Inverse Multiquadric
interpolation on 2D RMSD projections. This approach is described in:

    R. Goswami, "Two-dimensional RMSD projections for reaction path visualization
    and validation," MethodsX, p. 103851, Mar. 2026,
    doi: 10.1016/j.mex.2026.103851.

.. versionadded:: 1.7.0
    Refactored from monolithic plt_neb.py to thin CLI wrapper.

"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

from rgpycrumbs._aux import ensure_import

# Lazy imports for optional dependencies
_adjust_text = None
_mpl = None
_plt = None
_pl = None


def _get_adjust_text():
    """Get adjustText module."""
    global _adjust_text
    if _adjust_text is None:
        _adjust_text = ensure_import("adjustText")
    return _adjust_text


def _get_mpl():
    """Get matplotlib modules."""
    global _mpl, _plt
    if _mpl is None:
        _mpl = ensure_import("matplotlib")
        _plt = ensure_import("matplotlib.pyplot")
    return _mpl, _plt


def _get_polars():
    """Get polars module."""
    global _pl
    if _pl is None:
        _pl = ensure_import("polars")
    return _pl


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_INPUT_PATTERN = "neb_*.dat"
DEFAULT_PATH_PATTERN = "neb_path_*.con"
IRA_KMAX_DEFAULT = 1.8


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
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to .con trajectory file.",
)
@click.option(
    "--source",
    type=click.Choice(["eon", "traj", "hdf5"]),
    default="eon",
    help="Data source.",
)
@click.option(
    "--input-h5",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to ChemGP NEB HDF5 file.",
)
@click.option(
    "--input-traj",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to extxyz trajectory file.",
)
@click.option(
    "--plot-type",
    type=click.Choice(["profile", "landscape"]),
    default="profile",
    help="Type of plot to generate.",
)
@click.option(
    "--landscape-mode",
    type=click.Choice(["path", "surface"]),
    default="surface",
    help="For landscape plot.",
)
@click.option(
    "--project-path/--no-project-path",
    is_flag=True,
    default=True,
    help="Project landscape coordinates.",
)
@click.option(
    "--plot-structures",
    type=click.Choice(["none", "all", "crit_points"]),
    default="none",
    help="Structures to render.",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output image filename.",
)
@click.option(
    "--figsize",
    nargs=2,
    type=(float, float),
    default=(5.37, 5.37),
    help="Figure width, height in inches.",
)
@click.option("--dpi", type=int, default=200, help="Resolution in DPI.")
@click.option("--title", default="NEB Path", help="Plot title.")
@click.option("--ira-kmax", default=IRA_KMAX_DEFAULT, help="kmax factor for IRA.")
@click.option(
    "--cache-file",
    type=click.Path(path_type=Path),
    default=Path(".neb_landscape.parquet"),
    help="Parquet cache file.",
)
@click.option(
    "--force-recompute", is_flag=True, default=False, help="Force re-calculation of RMSD."
)
@click.option(
    "--show-pts/--no-show-pts",
    is_flag=True,
    default=True,
    help="Show data points on surface.",
)
@click.option("--show-legend", is_flag=True, default=False, help="Show colorbar legend.")
@click.option(
    "--surface-type",
    default="grad_imq",
    help="Surface interpolation method (grad_imq, grad_matern, grad_imq_ny, rbf).",
)
@click.option(
    "--landscape-path",
    type=click.Choice(["final", "all", "none"]),
    default="final",
    help="Which path steps to overlay on landscape.",
)
def main(
    input_dat_pattern,
    input_path_pattern,
    con_file,
    source,
    input_h5,
    input_traj,
    plot_type,
    landscape_mode,
    project_path,
    plot_structures,
    output_file,
    figsize,
    dpi,
    title,
    ira_kmax,
    cache_file,
    force_recompute,
    surface_type,
    show_pts,
    show_legend,
    landscape_path,
):
    """NEB reaction path and landscape plotting.

    Delegates to chemparseplot for parsing and plotting.
    """
    # Import chemparseplot modules (will be available via PEP 723)
    from chemparseplot.parse.eon.neb import (
        aggregate_neb_landscape_data,
        load_structures_and_calculate_additional_rmsd,
    )
    from chemparseplot.parse.file_ import find_file_paths
    from chemparseplot.parse.trajectory.hdf5 import (
        history_to_landscape_df as hdf5_history_to_landscape_df,
    )
    from chemparseplot.parse.trajectory.hdf5 import (
        result_to_atoms_list,
    )
    from chemparseplot.parse.trajectory.neb import (
        load_trajectory,
        trajectory_to_landscape_df,
    )
    from chemparseplot.plot.neb import (
        plot_landscape_path_overlay,
        plot_landscape_surface,
    )
    from chemparseplot.plot.theme import get_theme, setup_global_theme

    pl = _get_polars()
    mpl, plt = _get_mpl()
    adjust_text = _get_adjust_text()
    import matplotlib.patheffects as path_effects
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import ArrowStyle

    # Setup theme
    active_theme = get_theme("ruhi")
    setup_global_theme(active_theme)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    has_strip = plot_structures in ["all", "crit_points"] and plot_type == "landscape"

    if has_strip:
        gs = GridSpec(2, 1, height_ratios=[1, 0.25], hspace=0.3, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(111)
        ax_strip = None

    # Load structures if needed
    atoms_list = None
    additional_atoms_data = []
    sp_data = None

    if con_file:
        try:
            atoms_list, additional_atoms_data, sp_data = (
                load_structures_and_calculate_additional_rmsd(
                    con_file, [], ira_kmax, None
                )
            )
        except Exception as e:
            log.error(f"Error loading structures: {e}")
            if plot_type == "landscape":
                log.critical("Cannot proceed without structures. Exiting.")
                sys.exit(1)

    # Load data based on source
    if plot_type == "landscape":
        if source == "traj":
            if not input_traj:
                log.critical("--input-traj is required when --source traj is used.")
                sys.exit(1)
            traj_atoms_list = load_trajectory(str(input_traj))
            df = trajectory_to_landscape_df(traj_atoms_list, ira_kmax=ira_kmax)
            if atoms_list is None:
                atoms_list = traj_atoms_list

        elif source == "hdf5":
            if not input_h5:
                log.critical("--input-h5 is required when --source hdf5 is used.")
                sys.exit(1)
            h5_str = str(input_h5)
            try:
                df = hdf5_history_to_landscape_df(h5_str, ira_kmax=ira_kmax)
            except Exception:
                log.warning("History read failed, falling back to single-step result.")
                hdf5_atoms = result_to_atoms_list(h5_str)
                df = trajectory_to_landscape_df(hdf5_atoms, ira_kmax=ira_kmax)
            if atoms_list is None:
                atoms_list = result_to_atoms_list(h5_str)

        else:  # eon source
            dat_paths = find_file_paths(input_dat_pattern)
            con_paths = find_file_paths(str(input_path_pattern))

            if not dat_paths:
                log.critical(f"No data files found for pattern: {input_dat_pattern}")
                sys.exit(1)

            y_col = 2  # energy

            df = aggregate_neb_landscape_data(
                dat_paths,
                con_paths,
                y_col,
                None,
                cache_file=cache_file,
                force_recompute=force_recompute,
                ira_kmax=ira_kmax,
                augment_dat=None,
                augment_con=None,
                ref_atoms=atoms_list[0] if atoms_list else None,
                prod_atoms=atoms_list[-1] if atoms_list else None,
            )

        # Surface generation
        if landscape_mode == "surface":
            df_surface = df.filter(pl.col("step") == df["step"].max())

            r_all = df_surface["r"].to_numpy()
            p_all = df_surface["p"].to_numpy()
            z_all = df_surface["z"].to_numpy()
            gr_all = df_surface["grad_r"].to_numpy()
            gp_all = df_surface["grad_p"].to_numpy()

            # Simple RBF smoothing heuristic
            # Simple smoothing heuristic for RBF
            rbf_smoothing = float(len(r_all)) * 0.01 if surface_type == "rbf" else None

            plot_landscape_surface(
                ax,
                r_all,
                p_all,
                gr_all,
                gp_all,
                z_all,
                method=surface_type,
                rbf_smooth=rbf_smoothing,
                cmap=active_theme.cmap_landscape,
                show_pts=show_pts,
                project_path=project_path,
            )

        # Path overlay
        if landscape_path != "none":
            if landscape_path == "all":
                # Overlay all optimization steps (faint for earlier steps)
                steps = sorted(df["step"].unique())
                for step_idx, step in enumerate(steps):
                    df_step = df.filter(pl.col("step") == step)
                    step_r = df_step["r"].to_numpy()
                    step_p = df_step["p"].to_numpy()
                    step_z = df_step["z"].to_numpy()
                    alpha = 0.3 if step < max(steps) else 1.0  # Highlight final
                    plot_landscape_path_overlay(
                        ax,
                        step_r,
                        step_p,
                        step_z,
                        active_theme.cmap_landscape,
                        "Relative Energy (eV)",
                        project_path=project_path,
                        alpha=alpha,
                    )
            else:  # final
                df_final = df.filter(pl.col("step") == df["step"].max())
                final_r = df_final["r"].to_numpy()
                final_p = df_final["p"].to_numpy()
                final_z = df_final["z"].to_numpy()
                plot_landscape_path_overlay(
                    ax,
                    final_r,
                    final_p,
                    final_z,
                    active_theme.cmap_landscape,
                    "Relative Energy (eV)",
                    project_path=project_path,
                )
        # Colorbar legend
        if show_legend:
            from matplotlib.cm import ScalarMappable

            norm = mpl.colors.Normalize(
                vmin=df_final["z"].min(), vmax=df_final["z"].max()
            )
            sm = ScalarMappable(cmap=active_theme.cmap_landscape, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label="Relative Energy (eV)")
            cbar.solids.set_alpha(0.8)

        # Structure strip
        if has_strip and ax_strip is not None and atoms_list:
            from chemparseplot.plot.neb import plot_structure_strip

            # Select structures based on mode
            if plot_structures == "crit_points":
                # Reactant, saddle (if available), product
                strip_atoms = [atoms_list[0], atoms_list[-1]]
                strip_labels = ["Reactant", "Product"]
                if sp_data:
                    strip_atoms.insert(1, sp_data["atoms"])
                    strip_labels.insert(1, "Saddle")
            else:
                # All structures (subsample if too many)
                max_structures = 12
                if len(atoms_list) > max_structures:
                    indices = np.linspace(
                        0, len(atoms_list) - 1, max_structures, dtype=int
                    )
                    strip_atoms = [atoms_list[i] for i in indices]
                    strip_labels = [f"{i}" for i in indices]
                else:
                    strip_atoms = atoms_list
                    strip_labels = [str(i) for i in range(len(atoms_list))]

            plot_structure_strip(
                ax_strip,
                strip_atoms,
                strip_labels,
                zoom=0.3,
                rotation="0x,90y,0z",
                theme_color="black",
                max_cols=6,
            )

        # Save
        # Save
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
            log.info(f"Saved: {output_file}")
        plt.close(fig)
    else:
        log.info(
            "Profile plot not yet implemented in thin wrapper - use chemparseplot directly"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
