#!/usr/bin/env python3
# This script follows the guidelines laid out here:
# https://realpython.com/python-script-structure/

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "cmcrameri",
#   "rich",
#   "ase",
# ]
# ///

import glob
import logging
from pathlib import Path
import sys
import io

import click
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import ArrowStyle
import numpy as np
from cmcrameri import cm
from rich.console import Console
from rich.logging import RichHandler
from scipy.interpolate import splrep, splev
from ase.io import read as ase_read
from ase.io import write as ase_write


# --- Constants & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    ],
)
log = logging.getLogger("rich")

DEFAULT_INPUT_PATTERN = "neb_*.dat"
DEFAULT_CMAP = "batlow"


def load_paths(file_pattern: str) -> list[Path]:
    """Finds and sorts files matching a glob pattern."""
    log.info(f"Searching for files with pattern: '{file_pattern}'")
    file_paths = sorted(Path(p) for p in glob.glob(file_pattern))
    if not file_paths:
        log.error(f"No files found matching '{file_pattern}'. Exiting.")
        sys.exit(1)
    log.info(f"Found {len(file_paths)} files to plot.")
    return file_paths


def plot_structure_insets(
    ax, atoms_list, rc_points, y_points, images_to_plot="all", plot_mode="energy"
):
    """
    Renders and plots selected atomic structures as insets on the provided matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the structure insets.
    atoms_list : list
        List of ASE Atoms objects representing the structures to plot.
    rc_points : list or array-like
        List of reaction coordinate values corresponding to each structure.
    y_points : list or array-like
        List of y-axis values (e.g., energy or eigenvalue) for each structure.
    images_to_plot : str, optional
        Determines which structures to plot as insets. Options:
            - "all": plot all structures.
            - "crit_points": plot only the initial, saddle, and final structures.
    plot_mode : str, optional
        Determines how the saddle point is selected. Options:
            - "energy": saddle is the structure with maximum y value.
            - "eigenvalue": saddle is the structure with minimum y value.

    Behavior
    --------
    If the number of structures does not match the number of reaction coordinate points,
    a warning is logged and no structures are plotted.
    The function does not return a value; it modifies the provided axis in-place.

    Usage Notes
    ----------
    - This function is intended to be used as part of a NEB (Nudged Elastic Band) path plotting routine.
    - The appearance and placement of insets may depend on the axis limits and figure size.
    """
    if len(atoms_list) != len(rc_points):
        log.warning(
            f"Mismatch between number of structures ({len(atoms_list)}) and "
            f"data points ({len(rc_points)}). Skipping structure plotting."
        )
        return

    # Determine which indices to plot
    plot_indices = []
    if images_to_plot == "all":
        plot_indices = range(len(atoms_list))
    elif images_to_plot == "crit_points":
        # Find the index of the "saddle" point based on the plot mode
        if plot_mode == "energy":
            saddle_index = np.argmax(y_points)  # Highest point for energy
        else:  # plot_mode == "eigenvalue"
            saddle_index = np.argmin(y_points)  # Lowest point for eigenvalue

        crit_indices = {0, saddle_index, len(atoms_list) - 1}
        plot_indices = sorted(list(crit_indices))

    for i in plot_indices:
        atoms = atoms_list[i]
        buf = io.BytesIO()
        ase_write(
            buf,
            atoms,
            format="png",
            rotation=("-75x, -30y, 0z"),
            show_unit_cell=0,
            scale=35,
        )
        buf.seek(0)
        img_data = plt.imread(buf)
        buf.close()

        imagebox = OffsetImage(img_data, zoom=0.4)
        if images_to_plot == "all":
            y_offset, rad = (60.0, 0.1) if i % 2 == 0 else (-60.0, -0.1)
            xybox = (15.0, y_offset)
            connectionstyle = f"arc3,rad={rad}"
        else:
            xybox = (15.0, 60.0)
            connectionstyle = "arc3,rad=0.1"

        ab = AnnotationBbox(
            imagebox,
            (rc_points[i], y_points[i]),  # Use the correct y-data for positioning
            xybox=xybox,
            frameon=False,
            xycoords="data",
            boxcoords="offset points",
            pad=0.1,
            arrowprops=dict(
                arrowstyle=ArrowStyle.Fancy(
                    head_length=0.4, head_width=0.4, tail_width=0.1
                ),
                connectionstyle=connectionstyle,
                linestyle="--",
                color="black",
                linewidth=0.8,
            ),
        )
        ax.add_artist(ab)
        ab.set_zorder(100)


def plot_energy_path(ax, path_data, color, alpha, zorder):
    """Plots a single interpolated energy path and its data points."""
    rc = path_data[1]
    energy = path_data[2]

    rc_fine = np.linspace(rc.min(), rc.max(), num=300)
    spline_representation = splrep(rc, energy, k=3)
    spline_y = splev(rc_fine, spline_representation)

    ax.plot(rc_fine, spline_y, color=color, alpha=alpha, zorder=zorder)
    ax.plot(
        rc,
        energy,
        linestyle="",
        marker="o",
        markersize=4,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )


def plot_eigenvalue_path(ax, path_data, color, alpha, zorder):
    """
    Plots a single interpolated eigenvalue path and its data points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to plot.
    path_data : list
        List containing path data. The second element (index 1) should be the reaction coordinate array,
        and the fifth element (index 4) should be the eigenvalue array.
    color : str or tuple
        Color specification for the plot line and markers.
    alpha : float
        Transparency level for the plot line and markers (0.0 transparent through 1.0 opaque).
    zorder : int
        Drawing order for the plot elements.

    Returns
    -------
    None
        This function modifies the provided axes object in place and does not return a value.

    Usage
    -----
    Call this function to plot an eigenvalue path on a matplotlib axes object, typically as part of a
    NEB (Nudged Elastic Band) analysis visualization. The function will plot both the interpolated
    spline and the original data points, and draw a horizontal reference line at y=0.
    """
    rc = path_data[1]
    eigenvalue = path_data[4]

    rc_fine = np.linspace(rc.min(), rc.max(), num=300)
    spline_representation = splrep(rc, eigenvalue, k=3)
    spline_y = splev(rc_fine, spline_representation)

    ax.plot(rc_fine, spline_y, color=color, alpha=alpha, zorder=zorder)
    ax.plot(
        rc,
        eigenvalue,
        linestyle="",
        marker="o",
        markersize=4,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.axhline(0, color="white", linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)


def setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor="gray"):
    """Applies labels, limits, and other plot aesthetics."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
    ax.set_facecolor(facecolor)
    ax.set_xlim(left=0)
    plt.tight_layout(pad=0.5)
    plt.grid(False)


@click.command()
@click.option(
    "--input-pattern",
    default=DEFAULT_INPUT_PATTERN,
    help=f"Glob pattern for input data files. Default: '{DEFAULT_INPUT_PATTERN}'",
)
@click.option(
    "--con-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a .con trajectory file to render structures.",
)
@click.option(
    "--plot-structures",
    type=click.Choice(["none", "all", "crit_points"], case_sensitive=False),
    default="none",
    help="Which structures to render on the final path. Requires --con-file.",
)
@click.option(
    "--plot-mode",
    type=click.Choice(["energy", "eigenvalue"], case_sensitive=False),
    default="energy",
    help="The primary quantity to plot on the Y-axis.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file name. If not provided, plot is shown interactively.",
)
@click.option(
    "--start", type=int, default=None, help="Starting file index to plot (inclusive)."
)
@click.option(
    "--end", type=int, default=None, help="Ending file index to plot (exclusive)."
)
@click.option(
    "--normalize-rc",
    is_flag=True,
    default=False,
    help="Normalize the reaction coordinate to a 0-1 scale.",
)
@click.option("--title", default="NEB Path Optimization", help="Plot title.")
@click.option("--xlabel", default=r"Reaction Coordinate ($\AA$)", help="X-axis label.")
@click.option("--ylabel", default="Relative Energy (eV)", help="Y-axis label.")
@click.option("--facecolor", default="gray", help="Background color")
@click.option(
    "--cmap",
    default=DEFAULT_CMAP,
    help=f"Colormap for paths (from cmcrameri). Default: '{DEFAULT_CMAP}'",
)
@click.option(
    "--highlight-last/--no-highlight-last",
    is_flag=True,
    default=True,
    help="Highlight the final path in red.",
)
def main(
    input_pattern: str,
    con_file: Path | None,
    plot_structures: str,
    plot_mode: str,
    output_file: Path | None,
    start: int | None,
    end: int | None,
    normalize_rc: bool,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    highlight_last: bool,
    facecolor: str,
):
    """
    Plots a series of NEB paths from .dat files.
    """
    if plot_structures != "none" and not con_file:
        log.error("--plot-structures requires a --con-file to be provided.")
        sys.exit(1)

    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    atoms_list = None
    if con_file:
        try:
            log.info(f"Reading structures from [cyan]{con_file}[/cyan]")
            atoms_list = ase_read(con_file, index=":")
        except Exception as e:
            log.error(f"Failed to read .con file: {e}")
            atoms_list = None

    all_file_paths = load_paths(input_pattern)
    file_paths_to_plot = all_file_paths[start:end]
    num_files = len(file_paths_to_plot)

    if num_files == 0:
        log.error("The specified start/end range resulted in zero files. Exiting.")
        sys.exit(1)

    colormap = getattr(cm, cmap)
    color_divisor = (num_files - 1) if num_files > 1 else 1.0

    # --- Set plot function and labels based on mode BEFORE the loop ---
    if plot_mode == "energy":
        plot_function = plot_energy_path
        y_data_column = 2  # Energy is the 3rd column
    else:  # plot_mode == "eigenvalue"
        plot_function = plot_eigenvalue_path
        y_data_column = 4  # Eigenvalue is the 5th column
        # Override default ylabel if in eigenvalue mode
        ylabel = r"Lowest Eigenvalue (eV/$\AA^2$)"

    # --- Plotting Loop ---
    for idx, file_path in enumerate(file_paths_to_plot):
        try:
            path_data = np.loadtxt(file_path, skiprows=1).T
            if path_data.shape[0] < y_data_column + 1:
                raise ValueError(f"file requires at least {y_data_column + 1} columns.")
        except (ValueError, IndexError) as e:
            log.warning(
                f"Skipping invalid or empty file [yellow]{file_path.name}[/yellow]: {e}"
            )
            continue

        if normalize_rc:
            rc = path_data[1]
            if rc.max() > 0:
                path_data[1] = rc / rc.max()
            xlabel = "Normalized Reaction Coordinate"

        is_last_file = idx == num_files - 1
        is_first_file = idx == 0

        if highlight_last and is_last_file:
            color, alpha, zorder = "red", 1.0, 20
            plot_function(ax, path_data, color, alpha, zorder)
            # If we have structures, plot them
            if atoms_list and plot_structures != "none":
                plot_structure_insets(
                    ax,
                    atoms_list,
                    path_data[1],
                    path_data[y_data_column],
                    plot_structures,
                    plot_mode,
                )
        else:
            color = colormap(idx / color_divisor)
            alpha = 1.0 if is_first_file else 0.5
            zorder = 10 if is_first_file else 5
            plot_function(ax, path_data, color, alpha, zorder)

    # --- Final Touches ---
    setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor)
    if normalize_rc:
        ax.set_xlim(0, 1)

    sm = plt.cm.ScalarMappable(
        cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max(1, num_files - 1))
    )
    cbar = fig.colorbar(sm, ax=ax, label="Optimization Step")

    if output_file:
        log.info(f"Saving plot to [green]{output_file}[/green]")
        plt.savefig(output_file, transparent=False)
    else:
        log.info("Displaying plot interactively...")
        plt.show()


if __name__ == "__main__":
    main()
