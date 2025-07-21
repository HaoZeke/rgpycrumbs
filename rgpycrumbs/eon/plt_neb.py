#!/usr/bin/env python3

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
from rich.logging import RichHandler
from scipy.interpolate import splrep, splev
from ase.io import read as ase_read
from ase.io import write as ase_write


# --- Constants & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False)],
)

DEFAULT_INPUT_PATTERN = "neb_*.dat"
DEFAULT_CMAP = "batlow"


def load_paths(file_pattern: str) -> list[Path]:
    """Finds and sorts files matching a glob pattern."""
    logging.info(f"Searching for files with pattern: '{file_pattern}'")
    file_paths = sorted(Path(p) for p in glob.glob(file_pattern))
    if not file_paths:
        logging.error(f"No files found matching '{file_pattern}'. Exiting.")
        sys.exit(1)
    logging.info(f"Found {len(file_paths)} files to plot.")
    return file_paths


def plot_structure_insets(ax, atoms_list, path_data, images_to_plot="all"):
    """Renders and plots selected atomic structures as insets on the graph."""
    rc_points = path_data[1]
    energy_points = path_data[2]

    if len(atoms_list) != len(rc_points):
        logging.warning(
            f"Mismatch between number of structures ({len(atoms_list)}) and "
            f"data points ({len(rc_points)}). Skipping structure plotting."
        )
        return

    # Determine which indices to plot
    plot_indices = []
    if images_to_plot == "all":
        plot_indices = range(len(atoms_list))
    elif images_to_plot == "crit_points":
        # Find the index of the highest energy point (saddle)
        saddle_index = np.argmax(energy_points)
        # Use a set to automatically handle cases where the saddle is an endpoint
        crit_indices = {0, saddle_index, len(atoms_list) - 1}
        plot_indices = sorted(list(crit_indices))

    for i in plot_indices:
        atoms = atoms_list[i]
        # Render atoms object to an in-memory PNG with better aesthetics
        buf = io.BytesIO()
        ase_write(
            buf,
            atoms,
            format="png",
            rotation=("-75x, -30y, 0z"),
            show_unit_cell=0,  # Do not render the simulation cell box
        )
        buf.seek(0)
        img_data = plt.imread(buf)
        buf.close()

        # Place the image on the plot
        # Increased zoom for larger images
        imagebox = OffsetImage(img_data, zoom=0.4)
        if images_to_plot == "all":
            if i % 2 == 0:
                y_offset = 60.0  # Even images go up
                rad = 0.1
            else:
                y_offset = -60.0 # Odd images go down
                rad = -0.1
            xybox = (15.0, y_offset)
            connectionstyle = f"arc3,rad={rad}"
        else: # For 'crit_points', a single offset is fine
            xybox = (15.0, 60.0)
            connectionstyle = "arc3,rad=0.1"
        # Create the annotation box for the image
        ab = AnnotationBbox(
            imagebox,
            (rc_points[i], energy_points[i]),
            # Offset farther up and slightly to the side
            xybox=xybox,
            frameon=False,
            xycoords="data",
            boxcoords="offset points",
            pad=0.1,
            arrowprops=dict(
                arrowstyle=ArrowStyle.Fancy(
                    head_length=0.4, head_width=0.4, tail_width=0.1
                ),  # No arrowhead is -
                connectionstyle=connectionstyle,
                linestyle="--",
                color="black",
                linewidth=0.8,
            ),
        )
        # Add the artist and set a high zorder to ensure it's on top
        ax.add_artist(ab)
        ab.set_zorder(100)


def plot_single_path(ax, path_data, color, alpha, zorder):
    """Plots a single interpolated energy path and its data points."""
    energy = path_data[2]
    rc = path_data[1]

    # Cubic spline interpolation for a smooth curve
    rc_fine = np.linspace(rc.min(), rc.max(), num=300)
    spline_representation = splrep(rc, energy, k=3)
    spline_y = splev(rc_fine, spline_representation)

    # Plot the smooth curve and the original data points
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


def setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor="gray"):
    """Applies labels, limits, and other plot aesthetics."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
    ax.set_facecolor(facecolor)
    ax.set_xlim(left=0)
    plt.tight_layout(pad=0.5)


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
    Plots a series of NEB energy paths from .dat files.
    """
    if plot_structures != "none" and not con_file:
        logging.error("--plot-structures requires a --con-file to be provided.")
        sys.exit(1)

    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    # Load atom structures if a con file is provided
    atoms_list = None
    if con_file:
        try:
            logging.info(f"Reading structures from [cyan]{con_file}[/cyan]")
            atoms_list = ase_read(con_file, index=":")
        except Exception as e:
            logging.error(f"Failed to read .con file: {e}")
            atoms_list = None

    all_file_paths = load_paths(input_pattern)
    original_num_files = len(all_file_paths)
    # Apply slicing for sub-range plotting
    file_paths_to_plot = all_file_paths[start:end]
    if len(file_paths_to_plot) < original_num_files:
        logging.info(
            f"Plotting sub-range: {len(file_paths_to_plot)} of {original_num_files} total files."
        )

    num_files = len(file_paths_to_plot)

    if num_files == 0:
        logging.error("The specified start/end range resulted in zero files. Exiting.")
        sys.exit(1)

    colormap = getattr(cm, cmap)
    color_divisor = (num_files - 1) if num_files > 1 else 1.0

    # --- Plotting Loop ---
    for idx, file_path in enumerate(file_paths_to_plot):
        try:
            path_data = np.loadtxt(file_path).T
            if path_data.size == 0:
                raise ValueError("contains no data")
        except (ValueError, IndexError) as e:
            logging.warning(
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
            # If we have structures, plot them
            if atoms_list and plot_structures != "none":
                plot_structure_insets(ax, atoms_list, path_data, plot_structures)
        else:
            color = colormap(idx / color_divisor)
            alpha = 1.0 if is_first_file else 0.5
            zorder = 10 if is_first_file else 5

        plot_single_path(ax, path_data, color, alpha, zorder)

    # --- Final Touches ---
    setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor)
    if normalize_rc:
        ax.set_xlim(0, 1)

    sm = plt.cm.ScalarMappable(
        cmap=colormap, norm=plt.Normalize(vmin=0, vmax=num_files - 1)
    )
    cbar = fig.colorbar(sm, ax=ax, label="Optimization Step")

    if output_file:
        logging.info(f"Saving plot to [green]{output_file}[/green]")
        plt.savefig(output_file, transparent=False)
    else:
        logging.info("Displaying plot interactively...")
        plt.show()


if __name__ == "__main__":
    main()
