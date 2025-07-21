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
# ]
# ///

import glob
import logging
from pathlib import Path
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from rich.logging import RichHandler
from scipy.interpolate import splrep, splev

# --- Constants & Setup ---
# Set up logging for clear feedback
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

    # Ensure axes start at 0 but extend to fit data
    ax.set_xlim(left=0)

    # Use a tight layout to minimize whitespace
    plt.tight_layout(pad=0.5)


@click.command()
@click.option(
    "--input-pattern",
    default=DEFAULT_INPUT_PATTERN,
    help=f"Glob pattern for input data files. Default: '{DEFAULT_INPUT_PATTERN}'",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file name. If not provided, plot is shown interactively.",
)
@click.option(
    "--start",
    type=int,
    default=None,
    help="Starting file index to plot (inclusive).",
)
@click.option(
    "--end",
    type=int,
    default=None,
    help="Ending file index to plot (exclusive).",
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

    This script reads all files matching the INPUT_PATTERN, plots each as a
    line on a single graph, and colors them according to their sequence.
    The final plot is saved to the specified OUTPUT_FILE.
    """
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

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
    # Prevent division by zero if only one file is plotted
    color_divisor = (num_files - 1) if num_files > 1 else 1.0

    # --- Plotting Loop ---
    for idx, file_path in enumerate(file_paths_to_plot):
        try:
            path_data = np.loadtxt(file_path).T
            # Explicitly check if the loaded array is empty, which causes the IndexError
            if path_data.size == 0:
                raise ValueError("contains no data")
        except (ValueError, IndexError) as e:
            logging.warning(
                f"Skipping invalid or empty file [yellow]{file_path.name}[/yellow]: {e}"
            )
            continue

        if normalize_rc:
            rc = path_data[1]
            rc_max = rc.max()
            if rc_max > 0:
                path_data[1] = rc / rc_max
            # Update the x-axis label for the final plot
            xlabel = "Normalized Reaction Coordinate"

        is_last_file = idx == num_files - 1
        is_first_file = idx == 0

        # Determine plot properties based on position in sequence
        if highlight_last and is_last_file:
            color, alpha, zorder = "red", 1.0, 20
        else:
            # Normalize index to get a color from the colormap
            color = colormap(idx / color_divisor)
            alpha = 1.0 if is_first_file else 0.5
            zorder = 10 if is_first_file else 5

        plot_single_path(ax, path_data, color, alpha, zorder)

    # --- Final Touches ---
    setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor)
    if normalize_rc:
        ax.set_xlim(0, 1) # Set x-axis limits for normalized plot

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
