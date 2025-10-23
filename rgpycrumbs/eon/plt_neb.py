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
#   "cmcrameri",
#   "rich",
#   "ase",
# ]
# ///

import glob
import io
import logging
import sys
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write
from cmcrameri import cm
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle
from rich.logging import RichHandler
from scipy.interpolate import CubicHermiteSpline, griddata, splev, splrep
from scipy.signal import savgol_filter

try:
    import ira_mod  # type: ignore
except ImportError:
    ira_mod = None  # IRA is optional. Handle gracefully.

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)
log = logging.getLogger("rich")


# --- Enumerations and Constants ---
class PlotType(Enum):
    """Defines the overall plot type."""

    PROFILE = "profile"
    LANDSCAPE = "landscape"


class RCMode(Enum):
    """Defines the reaction coordinate for profile plots."""

    PATH = "path"  # Default path distance
    RMSD = "rmsd"  # RMSD from reactant


class PlotMode(Enum):
    """Defines the primary data to be plotted (Y-axis or color)."""

    ENERGY = "energy"
    EIGENVALUE = "eigenvalue"


class SplineMethod(Enum):
    """Defines the interpolation method for profile plots."""

    HERMITE = "hermite"  # Cubic Hermite spline (uses derivatives/forces)
    SPLINE = "spline"  # Standard cubic spline (no derivatives)


DEFAULT_INPUT_PATTERN = "neb_*.dat"
DEFAULT_CMAP = "batlow"


@dataclass
class SmoothingParams:
    """Parameters for Savitzky-Golay smoothing of force data."""

    window_length: int = 5
    polyorder: int = 2


# Datastructure for inset positioning
InsetImagePos = namedtuple("InsetImagePos", "x y rad")


# --- Utility Functions ---
def load_paths(file_pattern: str) -> list[Path]:
    """Finds and sorts files matching a glob pattern."""
    log.info(f"Searching for files with pattern: '{file_pattern}'")
    file_paths = sorted(Path(p) for p in glob.glob(file_pattern))
    if not file_paths:
        log.error(f"No files found matching '{file_pattern}'. Exiting.")
        sys.exit(1)
    log.info(f"Found {len(file_paths)} file(s).")
    return file_paths


def calculate_rmsd_from_ref(
    atoms_list: list, ira_instance, ref_atom: "ase.Atoms"
) -> np.ndarray:
    """
    Calculates the RMSD of each structure in a list relative to a reference.

    Uses the Iterative Reordering and Alignment (IRA) algorithm to find the
    optimal alignment and permutation before calculating RMSD.

    Parameters
    ----------
    atoms_list : list
        A list of ASE Atoms objects.
    ira_instance : ira_mod.IRA
        An instantiated IRA object.
    ref_atom : ase.Atoms
        The reference Atoms object to align against.

    Returns
    -------
    np.ndarray
        An array of RMSD values, one for each structure in `atoms_list`.
    """
    nat_ref = len(ref_atom)
    typ_ref = ref_atom.get_atomic_numbers()
    coords_ref = ref_atom.get_positions()
    kmax_factor = 1.8  # IRA parameter
    rmsd_values = np.zeros(len(atoms_list))

    for i, atom_i in enumerate(atoms_list):
        nat_i = len(atom_i)
        typ_i = atom_i.get_atomic_numbers()
        coords_i = atom_i.get_positions()

        if atom_i is ref_atom:
            rmsd_values[i] = 0.0
            continue

        # Perform IRA match
        r, t, p, hd = ira_instance.match(
            nat_ref, typ_ref, coords_ref, nat_i, typ_i, coords_i, kmax_factor
        )

        # Apply alignment and permutation
        coords_i_aligned = (coords_i @ r.T) + t
        coords_i_aligned_permuted = coords_i_aligned[p]

        # Calculate RMSD
        diff_sq = (coords_ref - coords_i_aligned_permuted) ** 2
        rmsd = np.sqrt(np.mean(np.sum(diff_sq, axis=1)))
        rmsd_values[i] = rmsd

    return rmsd_values


def calculate_landscape_coords(atoms_list: list, ira_instance):
    """
    Calculates 2D landscape coordinates (RMSD-R, RMSD-P) for a path.

    Parameters
    ----------
    atoms_list : list
        List of ASE Atoms objects representing the path.
    ira_instance : ira_mod.IRA
        An instantiated IRA object.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - rmsd_r: Array of RMSD values relative to the reactant (first image).
        - rmsd_p: Array of RMSD values relative to the product (last image).
    """
    log.info(
        "Calculating landscape coordinates using [bold magenta]ira.match[/bold magenta]..."
    )
    rmsd_r = calculate_rmsd_from_ref(atoms_list, ira_instance, ref_atom=atoms_list[0])
    rmsd_p = calculate_rmsd_from_ref(atoms_list, ira_instance, ref_atom=atoms_list[-1])
    log.info("Landscape coordinate calculation complete.")
    return rmsd_r, rmsd_p


# --- Plotting Functions ---
def plot_single_inset(
    ax, atoms, x_coord, y_coord, xybox=(15.0, 60.0), rad=0.0
):  # <-- Added rad
    """
    Renders a single ASE Atoms object and plots it as an inset.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    atoms : ase.Atoms
        The atomic structure to render.
    x_coord : float
        The x-data coordinate to anchor the arrow to.
    y_coord : float
        The y-data coordinate to anchor the arrow to.
    xybox : tuple, optional
        The (x, y) offset in points for placing the image box.
    rad : float, optional
        The connection style 'rad' parameter for the arrow.
    """
    buf = io.BytesIO()
    ase_write(
        buf, atoms, format="png", rotation=("0x, 90y, 0z"), show_unit_cell=0, scale=35
    )
    buf.seek(0)
    img_data = plt.imread(buf)
    buf.close()

    imagebox = OffsetImage(img_data, zoom=0.4)
    ab = AnnotationBbox(
        imagebox,
        (x_coord, y_coord),
        xybox=xybox,
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0.1,
        arrowprops={
            "arrowstyle": ArrowStyle.Fancy(
                head_length=0.4, head_width=0.4, tail_width=0.1
            ),
            "connectionstyle": f"arc3,rad={rad}",
            "linestyle": "-",
            "color": "black",
            "linewidth": 1.5,
        },
    )
    ax.add_artist(ab)
    ab.set_zorder(100)  # Ensure inset is drawn on top


def plot_structure_insets(
    ax,
    atoms_list,
    x_coords,
    y_coords,
    saddle_data,
    images_to_plot,
    plot_mode,
    draw_reactant: InsetImagePos | None = None,
    draw_saddle: InsetImagePos | None = None,
    draw_product: InsetImagePos | None = None,
):
    """
    Plots insets for critical points (reactant, saddle, product) or all images.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    atoms_list : list
        List of all ASE Atoms objects for the path.
    x_coords : np.ndarray
        Array of x-coordinates (RC or RMSD-R) for each image.
    y_coords : np.ndarray
        Array of y-coordinates (Energy, Eigenvalue, or RMSD-P) for each image.
    saddle_data : np.ndarray
        Data used to find the saddle point. For energy mode, this is the
        energy array. For eigenvalue mode, this is the eigenvalue array.
    images_to_plot : str
        Which images to plot: "all" or "crit_points".
    plot_mode : str
        "energy" or "eigenvalue", used to determine saddle point logic.
    draw_reactant : InsetImagePos
        Positioning info for the reactant inset.
    draw_saddle : InsetImagePos
        Positioning info for the saddle inset.
    draw_product : InsetImagePos
        Positioning info for the product inset.
    """
    if draw_reactant is None:
        draw_reactant = InsetImagePos(15, 60, 0.1)
    if draw_saddle is None:
        draw_saddle = InsetImagePos(15, 60, 0.1)
    if draw_product is None:
        draw_product = InsetImagePos(15, 60, 0.1)
    if len(atoms_list) != len(x_coords) or len(atoms_list) != len(y_coords):
        log.warning(
            f"Mismatch between number of structures ({len(atoms_list)}) and data points ({len(x_coords)}). Skipping structure plotting."
        )
        return

    plot_indices = []
    saddle_index = -1  # Initialize
    if images_to_plot == "all":
        plot_indices = range(len(atoms_list))
    elif images_to_plot == "crit_points":
        if plot_mode == "energy":
            # Saddle is max energy, *excluding* endpoints
            saddle_index = np.argmax(saddle_data[1:-1]) + 1
        else:  # plot_mode == "eigenvalue"
            # Saddle is min eigenvalue (can be any point)
            saddle_index = np.argmin(saddle_data)

        crit_indices = {0, saddle_index, len(atoms_list) - 1}
        plot_indices = sorted(crit_indices)

    # Plot the selected structures
    for i in plot_indices:
        if images_to_plot == "all":
            y_offset = 60.0 if i % 2 == 0 else -60.0
            xybox = (15.0, y_offset)
            rad = 0.1 if i % 2 == 0 else -0.1
        elif i == 0:
            xybox = (draw_reactant.x, draw_reactant.y)
            rad = draw_reactant.rad
        elif i == saddle_index:
            xybox = (draw_saddle.x, draw_saddle.y)
            rad = draw_saddle.rad
        else:  # Product
            xybox = (draw_product.x, draw_product.y)
            rad = draw_product.rad

        plot_single_inset(
            ax, atoms_list[i], x_coords[i], y_coords[i], xybox=xybox, rad=rad
        )


def plot_energy_path(
    ax, path_data, color, alpha, zorder, method="hermite", smoothing=SmoothingParams()
):
    """
    Plots a single interpolated energy path and its data points.

    Supports two interpolation methods:
    - 'hermite': Cubic Hermite spline. Uses energy values and their
      derivatives (taken from the parallel force `f_para`). This is
      often a more physically accurate interpolation for NEB paths.
    - 'spline': Standard cubic spline. Ignores derivative information.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    path_data : np.ndarray
        2D array of data (from neb_*.dat), transposed.
        Expected: path_data[1] = rc, path_data[2] = energy, path_data[3] = f_para
    color : str or tuple
        Color for the plot.
    alpha : float
        Transparency for the plot.
    zorder : int
        Plotting layer order.
    method : str, optional
        Interpolation method: "hermite" or "spline".
    smoothing : SmoothingParams, optional
        Parameters for Savitzky-Golay filter if using Hermite spline.
    """
    rc = path_data[1]
    energy = path_data[2]
    f_para = path_data[3]  # Parallel force
    deriv = -f_para  # Derivative dE/d(rc) = -F_parallel

    try:
        # Sort data by reaction coordinate for correct interpolation
        sort_indices = np.argsort(rc)
        rc_sorted = rc[sort_indices]
        energy_sorted = energy[sort_indices]

        # Normalize RC to [0, 1] for stable spline fitting
        rc_min, rc_max = rc_sorted.min(), rc_sorted.max()
        rc_norm_sorted = (rc_sorted - rc_min) / (rc_max - rc_min)
        rc_fine_norm = np.linspace(0, 1, num=300)

        if method == "hermite":
            # Smooth the derivatives to reduce noise
            deriv_smooth = savgol_filter(
                deriv,
                window_length=smoothing.window_length,
                polyorder=smoothing.polyorder,
            )
            deriv_smooth_sorted = deriv_smooth[sort_indices]

            # Use Hermite spline which respects both values and derivatives
            hermite_spline = CubicHermiteSpline(
                rc_norm_sorted, energy_sorted, deriv_smooth_sorted
            )
            spline_y = hermite_spline(rc_fine_norm)
        else:
            # Use standard cubic spline
            spline_representation = splrep(rc_norm_sorted, energy_sorted, k=3)
            spline_y = splev(rc_fine_norm, spline_representation)

        # Rescale fine RC back to original units for plotting
        rc_plot_fine = rc_fine_norm * (rc_max - rc_min) + rc_min

    except Exception as e:
        log.warning(f"Interpolation failed ({e}), falling back to standard spline.")
        spline_representation = splrep(rc, energy, k=3)
        rc_fine = np.linspace(rc.min(), rc.max(), num=300)
        spline_y = splev(rc_fine, spline_representation)
        rc_plot_fine = rc_fine

    # Plot the interpolated line
    ax.plot(rc_plot_fine, spline_y, color=color, alpha=alpha, zorder=zorder)
    # Plot the original data points
    ax.plot(
        rc,
        energy,
        marker="o",
        linestyle="None",
        color=color,
        markersize=6,
        alpha=alpha,
        zorder=zorder + 1,
        markerfacecolor=color,
        markeredgewidth=0.5,
    )


def plot_eigenvalue_path(ax, path_data, color, alpha, zorder):
    """
    Plots a single interpolated eigenvalue path and its data points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to plot.
    path_data : np.ndarray
        2D array of data (from neb_*.dat), transposed.
        Expected: path_data[1] = rc, path_data[4] = eigenvalue
    color : str or tuple
        Color specification for the plot line and markers.
    alpha : float
        Transparency level for the plot line and markers.
    zorder : int
        Drawing order for the plot elements.
    """
    rc = path_data[1]
    eigenvalue = path_data[4]

    try:
        # Sort data by reaction coordinate
        sort_indices = np.argsort(rc)
        rc_sorted = rc[sort_indices]
        eigenvalue_sorted = eigenvalue[sort_indices]
    except ValueError:
        log.warning("Could not sort eigenvalue data, plotting as is.")
        rc_sorted = rc
        eigenvalue_sorted = eigenvalue

    # Interpolate using a standard cubic spline
    rc_fine = np.linspace(rc.min(), rc.max(), num=300)
    spline_representation = splrep(rc_sorted, eigenvalue_sorted, k=3)
    spline_y = splev(rc_fine, spline_representation)

    # Plot the interpolated line
    ax.plot(rc_fine, spline_y, color=color, alpha=alpha, zorder=zorder)
    # Plot the original data points
    ax.plot(
        rc,
        eigenvalue,
        marker="o",
        linestyle="None",
        color=color,
        markersize=6,
        alpha=alpha,
        zorder=zorder + 1,
        markerfacecolor=color,
        markeredgewidth=0.5,
    )
    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color="white", linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)


def plot_landscape_path(ax, rmsd_r, rmsd_p, z_data, cmap, z_label):
    """
    Plots the 1D path on the 2D RMSD landscape, colored by z_data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    rmsd_r : np.ndarray
        RMSD from reactant (x-axis).
    rmsd_p : np.ndarray
        RMSD from product (y-axis).
    z_data : np.ndarray
        Data for coloring the path (e.g., energy or eigenvalue).
    cmap : str
        Name of the colormap to use.
    z_label : str
        Label for the colorbar.
    """
    fig = ax.get_figure()
    norm = plt.Normalize(z_data.min(), z_data.max())
    colormap = getattr(cm, cmap)

    # Create a LineCollection to color the path segments
    points = np.array([rmsd_r, rmsd_p]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=colormap, norm=norm, zorder=30)
    segment_z = (z_data[:-1] + z_data[1:]) / 2  # Color by segment midpoint
    lc.set_array(segment_z)
    lc.set_linewidth(3)
    ax.add_collection(lc)

    # Plot points on top
    ax.scatter(
        rmsd_r,
        rmsd_p,
        c=z_data,
        cmap=colormap,
        norm=norm,
        edgecolors="black",
        linewidths=0.5,
        zorder=40,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, ax=ax, label=z_label)


def plot_interpolated_landscape(ax, rmsd_r, rmsd_p, z_data, cmap):
    """
    Generates and plots an interpolated 2D surface (contour plot).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    rmsd_r : np.ndarray
        RMSD from reactant (x-axis).
    rmsd_p : np.ndarray
        RMSD from product (y-axis).
    z_data : np.ndarray
        Data for coloring the path (z-axis).
    cmap : str
        Name of the colormap to use.
    """
    log.info("Generating interpolated 2D surface...")
    xi = np.linspace(rmsd_r.min(), rmsd_r.max(), 100)
    yi = np.linspace(rmsd_p.min(), rmsd_p.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate Z data onto the grid
    Z = griddata((rmsd_r, rmsd_p), z_data, (X, Y), method="cubic")

    colormap = getattr(cm, cmap)
    ax.contourf(X, Y, Z, levels=20, cmap=colormap, alpha=0.75, zorder=10)


def setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor="gray"):
    """Applies labels, limits, and other plot aesthetics."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
    ax.set_facecolor(facecolor)
    # Only set xlim(left=0) if not an RMSD-R plot (which starts at 0 anyway)
    if xlabel != r"RMSD from Reactant ($\AA$)":
        ax.set_xlim(left=0)
    plt.grid(False)
    plt.tight_layout(pad=0.5)


# --- CLI ---
@click.command()
@click.option(
    "--input-pattern",
    default=DEFAULT_INPUT_PATTERN,
    help="Glob pattern for input data files.",
)
@click.option(
    "--con-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to .con trajectory file.",
)
@click.option(
    "--additional-con",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to additional .con file to highlight (requires IRA).",
)
@click.option(
    "--plot-type",
    type=click.Choice([e.value for e in PlotType]),
    default=PlotType.PROFILE.value,
    help="Type of plot to generate: 'profile' (1D path) or 'landscape' (2D RMSD plot).",
)
@click.option(
    "--landscape-mode",
    type=click.Choice(["path", "surface"]),
    default="path",
    help="For landscape plot: 'path' (only 1D path) or 'surface' (interpolated 2D surface).",
)
@click.option(
    "--rc-mode",
    type=click.Choice([e.value for e in RCMode]),
    default=RCMode.PATH.value,
    help="Reaction coordinate for profile plot: 'path' (RC) or 'rmsd' (from reactant).",
)
@click.option(
    "--plot-structures",
    type=click.Choice(["none", "all", "crit_points"]),
    default="none",
    help="Structures to render on the path. Requires --con-file.",
)
@click.option(
    "--plot-mode",
    type=click.Choice([e.value for e in PlotMode]),
    default=PlotMode.ENERGY.value,
    help="Quantity to plot on y-axis (profile) or color (landscape)",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output image filename. If not provided, plot is shown interactively.",
)
@click.option(
    "--start", type=int, default=None, help="Start file index for profile plot."
)
@click.option("--end", type=int, default=None, help="End file index for profile plot.")
@click.option(
    "--normalize-rc", is_flag=True, default=False, help="Normalize reaction coordinate."
)
@click.option("--title", default="NEB Path", help="Plot title.")
@click.option("--xlabel", default=None, help="X-axis label (overrides default).")
@click.option("--ylabel", default=None, help="Y-axis label (overrides default).")
@click.option("--facecolor", default="gray", help="Background color.")
@click.option("--cmap", default=DEFAULT_CMAP, help="Colormap for paths.")
@click.option(
    "--highlight-last/--no-highlight-last",
    is_flag=True,
    default=True,
    help="Highlight last path in red.",
)
@click.option(
    "--spline-method",
    type=click.Choice([e.value for e in SplineMethod]),
    default=SplineMethod.HERMITE.value,
    help="Spline interpolation method for energy profiles.",
)
@click.option(
    "--savgol-window",
    type=int,
    default=5,
    help="Savitzky-Golay filter window length (for Hermite spline).",
)
@click.option(
    "--savgol-order",
    type=int,
    default=2,
    help="Savitzky-Golay filter polynomial order (for Hermite spline).",
)
@click.option(
    "--draw-reactant",
    type=(float, float, float),
    nargs=3,
    default=(15, 60, 0.1),
    show_default=True,
    help="Positioning for the reactant inset (x, y, rad).",
)
@click.option(
    "--draw-saddle",
    type=(float, float, float),
    nargs=3,
    default=(15, 60, 0.1),
    show_default=True,
    help="Positioning for the saddle inset (x, y, rad).",
)
@click.option(
    "--draw-product",
    type=(float, float, float),
    nargs=3,
    default=(15, 60, 0.1),
    show_default=True,
    help="Positioning for the product inset (x, y, rad).",
)
def main(
    input_pattern,
    con_file,
    additional_con,
    plot_type,
    landscape_mode,
    rc_mode,
    plot_structures,
    plot_mode,
    output_file,
    start,
    end,
    normalize_rc,
    title,
    xlabel,
    ylabel,
    facecolor,
    cmap,
    highlight_last,
    spline_method,
    savgol_window,
    savgol_order,
    draw_reactant,
    draw_saddle,
    draw_product,
):
    """Main entry point for NEB plot script."""
    # Check for IRA dependency
    if ira_mod is None and (
        plot_type == "landscape" or rc_mode == "rmsd" or additional_con
    ):
        log.critical("Needs 'ira_mod' library for landscape, rmsd, or additional-con.")
        log.critical("Please install it to use these options. Exiting.")
        sys.exit(1)

    if plot_structures != "none" and not con_file:
        log.error("--plot-structures requires a --con-file to be provided. Exiting.")
        sys.exit(1)

    smoothing_params = SmoothingParams(
        window_length=savgol_window, polyorder=savgol_order
    )

    # --- Instantiate Inset Position objects ---
    image_pos_reactant = InsetImagePos(*draw_reactant)
    image_pos_saddle = InsetImagePos(*draw_saddle)
    image_pos_product = InsetImagePos(*draw_product)

    # --- Load Structures ---
    atoms_list = None
    if con_file:
        try:
            log.info(f"Reading structures from [cyan]{con_file}[/cyan]")
            atoms_list = ase_read(con_file, index=":")
            log.info(f"Loaded {len(atoms_list)} structures.")
        except Exception as e:
            log.error(f"Failed to read .con file: {e}")
            atoms_list = None
            if plot_type == "landscape" or rc_mode == "rmsd" or additional_con:
                log.critical("Cannot proceed without structures. Exiting.")
                sys.exit(1)

    additional_atoms = None
    add_rmsd_r = None
    add_rmsd_p = None
    if additional_con and atoms_list is not None:
        try:
            log.info(f"Reading additional structure from [cyan]{additional_con}[/cyan]")
            additional_atoms = ase_read(additional_con)
            ira_instance = ira_mod.IRA()
            # Calculate 2D coordinates for this single point
            add_rmsd_r = calculate_rmsd_from_ref(
                [additional_atoms], ira_instance, ref_atom=atoms_list[0]
            )[0]
            add_rmsd_p = calculate_rmsd_from_ref(
                [additional_atoms], ira_instance, ref_atom=atoms_list[-1]
            )[0]
            log.info(f"... RMSD_R = {add_rmsd_r:.3f} Å, RMSD_P = {add_rmsd_p:.3f} Å")
        except Exception as e:
            log.error(f"Failed to read or process --additional-con: {e}")
            additional_atoms = None

    # --- Setup Plot ---
    plt.style.use("bmh")
    # Reset some style params
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    # --- LANDSCAPE PLOT ---
    if plot_type == PlotType.LANDSCAPE.value:
        ira_instance = ira_mod.IRA()
        rmsd_r, rmsd_p = calculate_landscape_coords(atoms_list, ira_instance)

        # For landscape, we only plot the *final* .dat file
        all_file_paths = load_paths(input_pattern)
        final_dat_file = all_file_paths[-1]
        log.info(
            f"Loading data from final path: [yellow]{final_dat_file.name}[/yellow]"
        )
        try:
            path_data = np.loadtxt(final_dat_file, skiprows=1).T
            y_data_column = 2 if plot_mode == PlotMode.ENERGY.value else 4
            z_data = path_data[y_data_column]
            if len(z_data) != len(atoms_list):
                errmsg = (
                    f"Structure count ({len(atoms_list)})"
                    f" != data point count ({len(z_data)}) in {final_dat_file.name}"
                )
                raise ValueError(errmsg)
        except Exception as e:
            log.error(f"Failed to load or parse {final_dat_file.name}: {e}")
            sys.exit(1)

        # Set labels
        xlabel = xlabel or r"RMSD from Reactant ($\AA$)"
        ylabel = ylabel or r"RMSD from Product ($\AA$)"
        z_label = (
            "Relative Energy (eV)"
            if plot_mode == PlotMode.ENERGY.value
            else r"Lowest Eigenvalue (eV/$\AA^2$)"
        )
        if title == "NEB Path":
            title = "NEB Landscape"  # More fitting title

        if landscape_mode == "surface":
            plot_interpolated_landscape(ax, rmsd_r, rmsd_p, z_data, cmap)

        plot_landscape_path(ax, rmsd_r, rmsd_p, z_data, cmap, z_label)

        if atoms_list and plot_structures != "none":
            plot_structure_insets(
                ax,
                atoms_list,
                rmsd_r,
                rmsd_p,
                z_data,
                plot_structures,
                plot_mode,
                draw_reactant=image_pos_reactant,
                draw_saddle=image_pos_saddle,
                draw_product=image_pos_product,
            )

        if additional_atoms:
            ax.plot(
                add_rmsd_r,
                add_rmsd_p,
                marker="*",
                markersize=20,
                color="white",
                zorder=98,
                label="Additional Structure",
            )
            if plot_structures != "none":
                # Use saddle position by default for the "additional" atom
                plot_single_inset(
                    ax,
                    additional_atoms,
                    add_rmsd_r,
                    add_rmsd_p,
                    xybox=(image_pos_saddle.x, image_pos_saddle.y),
                    rad=image_pos_saddle.rad,
                )

        setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor)

    # --- PROFILE PLOT ---
    else:
        rmsd_rc = None
        default_xlabel = r"Reaction Coordinate ($\AA$)"
        if rc_mode == RCMode.RMSD.value and atoms_list is not None:
            ira_instance = ira_mod.IRA()
            rmsd_rc = calculate_rmsd_from_ref(
                atoms_list, ira_instance, ref_atom=atoms_list[0]
            )
            default_xlabel = r"RMSD from Reactant ($\AA$)"
            normalize_rc = False  # Normalizing RMSD doesn't make sense

        # Set labels
        xlabel = xlabel or default_xlabel
        ylabel = ylabel or (
            "Relative Energy (eV)"
            if plot_mode == PlotMode.ENERGY.value
            else r"Lowest Eigenvalue (eV/$\AA^2$)"
        )

        all_file_paths = load_paths(input_pattern)
        file_paths_to_plot = all_file_paths[start:end]
        num_files = len(file_paths_to_plot)
        if num_files == 0:
            log.error("The specified start/end range resulted in zero files. Exiting.")
            sys.exit(1)

        colormap = getattr(cm, cmap)
        color_divisor = (num_files - 1) if num_files > 1 else 1.0

        # Define the plotting function based on mode
        plot_function = (
            lambda ax, pd, c, a, z: plot_energy_path(
                ax, pd, c, a, z, method=spline_method, smoothing=smoothing_params
            )
            if plot_mode == PlotMode.ENERGY.value
            else plot_eigenvalue_path
        )
        y_data_column = 2 if plot_mode == PlotMode.ENERGY.value else 4

        # --- Plotting Loop (Profile) ---
        for idx, file_path in enumerate(file_paths_to_plot):
            try:
                path_data = np.loadtxt(file_path, skiprows=1).T
            except (ValueError, IndexError) as e:
                log.warning(
                    f"Skipping invalid or empty file [yellow]{file_path.name}[/yellow]: {e}"
                )
                continue

            # Check for RC mode override
            if rc_mode == RCMode.RMSD.value and rmsd_rc is not None:
                if rmsd_rc.shape[0] != path_data.shape[1]:
                    log.warning(
                        f"Skipping [yellow]{file_path.name}[/yellow]: Mismatch in image count between .con ({rmsd_rc.shape[0]}) and .dat ({path_data.shape[1]})."
                    )
                    continue
                path_data[1] = rmsd_rc  # Replace path RC with RMSD RC
            elif normalize_rc:
                rc = path_data[1]
                if rc.max() > 0:
                    path_data[1] = rc / rc.max()
                if not xlabel:
                    xlabel = "Normalized Reaction Coordinate"

            rc_for_insets = path_data[1]
            y_for_insets = path_data[y_data_column]

            is_last_file = idx == num_files - 1
            is_first_file = idx == 0

            if highlight_last and is_last_file:
                color, alpha, zorder = "red", 1.0, 20
                plot_function(ax, path_data, color, alpha, zorder)
                # Plot structures on the *last* path
                if atoms_list and plot_structures != "none":
                    plot_structure_insets(
                        ax,
                        atoms_list,
                        rc_for_insets,
                        y_for_insets,
                        y_for_insets,  # Saddle data is Y-data
                        plot_structures,
                        plot_mode,
                        draw_reactant=image_pos_reactant,
                        draw_saddle=image_pos_saddle,
                        draw_product=image_pos_product,
                    )
            else:
                color = colormap(idx / color_divisor)
                alpha = 1.0 if is_first_file else 0.5
                zorder = 10 if is_first_file else 5
                plot_function(ax, path_data, color, alpha, zorder)

        setup_plot_aesthetics(ax, title, xlabel, ylabel, facecolor)
        if rc_mode == RCMode.PATH.value and normalize_rc:
            ax.set_xlim(0, 1)

        # Highlight additional structure (if in RMSD mode)
        if additional_atoms and rc_mode == RCMode.RMSD.value:
            log.info(
                f"Highlighting additional structure at RMSD_R = {add_rmsd_r:.3f} Å"
            )
            ax.axvline(add_rmsd_r, color="black", linestyle=":", linewidth=2, zorder=90)
            if plot_structures != "none":
                # Place inset near the top of the plot
                y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_pos = ax.get_ylim()[0] + 0.9 * y_span
                # Use saddle position by default
                plot_single_inset(
                    ax,
                    additional_atoms,
                    add_rmsd_r,
                    y_pos,
                    xybox=(image_pos_saddle.x, image_pos_saddle.y),
                    rad=image_pos_saddle.rad,
                )

        # Add colorbar for optimization steps
        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max(1, num_files - 1))
        )
        fig.colorbar(sm, ax=ax, label="Optimization Step")

    # --- Finalize ---
    if output_file:
        log.info(f"Saving plot to [green]{output_file}[/green]")
        plt.savefig(output_file, transparent=False, bbox_inches="tight")
    else:
        log.info("Displaying plot interactively.")
        plt.show()


if __name__ == "__main__":
    main()
