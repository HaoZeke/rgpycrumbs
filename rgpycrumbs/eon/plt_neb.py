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
#   "polars",
# ]
# ///

import glob
import io
import logging
import re
import sys
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import click
import cmcrameri.cm  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ase.io import read as ase_read
from ase.io import write as ase_write
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle
from rich.logging import RichHandler
from scipy.interpolate import (
    CubicHermiteSpline,
    RBFInterpolator,
    griddata,
    splev,
    splrep,
)
from scipy.signal import savgol_filter

from rgpycrumbs._aux import _import_from_parent_env

# IRA is optional, use None if not present
ira_mod = _import_from_parent_env("ira_mod")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)
log = logging.getLogger("rich")


# --- Enumerations, Constants, and Themes ---

# --- Color Definitions ---
RUHI_COLORS = {
    "coral": "#FF655D",
    "sunshine": "#F1DB4B",
    "teal": "#004D40",
    "sky": "#1E88E5",
    "magenta": "#D81B60",
}

MIN_PATH_LENGTH = 1e-6


# --- Custom Colormap Generation ---
def build_cmap(hex_list, name):
    """Build and register a LinearSegmentedColormap from a list of hex colors."""
    cols = [c.strip() for c in hex_list]
    cmap = LinearSegmentedColormap.from_list(name, cols, N=256)
    mpl.colormaps.register(cmap)
    return cmap


# Build and register Ruhi colormaps
build_cmap(
    [
        RUHI_COLORS["coral"],
        RUHI_COLORS["sunshine"],
        RUHI_COLORS["teal"],
        RUHI_COLORS["sky"],
        RUHI_COLORS["magenta"],
    ],
    name="ruhi_full",
)
build_cmap(
    [
        RUHI_COLORS["teal"],
        RUHI_COLORS["sky"],  # Low
        RUHI_COLORS["magenta"],  # Mid
        RUHI_COLORS["coral"],  # High
        RUHI_COLORS["sunshine"],
    ],
    name="ruhi_diverging",
)


# --- Theme Dataclass ---
@dataclass(frozen=True)
class PlotTheme:
    """Holds all aesthetic parameters for a matplotlib theme."""

    name: str
    font_family: str
    font_size: int
    facecolor: str
    textcolor: str
    edgecolor: str
    gridcolor: str
    cmap_profile: str  # Colormap for 1D profile paths
    cmap_landscape: str  # Colormap for 2D landscape surface
    highlight_color: str


# --- Theme Definitions ---
BATLOW_THEME = PlotTheme(
    name="cmc.batlow",
    font_family="Atkinson Hyperlegible",
    font_size=12,
    facecolor="white",
    textcolor="black",
    edgecolor="black",
    gridcolor="#FFFFFF",
    cmap_profile="cmc.batlow",
    cmap_landscape="cmc.batlow",
    highlight_color="#FF0000",  # red
)

# Theme 2: "ruhi"
RUHI_THEME = PlotTheme(
    name="ruhi",
    font_family="Atkinson Hyperlegible",
    font_size=12,
    facecolor="white",
    textcolor="black",
    edgecolor="black",
    gridcolor="floralwhite",
    cmap_profile="ruhi_diverging",
    cmap_landscape="ruhi_diverging",
    highlight_color="black",
)

# Updated THEMES dictionary
THEMES = {
    "cmc.batlow": BATLOW_THEME,
    "ruhi": RUHI_THEME,
}


# --- Other Constants ---
class PlotType(Enum):
    """Defines the overall plot type."""

    PROFILE = "profile"
    LANDSCAPE = "landscape"


class RCMode(Enum):
    """Defines the reaction coordinate for profile plots."""

    PATH = "path"  # Default path distance
    RMSD = "rmsd"  # RMSD from reactant
    INDEX = "index"  # Image number (0, 1, 2...)


class PlotMode(Enum):
    """Defines the primary data to be plotted (Y-axis or color)."""

    ENERGY = "energy"
    EIGENVALUE = "eigenvalue"


class SplineMethod(Enum):
    """Defines the interpolation method for profile plots."""

    HERMITE = "hermite"  # Cubic Hermite spline (uses derivatives/forces)
    SPLINE = "spline"  # Standard cubic spline (no derivatives)


DEFAULT_INPUT_PATTERN = "neb_*.dat"
DEFAULT_PATH_PATTERN = "neb_path_*.con"
ROUNDING_DF = 3
RBF_SMOOTHING = 1e-2
IRA_KMAX_DEFAULT = 1.8


@dataclass
class SmoothingParams:
    """Parameters for Savitzky-Golay smoothing of force data."""

    window_length: int = 5
    polyorder: int = 2


# Datastructure for inset positioning
InsetImagePos = namedtuple("InsetImagePos", "x y rad")


# --- Utility Functions ---


def setup_global_theme(theme: PlotTheme):
    """Sets global plt.rcParams based on the theme *before* plot creation."""
    log.info(f"Setting global rcParams for [bold cyan]{theme.name}[/bold cyan] theme")

    font_family_to_use = theme.font_family
    try:
        # Check if the font is available to matplotlib
        mpl.font_manager.findfont(theme.font_family, fallback_to_default=False)
        log.info(f"Font '{theme.font_family}' found by matplotlib.")
    except Exception:
        log.warning(
            f"[bold red]Font '{theme.font_family}' not found.[/bold red] "
            f"Falling back to 'sans-serif'."
        )
        log.warning(
            "For custom fonts to work, they must be installed on your system "
            "and recognized by matplotlib."
        )
        log.warning(
            f"You may need to clear the matplotlib cache: [cyan]{mpl.get_cachedir()}[/cyan]"
        )
        font_family_to_use = "sans-serif"

    plt.rcParams.update(
        {
            "font.size": theme.font_size,
            "font.family": font_family_to_use,
            "text.color": theme.textcolor,
            "axes.labelcolor": theme.textcolor,
            "xtick.color": theme.textcolor,
            "ytick.color": theme.textcolor,
            "axes.edgecolor": theme.edgecolor,
            "axes.titlecolor": theme.textcolor,
            "figure.facecolor": theme.facecolor,  # Set global figure facecolor
            "axes.titlesize": theme.font_size * 1.1,  # Make title a bit larger
            "axes.labelsize": theme.font_size,
            "xtick.labelsize": theme.font_size,
            "ytick.labelsize": theme.font_size,
            "legend.fontsize": theme.font_size,
            "savefig.facecolor": theme.facecolor,  # Ensure saved figs also have bg
            "savefig.transparent": False,
        }
    )


def apply_plot_theme(ax: plt.Axes, theme: PlotTheme):
    """Applies theme properties *specific* to an axis instance."""
    log.info(f"Applying axis-specific theme for [bold cyan]{theme.name}[/bold cyan]")

    # Set axis-specific properties
    ax.set_facecolor(theme.facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.edgecolor)

    # These are now mostly redundant if rcParams worked, but serve as a final failsafe
    ax.tick_params(axis="x", colors=theme.textcolor)
    ax.tick_params(axis="y", colors=theme.textcolor)
    ax.yaxis.label.set_color(theme.textcolor)
    ax.xaxis.label.set_color(theme.textcolor)
    ax.title.set_color(theme.textcolor)


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
    atoms_list: list, ira_instance, ref_atom: "ase.Atoms", ira_kmax: float
) -> np.ndarray:
    """
    Calculates the RMSD of each structure in a list relative to a reference.

    Uses the Iterative Reordering and Alignment (IRA) algorithm to find the
    optimal alignment and permutation before calculating RMSD.

    :param atoms_list: A list of ASE Atoms objects.
    :type atoms_list: list
    :param ira_instance: An instantiated IRA object.
    :type ira_instance: ira_mod.IRA
    :param ref_atom: The reference Atoms object to align against.
    :type ref_atom: ase.Atoms
    :param ira_kmax: kmax factor for IRA.
    :type ira_kmax: float
    :return: An array of RMSD values, one for each structure in `atoms_list`.
    :rtype: np.ndarray
    """
    nat_ref = len(ref_atom)
    typ_ref = ref_atom.get_atomic_numbers()
    coords_ref = ref_atom.get_positions()
    kmax_factor = ira_kmax
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


def calculate_landscape_coords(
    atoms_list: list, ira_instance: "ira_mod.IRA", ira_kmax: float
):
    """
    Calculates 2D landscape coordinates (RMSD-R, RMSD-P) for a path.

    :param atoms_list: List of ASE Atoms objects representing the path.
    :type atoms_list: list
    :param ira_instance: An instantiated IRA object.
    :type ira_instance: ira_mod.IRA
    :param ira_kmax: kmax factor for IRA.
    :type ira_kmax: float
    :return: A tuple of (rmsd_r, rmsd_p) arrays relative to reactant and product.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    log.info(
        "Calculating landscape coordinates using [bold magenta]ira.match[/bold magenta]..."
    )
    rmsd_r = calculate_rmsd_from_ref(
        atoms_list, ira_instance, ref_atom=atoms_list[0], ira_kmax=ira_kmax
    )
    rmsd_p = calculate_rmsd_from_ref(
        atoms_list, ira_instance, ref_atom=atoms_list[-1], ira_kmax=ira_kmax
    )
    log.info("Landscape coordinate calculation complete.")
    return rmsd_r, rmsd_p


def _load_or_compute_data(
    cache_file: Path | None,
    force_recompute: bool,
    validation_check: Callable[[pl.DataFrame], None],
    computation_callback: Callable[[], pl.DataFrame],
    context_name: str,
) -> pl.DataFrame:
    """
    Retrieves data from a parquet cache or triggers a computation callback.

    :param cache_file: The path to the cache file.
    :type cache_file: Path | None
    :param force_recompute: If True, skips loading and forces computation.
    :type force_recompute: bool
    :param validation_check: A function that receives the loaded DataFrame and raises ValueError
                             if the schema appears incorrect (e.g., missing columns).
    :type validation_check: Callable
    :param computation_callback: A function that performs the heavy calculation and returns a DataFrame
                                 if the cache remains unavailable or invalid.
    :type computation_callback: Callable
    :param context_name: Label for logging (e.g., "Profile" or "Landscape").
    :type context_name: str
    :return: The requested data.
    :rtype: pl.DataFrame
    """
    # 1. Attempt to load from cache
    if cache_file and cache_file.exists() and not force_recompute:
        log.info(
            f"Loading cached {context_name} data from [green]{cache_file}[/green]..."
        )
        try:
            df = pl.read_parquet(cache_file)
            validation_check(df)  # Will raise ValueError if invalid
            log.info(f"Loaded {df.height} rows from cache.")
            return df
        except Exception as e:
            log.warning(f"Cache load failed or invalid: {e}. Recomputing...")

    # 2. Compute if cache failed, didn't exist, or recompute requested
    log.info(f"Computing {context_name} data...")
    df = computation_callback()

    # 3. Save to cache
    if cache_file:
        log.info(f"Saving {context_name} cache to [green]{cache_file}[/green]...")
        try:
            df.write_parquet(cache_file)
        except Exception as e:
            log.error(f"Failed to write cache file: {e}")

    return df


# --- Plotting Functions ---
def plot_single_inset(
    ax,
    atoms,
    x_coord,
    y_coord,
    xybox=(15.0, 60.0),
    rad=0.0,
    zoom=0.4,
    ase_rotation="0x, 90y, 0z",
    arrow_head_length=0.4,
    arrow_head_width=0.4,
    arrow_tail_width=0.1,
):
    """
    Renders a single ASE Atoms object and plots it as an inset.

    :param ax: The axis to plot on.
    :type ax: matplotlib.axes.Axes
    :param atoms: The atomic structure to render.
    :type atoms: ase.Atoms
    :param x_coord: The x-data coordinate to anchor the arrow to.
    :type x_coord: float
    :param y_coord: The y-data coordinate to anchor the arrow to.
    :type y_coord: float
    :param xybox: The (x, y) offset in points for placing the image box.
    :type xybox: tuple, optional
    :param rad: The connection style 'rad' parameter for the arrow.
    :type rad: float, optional
    :param zoom: Scale the inset image.
    :type zoom: float
    :param ase_rotation: ASE rotation string for structure insets.
    :type ase_rotation: str
    :param arrow_head_length: Arrow head length for insets.
    :type arrow_head_length: float
    :param arrow_head_width: Arrow head width for insets.
    :type arrow_head_width: float
    :param arrow_tail_width: Arrow tail width for insets.
    :type arrow_tail_width: float
    """
    buf = io.BytesIO()
    ase_write(
        buf, atoms, format="png", rotation=ase_rotation, show_unit_cell=0, scale=35
    )
    buf.seek(0)
    img_data = plt.imread(buf)
    buf.close()

    imagebox = OffsetImage(img_data, zoom=zoom)
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
                head_length=arrow_head_length,
                head_width=arrow_head_width,
                tail_width=arrow_tail_width,
            ),
            "connectionstyle": f"arc3,rad={rad}",
            "linestyle": "-",
            "color": "black",  # NOTE: This is hardcoded, could be themed
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
    zoom_ratio: float = 0.4,
    ase_rotation: str = "0x, 90y, 0z",
    arrow_head_length: float = 0.4,
    arrow_head_width: float = 0.4,
    arrow_tail_width: float = 0.1,
):
    """
    Plots insets for critical points (reactant, saddle, product) or all images.

    :param ax: The axis to plot on.
    :type ax: matplotlib.axes.Axes
    :param atoms_list: List of all ASE Atoms objects for the path.
    :type atoms_list: list
    :param x_coords: Array of x-coordinates (RC or RMSD-R) for each image.
    :type x_coords: np.ndarray
    :param y_coords: Array of y-coordinates (Energy, Eigenvalue, or RMSD-P) for each image.
    :type y_coords: np.ndarray
    :param saddle_data: Data used to find the saddle point. For energy mode, this is the
                        energy array. For eigenvalue mode, this is the eigenvalue array.
    :type saddle_data: np.ndarray
    :param images_to_plot: Which images to plot: "all" or "crit_points".
    :type images_to_plot: str
    :param plot_mode: "energy" or "eigenvalue", used to determine saddle point logic.
    :type plot_mode: str
    :param draw_reactant: Positioning info for the reactant inset.
    :type draw_reactant: InsetImagePos
    :param draw_saddle: Positioning info for the saddle inset.
    :type draw_saddle: InsetImagePos
    :param draw_product: Positioning info for the product inset.
    :type draw_product: InsetImagePos
    :param zoom_ratio: Scale the inset image.
    :type zoom_ratio: float
    :param ase_rotation: ASE rotation string for structure insets.
    :type ase_rotation: str
    :param arrow_head_length: Arrow head length for insets.
    :type arrow_head_length: float
    :param arrow_head_width: Arrow head width for insets.
    :type arrow_head_width: float
    :param arrow_tail_width: Arrow tail width for insets.
    :type arrow_tail_width: float
    """
    if draw_reactant is None:
        draw_reactant = InsetImagePos(15, 60, 0.1)
    if draw_saddle is None:
        draw_saddle = InsetImagePos(15, 60, 0.1)
    if draw_product is None:
        draw_product = InsetImagePos(15, 60, 0.1)
    if len(atoms_list) != len(x_coords) or len(atoms_list) != len(y_coords):
        log.warning(
            f"Mismatch between number of structures ({len(atoms_list)})"
            f" and data points ({len(x_coords)}). Skipping structure plotting."
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
            ax,
            atoms_list[i],
            x_coords[i],
            y_coords[i],
            xybox=xybox,
            rad=rad,
            zoom=zoom_ratio,
            ase_rotation=ase_rotation,
            arrow_head_length=arrow_head_length,
            arrow_head_width=arrow_head_width,
            arrow_tail_width=arrow_tail_width,
        )


def plot_energy_path(
    ax, path_data, color, alpha, zorder, method="hermite", smoothing=SmoothingParams()
):
    """
    Plots a single interpolated energy path and its data points.

    Supports two interpolation methods:

    * 'hermite': Cubic Hermite spline. Uses energy values and their
      derivatives (taken from the parallel force `f_para`). This is
      often a more physically accurate interpolation for NEB paths.
    * 'spline': Standard cubic spline. Ignores derivative information.

    :param ax: The axis to plot on.
    :type ax: matplotlib.axes.Axes
    :param path_data: 2D array of data (from neb_*.dat), transposed.
                      Expected: path_data[1] = rc, path_data[2] = energy, path_data[3] = f_para
    :type path_data: np.ndarray
    :param color: Color for the plot.
    :type color: str or tuple
    :param alpha: Transparency for the plot.
    :type alpha: float
    :param zorder: Plotting layer order.
    :type zorder: int
    :param method: Interpolation method: "hermite" or "spline".
    :type method: str, optional
    :param smoothing: Parameters for Savitzky-Golay filter if using Hermite spline.
    :type smoothing: SmoothingParams, optional
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
        path_length = rc_max - rc_min
        if path_length > MIN_PATH_LENGTH:
            rc_norm_sorted = (rc_sorted - rc_min) / path_length
        else:
            rc_norm_sorted = rc_sorted
        rc_fine_norm = np.linspace(0, 1, num=300)

        if method == "hermite":
            # Smooth the derivatives to reduce noise
            deriv_smooth = savgol_filter(
                deriv,
                window_length=smoothing.window_length,
                polyorder=smoothing.polyorder,
            )
            deriv_smooth_sorted = deriv_smooth[sort_indices]
            deriv_scaled = deriv_smooth_sorted * path_length

            # Use Hermite spline which respects both values and derivatives
            hermite_spline = CubicHermiteSpline(
                rc_norm_sorted, energy_sorted, deriv_scaled
            )
            spline_y = hermite_spline(rc_fine_norm)
        else:
            # Use standard cubic spline
            spline_representation = splrep(rc_norm_sorted, energy_sorted, k=3)
            spline_y = splev(rc_fine_norm, spline_representation)

        # Rescale fine RC back to original units for plotting
        rc_plot_fine = rc_fine_norm * path_length + rc_min

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


def plot_eigenvalue_path(ax, path_data, color, alpha, zorder, grid_color="white"):
    """
    Plots a single interpolated eigenvalue path and its data points.

    :param ax: The matplotlib axes object on which to plot.
    :type ax: matplotlib.axes.Axes
    :param path_data: 2D array of data (from neb_*.dat), transposed.
                      Expected: path_data[1] = rc, path_data[4] = eigenvalue
    :type path_data: np.ndarray
    :param color: Color specification for the plot line and markers.
    :type color: str or tuple
    :param alpha: Transparency level for the plot line and markers.
    :type alpha: float
    :param zorder: Drawing order for the plot elements.
    :type zorder: int
    :param grid_color: Color for the horizontal zero line.
    :type grid_color: str, optional
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
    ax.axhline(0, color=grid_color, linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)


def plot_landscape_path(ax, rmsd_r, rmsd_p, z_data, cmap, z_label):
    """
    Plots the 1D path on the 2D RMSD landscape, colored by z_data.
    ...
    """
    fig = ax.get_figure()
    norm = plt.Normalize(z_data.min(), z_data.max())

    try:
        colormap = mpl.colormaps.get_cmap(cmap)
    except ValueError:
        log.warning(f"Colormap '{cmap}' not in registry. Falling back to 'batlow'.")
        colormap = mpl.colormaps.get_cmap("cmc.batlow")

    points = np.array([rmsd_r, rmsd_p]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=colormap, norm=norm, zorder=30)
    segment_z = (z_data[:-1] + z_data[1:]) / 2
    lc.set_array(segment_z)
    lc.set_linewidth(3)
    ax.add_collection(lc)

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

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, ax=ax, label=z_label)


def plot_interpolated_grid(
    ax, rmsd_r, rmsd_p, z_data, show_pts, cmap, scatter_r=None, scatter_p=None
):
    """
    Generates and plots an interpolated 2D surface (contour plot) with splines.

    This may have artifacts where samples are not present, debug with show_pts
    or use with "last".

    :param ax: The axis to plot on.
    :type ax: matplotlib.axes.Axes
    :param rmsd_r: RMSD from reactant (x-axis).
    :type rmsd_r: np.ndarray
    :param rmsd_p: RMSD from product (y-axis).
    :type rmsd_p: np.ndarray
    :param z_data: Data for coloring the path (z-axis).
    :type z_data: np.ndarray
    :param show_pts: Whether to show scatter points.
    :type show_pts: bool
    :param cmap: Name of the colormap to use.
    :type cmap: str
    :param scatter_r: Optional separate x-coords for scatter points.
    :type scatter_r: np.ndarray, optional
    :param scatter_p: Optional separate y-coords for scatter points.
    :type scatter_p: np.ndarray, optional
    """
    log.info("Generating interpolated 2D surface...")
    log.info("Visualised best with 'last'")
    xi = np.linspace(rmsd_r.min(), rmsd_r.max(), 100)
    yi = np.linspace(rmsd_p.min(), rmsd_p.max(), 100)
    x, y = np.meshgrid(xi, yi)

    z = griddata((rmsd_r, rmsd_p), z_data, (x, y), method="cubic")

    try:
        colormap = mpl.colormaps.get_cmap(cmap)
    except ValueError:
        log.warning(f"Colormap '{cmap}' not in registry. Falling back to 'batlow'.")
        colormap = mpl.colormaps.get_cmap("cmc.batlow")

    ax.contourf(x, y, z, levels=20, cmap=colormap, alpha=0.75, zorder=10)
    if show_pts:
        pts_r = scatter_r if scatter_r is not None else rmsd_r
        pts_p = scatter_p if scatter_p is not None else rmsd_p
        ax.scatter(pts_r, pts_p, c="k", s=12, marker=".", alpha=0.6, zorder=40)


def plot_interpolated_rbf(
    ax,
    rmsd_r,
    rmsd_p,
    z_data,
    show_pts,
    rbf_smoothing,
    cmap,
    scatter_r=None,
    scatter_p=None,
):
    """
    Generates and plots an interpolated 2D surface (contour plot).

    :param ax: The axis to plot on.
    :type ax: matplotlib.axes.Axes
    :param rmsd_r: RMSD from reactant (x-axis).
    :type rmsd_r: np.ndarray
    :param rmsd_p: RMSD from product (y-axis).
    :type rmsd_p: np.ndarray
    :param z_data: Data for coloring the path (z-axis).
    :type z_data: np.ndarray
    :param show_pts: Whether to show scatter points.
    :type show_pts: bool
    :param rbf_smoothing: Smoothing parameter for RBF interpolation.
    :type rbf_smoothing: float
    :param cmap: Name of the colormap to use.
    :type cmap: str
    :param scatter_r: Optional separate x-coords for scatter points.
    :type scatter_r: np.ndarray, optional
    :param scatter_p: Optional separate y-coords for scatter points.
    :type scatter_p: np.ndarray, optional
    """
    log.info("Generating interpolated RBF 2D surface...")
    # Prepare input points for the interpolator: shape (n_samples, 2)
    pts = np.column_stack([np.asarray(rmsd_r).ravel(), np.asarray(rmsd_p).ravel()])
    vals = np.asarray(z_data).ravel()
    rbf = RBFInterpolator(
        pts, vals, kernel="thin_plate_spline", smoothing=rbf_smoothing
    )
    nx, ny = 150, 150
    xg = np.linspace(rmsd_r.min(), rmsd_r.max(), nx)
    yg = np.linspace(rmsd_p.min(), rmsd_p.max(), ny)
    xg, yg = np.meshgrid(xg, yg)

    query_pts = np.column_stack([xg.ravel(), yg.ravel()])
    zflat = rbf(query_pts)  # returns shape (nx*ny,)
    zg = zflat.reshape(xg.shape)

    try:
        colormap = mpl.colormaps.get_cmap(cmap)
    except ValueError:
        log.warning(f"Colormap '{cmap}' not in registry. Falling back to 'batlow'.")
        colormap = mpl.colormaps.get_cmap("cmc.batlow")

    ax.contourf(xg, yg, zg, levels=20, cmap=colormap, alpha=0.75, zorder=10)

    if show_pts:
        pts_r = scatter_r if scatter_r is not None else rmsd_r
        pts_p = scatter_p if scatter_p is not None else rmsd_p
        ax.scatter(pts_r, pts_p, c="k", s=12, marker=".", alpha=0.6, zorder=40)


def setup_plot_aesthetics(ax, title, xlabel, ylabel):
    """Applies labels, limits, and other plot aesthetics."""
    ax.set_xlabel(xlabel, weight="bold")
    ax.set_ylabel(ylabel, weight="bold")
    ax.set_title(title)
    ax.minorticks_on()
    if xlabel != r"RMSD from Reactant ($\AA$)":
        ax.set_xlim(left=0)
    plt.grid(False)
    plt.tight_layout(pad=0.5)


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
    "--rbf-smoothing",
    type=float,
    default=RBF_SMOOTHING,
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
    help="For landscape plot: 'path' (only 1D path) or 'surface' (interpolated 2D surface).",
)
@click.option(
    "--landscape-path",
    type=click.Choice(["last", "all"]),
    default="all",
    help="Last uses an interpolation only on the last path, otherwise use all points.",
)
@click.option(
    "--rc-mode",
    type=click.Choice([e.value for e in RCMode]),
    default=RCMode.PATH.value,
    help="Reaction coordinate for profile plot: 'path' (file's RC) or 'rmsd' (RMSD from reactant).",
)
@click.option(
    "--plot-structures",
    type=click.Choice(["none", "all", "crit_points"]),
    default="none",
    help="Structures to render on the path. Requires --con-file.",
)
@click.option(
    "--surface-type",
    type=click.Choice(["grid", "rbf"]),
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
    type=click.Choice([e.value for e in PlotMode]),
    default=PlotMode.ENERGY.value,
    help="Quantity to plot on y-axis (profile) or color (landscape): 'energy' or 'eigenvalue'.",
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
# --- Theme and Override Options ---
@click.option(
    "--theme",
    type=click.Choice(THEMES.keys(), case_sensitive=False),
    default="ruhi",
    help="The plotting theme to use.",
)
@click.option(
    "--cmap-profile",
    default=None,
    help="Colormap for profile plot (overrides theme default).",
)
@click.option(
    "--cmap-landscape",
    default=None,
    help="Colormap for landscape plot (overrides theme default).",
)
@click.option(
    "--facecolor",
    type=str,
    default=None,
    help="Background color (overrides theme default).",
)
@click.option(
    "--fontsize-base",
    type=int,
    default=None,
    help="Base font size (overrides theme default).",
)
# --- Figure and Inset Options ---
@click.option(
    "--figsize",
    nargs=2,
    type=(float, float),
    default=(10, 7),
    show_default=True,
    help="Figure width, height in inches.",
)
@click.option(
    "--fig-height",
    type=float,
    default=None,
    help="Figure height in inches. Must be used *with* --aspect-ratio.",
)
@click.option(
    "--aspect-ratio",
    type=float,
    default=None,
    help="Figure aspect ratio (width/height). Must be used *with* --fig-height.",
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
    help="ASE rotation string for structure insets (e.g., '45x,30y,0z').",
)
@click.option(
    "--arrow-head-length",
    type=float,
    default=0.4,
    show_default=True,
    help="Arrow head length for insets (points).",
)
@click.option(
    "--arrow-head-width",
    type=float,
    default=0.4,
    show_default=True,
    help="Arrow head width for insets (points).",
)
@click.option(
    "--arrow-tail-width",
    type=float,
    default=0.1,
    show_default=True,
    help="Arrow tail width for insets (points).",
)
# --- Path/Spline Options ---
@click.option(
    "--highlight-last/--no-highlight-last",
    is_flag=True,
    default=True,
    help="Highlight last path (uses theme's 'highlight_color').",
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
# --- Inset Position Options ---
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
@click.option(
    "--cache-file",
    type=click.Path(path_type=Path),
    default=Path(".neb_landscape.parquet"),
    help="Parquet file to cache RMSD calculations (speeds up re-runs).",
)
@click.option(
    "--force-recompute",
    is_flag=True,
    default=False,
    help="Ignore cache and force re-calculation of RMSD.",
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
    # Caching
    cache_file,
    force_recompute,
    ira_kmax,
):
    """Main entry point for NEB plot script."""

    # --- 1. Setup Theme ---
    selected_theme = _setup_theme(
        theme, cmap_profile, cmap_landscape, fontsize_base, facecolor
    )
    setup_global_theme(selected_theme)

    # --- 2. Dependency Checks ---
    _run_dependency_checks(
        plot_type,
        rc_mode,
        additional_con,
        plot_structures,
        con_file,
    )

    # --- 3. Setup Figure ---
    final_figsize = _determine_figsize(figsize, fig_height, aspect_ratio)
    fig, ax = plt.subplots(figsize=final_figsize, dpi=dpi)
    apply_plot_theme(ax, selected_theme)

    # --- 4. Load Structures ---
    atoms_list, additional_atoms_data = _load_structures(
        con_file, additional_con, plot_type, rc_mode
    )

    # --- 5. Delegate to Plotting Function ---
    if plot_type == PlotType.LANDSCAPE.value:
        final_xlabel = xlabel or r"RMSD from Reactant ($\AA$)"
        final_ylabel = ylabel or r"RMSD from Product ($\AA$)"
        final_title = "NEB Landscape" if title == "NEB Path" else title

        _plot_landscape(
            ax=ax,
            atoms_list=atoms_list,
            input_dat_pattern=input_dat_pattern,
            input_path_pattern=input_path_pattern,
            con_file=con_file,
            additional_atoms_data=additional_atoms_data,
            landscape_path=landscape_path,
            landscape_mode=landscape_mode,
            plot_mode=plot_mode,
            surface_type=surface_type,
            plot_structures=plot_structures,
            rbf_smoothing=rbf_smoothing,
            rounding=rounding,
            show_pts=show_pts,
            selected_theme=selected_theme,
            draw_reactant=draw_reactant,
            draw_saddle=draw_saddle,
            draw_product=draw_product,
            zoom_ratio=zoom_ratio,
            ase_rotation=ase_rotation,
            arrow_head_length=arrow_head_length,
            arrow_head_width=arrow_head_width,
            arrow_tail_width=arrow_tail_width,
            cache_file=cache_file,
            force_recompute=force_recompute,
            ira_kmax=ira_kmax,
        )
        setup_plot_aesthetics(ax, final_title, final_xlabel, final_ylabel)

    else:  # Profile Plot
        final_title = title
        final_xlabel, final_ylabel = _get_profile_labels(
            rc_mode, plot_mode, xlabel, ylabel, atoms_list
        )

        _plot_profile(
            ax=ax,
            fig=fig,
            input_dat_pattern=input_dat_pattern,
            atoms_list=atoms_list,
            additional_atoms_data=additional_atoms_data,
            rc_mode=rc_mode,
            plot_mode=plot_mode,
            plot_structures=plot_structures,
            start=start,
            end=end,
            normalize_rc=normalize_rc,
            highlight_last=highlight_last,
            selected_theme=selected_theme,
            spline_method=spline_method,
            savgol_window=savgol_window,
            savgol_order=savgol_order,
            draw_reactant=draw_reactant,
            draw_saddle=draw_saddle,
            draw_product=draw_product,
            zoom_ratio=zoom_ratio,
            ase_rotation=ase_rotation,
            arrow_head_length=arrow_head_length,
            arrow_head_width=arrow_head_width,
            arrow_tail_width=arrow_tail_width,
            cache_file=cache_file,
            force_recompute=force_recompute,
        )
        setup_plot_aesthetics(ax, final_title, final_xlabel, final_ylabel)
        if rc_mode == RCMode.PATH.value and normalize_rc:
            ax.set_xlim(0, 1)

    # --- 6. Finalize ---
    if output_file:
        if not output_file.parent.exists():
            log.info(f"Creating output directory: [cyan]{output_file.parent}[/cyan]")
            output_file.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Saving plot to [green]{output_file}[/green]")
        plt.savefig(
            output_file,
            transparent=False,
            bbox_inches="tight",
            dpi=dpi,
        )
    else:
        log.info("Displaying plot interactively.")
        plt.show()


# --- Helper Functions ---


def _setup_theme(theme, cmap_profile, cmap_landscape, fontsize_base, facecolor):
    """Loads the selected theme and applies any user overrides."""
    selected_theme = THEMES[theme]
    if cmap_profile is not None:
        selected_theme = replace(selected_theme, cmap_profile=cmap_profile)
        log.info(
            f"Overriding theme profile colormap with [bold magenta]{cmap_profile}[/bold magenta]"
        )
    if cmap_landscape is not None:
        selected_theme = replace(selected_theme, cmap_landscape=cmap_landscape)
        log.info(
            f"Overriding theme landscape colormap with [bold magenta]{cmap_landscape}[/bold magenta]"
        )
    if fontsize_base is not None:
        selected_theme = replace(selected_theme, font_size=fontsize_base)
        log.info(
            f"Overriding theme font size with [bold magenta]{fontsize_base}[/bold magenta]"
        )
    if facecolor is not None:
        selected_theme = replace(selected_theme, facecolor=facecolor)
        log.info(
            f"Overriding theme facecolor with [bold magenta]{facecolor}[/bold magenta]"
        )
    return selected_theme


def _run_dependency_checks(
    plot_type,
    rc_mode,
    additional_con,
    plot_structures,
    con_file,
):
    """Validates dependencies and option combinations."""
    if ira_mod is None and (
        plot_type == "landscape" or rc_mode == "rmsd" or additional_con
    ):
        log.critical(
            "The 'ira_mod' library is required for landscape, rmsd, or additional-con features."
        )
        log.critical("Please install it to use these options. Exiting.")
        sys.exit(1)

    if plot_structures != "none" and not con_file:
        log.error("--plot-structures requires a --con-file to be provided. Exiting.")
        sys.exit(1)


def _determine_figsize(figsize, fig_height, aspect_ratio):
    """Determines the final figure size based on user inputs."""
    if fig_height is not None and aspect_ratio is not None:
        width_in = fig_height * aspect_ratio
        final_figsize = (width_in, fig_height)
        log.info(
            f'Using aspect ratio: height=[bold cyan]{fig_height:.2f}"[/bold cyan], '
            f"aspect=[bold cyan]{aspect_ratio:.2f}[/bold cyan] "
            f'-> width=[bold cyan]{width_in:.2f}"[/bold cyan]'
        )
    elif fig_height is not None or aspect_ratio is not None:
        log.error(
            "[bold red]Error:[/bold red] --fig-height and --aspect-ratio must be used together."
        )
        log.error(f"Falling back to default figsize: {figsize}")
        final_figsize = figsize
    else:
        final_figsize = figsize
        log.info(
            f'Using figsize: width=[bold cyan]{figsize[0]:.2f}"[/bold cyan], '
            f'height=[bold cyan]{figsize[1]:.2f}"[/bold cyan]'
        )
    return final_figsize


def _load_structures(con_file, additional_con, plot_type, rc_mode):
    """Loads atoms from .con files and calcs RMSD for additional structure."""
    atoms_list = None
    if con_file:
        try:
            log.info(f"Reading structures from [cyan]{con_file}[/cyan]")
            atoms_list = ase_read(con_file, index=":")
            log.info(f"Loaded {len(atoms_list)} structures.")
        except Exception as e:
            log.error(f"Failed to read .con file: {e}")
            atoms_list = None
            if plot_type == "landscape" or rc_mode == "rmsd":
                log.critical("Cannot proceed without structures. Exiting.")
                sys.exit(1)

    additional_atoms_data = None
    if additional_con and atoms_list is not None:
        try:
            log.info(f"Reading additional structure from [cyan]{additional_con}[/cyan]")
            additional_atoms = ase_read(additional_con)
            ira_instance = ira_mod.IRA()
            add_rmsd_r = calculate_rmsd_from_ref(
                [additional_atoms], ira_instance, ref_atom=atoms_list[0]
            )[0]
            add_rmsd_p = calculate_rmsd_from_ref(
                [additional_atoms], ira_instance, ref_atom=atoms_list[-1]
            )[0]
            log.info(f"... RMSD_R = {add_rmsd_r:.3f} Å, RMSD_P = {add_rmsd_p:.3f} Å")
            additional_atoms_data = (additional_atoms, add_rmsd_r, add_rmsd_p)
        except Exception as e:
            log.error(f"Failed to read or process --additional-con: {e}")

    return atoms_list, additional_atoms_data


def _validate_data_atoms_match(z_data, atoms, dat_file_name):
    """Checks if data points count matches structure count, exits on mismatch."""
    if len(z_data) != len(atoms):
        errmsg = (
            f"Structure count ({len(atoms)}) != data point count "
            f"({len(z_data)}) in {dat_file_name}"
        )
        log.error(errmsg)
        raise ValueError(errmsg)


def _get_profile_labels(rc_mode, plot_mode, xlabel, ylabel, atoms_list):
    """Determines default labels for profile plots."""
    default_xlabel = r"Reaction Coordinate ($\AA$)"

    if rc_mode == RCMode.RMSD.value and atoms_list is not None:
        default_xlabel = r"RMSD from Reactant ($\AA$)"
    elif rc_mode == RCMode.INDEX.value:
        default_xlabel = "Image Index"

    final_xlabel = xlabel or default_xlabel

    default_ylabel = (
        "Relative Energy (eV)"
        if plot_mode == PlotMode.ENERGY.value
        else r"Lowest EigenValue (eV/$\AA^2$)"
    )
    final_ylabel = ylabel or default_ylabel

    return final_xlabel, final_ylabel


def _aggregate_all_paths(
    all_dat_paths,
    all_con_paths,
    y_data_column,
    ira_instance,
    cache_file: Path | None = None,
    force_recompute: bool = False,
    ira_kmax: float = IRA_KMAX_DEFAULT,
):
    """
    Loads and aggregates data from all .dat and .con files for 2D surface using
    Polars and averaging points within each bin. Optionally, a cache file is
    used.

    Returns:
      df_raw
    where df_raw is a Polars DataFrame with columns: [r, p, z, step]
    """

    def validate_landscape_cache(df: pl.DataFrame):
        if "p" not in df.columns:
            verr = "Cache missing 'p' column (looks like profile data)."
            raise ValueError(verr)

    def compute_landscape_data() -> pl.DataFrame:
        all_dfs = []

        # Synchronization check
        paths_dat = all_dat_paths
        paths_con = all_con_paths
        if len(paths_dat) != len(paths_con):
            log.warning(f"Mismatch: {len(paths_dat)} dat vs {len(paths_con)} con.")
            min_len = min(len(paths_dat), len(paths_con))
            paths_dat = paths_dat[:min_len]
            paths_con = paths_con[:min_len]

        for step_idx, (dat_file, con_file_step) in enumerate(
            zip(paths_dat, paths_con, strict=True)
        ):
            try:
                path_data = np.loadtxt(dat_file, skiprows=1).T
                z_data_step = path_data[y_data_column]
                atoms_list_step = ase_read(con_file_step, index=":")
                _validate_data_atoms_match(z_data_step, atoms_list_step, dat_file.name)

                rmsd_r, rmsd_p = calculate_landscape_coords(
                    atoms_list_step, ira_instance, ira_kmax
                )

                all_dfs.append(
                    pl.DataFrame(
                        {
                            "r": rmsd_r,
                            "p": rmsd_p,
                            "z": z_data_step,
                            "step": int(step_idx),
                        }
                    )
                )
            except Exception as e:
                log.warning(f"Failed to process step {step_idx} ({dat_file.name}): {e}")
                continue

        if not all_dfs:
            rerr = "No data could be aggregated from files."
            raise RuntimeError(rerr)

        return pl.concat(all_dfs)

    # Execute via Handler
    return _load_or_compute_data(
        cache_file=cache_file,
        force_recompute=force_recompute,
        validation_check=validate_landscape_cache,
        computation_callback=compute_landscape_data,
        context_name="Landscape",
    )


# --- Main Plotting Functions ---


def _plot_landscape(
    ax,
    atoms_list,
    input_dat_pattern,
    input_path_pattern,
    con_file,
    additional_atoms_data,
    landscape_path,
    landscape_mode,
    plot_mode,
    surface_type,
    plot_structures,
    show_pts,
    rbf_smoothing,
    rounding,
    selected_theme,
    cache_file,
    force_recompute,
    ira_kmax,
    # Inset args
    draw_reactant,
    draw_saddle,
    draw_product,
    zoom_ratio,
    ase_rotation,
    arrow_head_length,
    arrow_head_width,
    arrow_tail_width,
):
    """Handles all logic for drawing 2D landscape plots."""
    ira_instance = ira_mod.IRA()
    # --- Efficient index-based pairing & truncation (fast; no Path.resolve) ---
    # 1) discover .dat and .con step files
    all_dat_paths = sorted(
        Path(p) for p in glob.glob(input_dat_pattern)
    )  # e.g. neb_*.dat
    con_pattern = str(input_path_pattern)  # e.g. neb_path_*.con
    all_con_paths = sorted(Path(p) for p in glob.glob(con_pattern))

    if not all_dat_paths:
        log.critical(
            f"No .dat files found matching pattern: {input_dat_pattern}. Exiting."
        )
        sys.exit(1)

    if not all_con_paths:
        log.warning(
            f"No .con files found matching pattern: {con_pattern}. Falling back to single --con-file: {con_file.name}"
        )
        all_con_paths = [con_file]

    # fast helper: extract numeric index from filename (returns None if none found)
    _num_re = re.compile(r"(\d{1,6})")

    def _index_from_name(p: Path) -> int | None:
        m = _num_re.search(p.name)
        if not m:
            return None
        # preserve numeric value (leading zeros ignored)
        return int(m.group(1))

    # build dicts index -> path for dat and con (only keep entries with numeric indices)
    dat_index_map = {}
    for p in all_dat_paths:
        idx = _index_from_name(p)
        if idx is not None:
            dat_index_map[idx] = p

    con_index_map = {}
    for p in all_con_paths:
        idx = _index_from_name(p)
        if idx is not None:
            con_index_map[idx] = p

    # 4) If both maps have numeric indices, build sorted matched index list
    common_indices = sorted(set(dat_index_map.keys()) & set(con_index_map.keys()))
    if common_indices:
        log.info(
            f"Found {len(common_indices)} numerically-matched dat/con step indices."
        )
        # If user supplied a specific --con-file and it has a numeric index,
        # truncate at that index
        supplied_idx = (
            _index_from_name(Path(con_file)) if con_file is not None else None
        )
        if supplied_idx is not None:
            # find the last index <= supplied_idx that exists in the common set
            # (if supplied index not present, use the largest common index < supplied_idx)
            allowed = [i for i in common_indices if i <= supplied_idx]
            if not allowed:
                log.warning(
                    f"Supplied --con-file index {supplied_idx} not found"
                    "among step files. Using full range of available indices."
                )
                use_indices = common_indices
            else:
                max_allowed = max(allowed)
                use_indices = [i for i in common_indices if i <= max_allowed]
                log.info(
                    f"Truncating to indices <= {max_allowed} (user requested {supplied_idx})."
                    f"Using {len(use_indices)} steps."
                )
        else:
            # no numeric index in supplied con_file or not provided: use all common indices
            use_indices = common_indices

        # Build ordered lists of matched dat and con paths by index
        all_dat_paths = [dat_index_map[i] for i in use_indices]
        all_con_paths = [con_index_map[i] for i in use_indices]

    else:
        # No numeric indices found (filenames don't contain numbers) -> fallback to positional/truncation
        log.debug(
            "No numeric indices found in filenames, falling back to positional truncation behavior."
        )
        # If supplied con_file appears among discovered step files, truncate by name match (cheap)
        if con_file is not None:
            supplied_name = Path(con_file).name
            names = [p.name for p in all_con_paths]
            if supplied_name in names:
                idx = names.index(supplied_name)
                log.info(
                    f"Truncating step .con list to user-provided file: {all_con_paths[idx].name} (index {idx})"
                )
                all_con_paths = all_con_paths[: idx + 1]
            else:
                log.debug(
                    "Provided --con-file not found among discovered step .con files by name;"
                    " proceeding with discovered step files."
                )
        # Finally truncate to min length to pair dat/con by order
        if len(all_dat_paths) != len(all_con_paths):
            min_len = min(len(all_dat_paths), len(all_con_paths))
            if min_len == 0:
                log.critical(
                    "No matching .dat or .con files available after fallback truncation. Exiting."
                )
                sys.exit(1)
            log.info(
                f"Truncating lists to common length {min_len}: "
                f"{len(all_dat_paths)} .dat files and {len(all_con_paths)} .con files."
            )
            all_dat_paths = all_dat_paths[:min_len]
            all_con_paths = all_con_paths[:min_len]

    if not all_dat_paths:
        log.critical(
            f"No .dat files found matching pattern: {input_dat_pattern}. Exiting."
        )
        sys.exit(1)
    if not all_con_paths:
        log.warning(f"No .con files found matching pattern: {con_pattern}.")
        log.warning(f"Falling back to single --con-file: {con_file.name}")
        all_con_paths = [con_file]

    y_data_column = 2 if plot_mode == PlotMode.ENERGY.value else 4
    z_label = (
        "Relative Energy (eV)"
        if plot_mode == PlotMode.ENERGY.value
        else r"Lowest Eigenvalue (eV/$\AA^2$)"
    )

    # --- Load Data (With Cache) ---
    # We now get a RAW dataframe with a 'step' column
    df_raw = _aggregate_all_paths(
        all_dat_paths,
        all_con_paths,
        y_data_column,
        ira_instance,
        cache_file=cache_file,
        force_recompute=force_recompute,
        ira_kmax=ira_kmax,
    )

    # --- Prepare Data for Surface Interpolation ---
    # We group and mean-aggregate for the surface generation to handle dense spots
    if landscape_path == "last":
        # Only use the final step for the surface
        max_step = df_raw["step"].max()
        df_surface_source = df_raw.filter(pl.col("step") == max_step)
    else:
        # Use all steps
        df_surface_source = df_raw

    # Rounding and Grouping for Surface Grid
    df_binned = df_surface_source.with_columns(
        pl.col("r").round(rounding).alias("r_rnd"),
        pl.col("p").round(rounding).alias("p_rnd"),
    )
    df_grouped = (
        df_binned.group_by(["r_rnd", "p_rnd"])
        .agg(
            pl.col("r").mean().alias("r_mean"),
            pl.col("p").mean().alias("p_mean"),
            pl.col("z").mean().alias("z_mean"),
        )
        .sort(["r_mean", "p_mean"])
    )

    rmsd_r = df_grouped["r_mean"].to_numpy()
    rmsd_p = df_grouped["p_mean"].to_numpy()
    z_data = df_grouped["z_mean"].to_numpy()
    all_pts_r = df_raw["r"].to_numpy()
    all_pts_p = df_raw["p"].to_numpy()

    # --- Plot Surface ---
    if landscape_mode == "surface":
        if surface_type == "grid":
            plot_interpolated_grid(
                ax,
                rmsd_r,
                rmsd_p,
                z_data,
                show_pts,
                selected_theme.cmap_landscape,
                scatter_r=all_pts_r,
                scatter_p=all_pts_p,
            )
        else:  # "rbf"
            plot_interpolated_rbf(
                ax,
                rmsd_r,
                rmsd_p,
                z_data,
                show_pts,
                rbf_smoothing,
                selected_theme.cmap_landscape,
                scatter_r=all_pts_r,
                scatter_p=all_pts_p,
            )

    # --- Plot Final Path Overlay ---
    # We can get this from the DF or the final file. Using the DF guarantees consistency.
    max_step = df_raw["step"].max()
    df_final = df_raw.filter(pl.col("step") == max_step)
    final_rmsd_r = df_final["r"].to_numpy()
    final_rmsd_p = df_final["p"].to_numpy()
    final_z_data = df_final["z"].to_numpy()

    plot_landscape_path(
        ax,
        final_rmsd_r,
        final_rmsd_p,
        final_z_data,
        selected_theme.cmap_landscape,
        z_label,
    )

    # --- Plot Structures (Insets) ---
    image_pos_reactant = InsetImagePos(*draw_reactant)
    image_pos_saddle = InsetImagePos(*draw_saddle)
    image_pos_product = InsetImagePos(*draw_product)

    if plot_structures != "none":
        # Need atoms list for insets.
        # Ensure we use the atoms from the *final* path to match the overlay.
        # (Assuming 'atoms_list' passed in main corresponds to the final CON file)
        plot_structure_insets(
            ax,
            atoms_list,
            final_rmsd_r,
            final_rmsd_p,
            final_z_data,
            plot_structures,
            plot_mode,
            draw_reactant=image_pos_reactant,
            draw_saddle=image_pos_saddle,
            draw_product=image_pos_product,
            zoom_ratio=zoom_ratio,
            ase_rotation=ase_rotation,
            arrow_head_length=arrow_head_length,
            arrow_head_width=arrow_head_width,
            arrow_tail_width=arrow_tail_width,
        )

    # --- Additional Structure ---
    if additional_atoms_data:
        additional_atoms, add_rmsd_r, add_rmsd_p = additional_atoms_data
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
            plot_single_inset(
                ax,
                additional_atoms,
                add_rmsd_r,
                add_rmsd_p,
                xybox=(image_pos_saddle.x, image_pos_saddle.y),
                rad=image_pos_saddle.rad,
                zoom=zoom_ratio,
                ase_rotation=ase_rotation,
                arrow_head_length=arrow_head_length,
                arrow_head_width=arrow_head_width,
                arrow_tail_width=arrow_tail_width,
            )


def _plot_profile(
    ax,
    fig,
    input_dat_pattern,
    atoms_list,
    additional_atoms_data,
    rc_mode,
    plot_mode,
    plot_structures,
    start,
    end,
    normalize_rc,
    highlight_last,
    selected_theme,
    # Spline args
    spline_method,
    savgol_window,
    savgol_order,
    # Inset args
    draw_reactant,
    draw_saddle,
    draw_product,
    zoom_ratio,
    ase_rotation,
    arrow_head_length,
    arrow_head_width,
    arrow_tail_width,
    cache_file=None,
    force_recompute=False,
):
    """Handles all logic for drawing 1D profile plots."""
    rmsd_rc = None
    if rc_mode == RCMode.RMSD.value and atoms_list is not None:

        def validate_profile_cache(df: pl.DataFrame):
            if "p" in df.columns:
                verr = "Cache contains 'p' column (looks like landscape data)."
                raise ValueError(verr)
            if df.height != len(atoms_list):
                verr = f"Size mismatch: {df.height} vs {len(atoms_list)} structures."
                raise ValueError(verr)

        def compute_profile_data() -> pl.DataFrame:
            ira_instance = ira_mod.IRA()
            r_vals = calculate_rmsd_from_ref(
                atoms_list, ira_instance, ref_atom=atoms_list[0]
            )
            return pl.DataFrame({"r": r_vals})

        try:
            df_rmsd = _load_or_compute_data(
                cache_file=cache_file,
                force_recompute=force_recompute,
                validation_check=validate_profile_cache,
                computation_callback=compute_profile_data,
                context_name="Profile RMSD",
            )
            rmsd_rc = df_rmsd["r"].to_numpy()
            normalize_rc = False  # Disable normalization if using RMSD
        except Exception as e:
            log.error(f"Could not secure RMSD data: {e}")
            # Fallback or exit depending on strictness; here we proceed without RMSD
            rmsd_rc = None

    all_file_paths = load_paths(input_dat_pattern)
    file_paths_to_plot = all_file_paths[start:end]
    num_files = len(file_paths_to_plot)
    if num_files == 0:
        log.error("The specified start/end range resulted in zero files. Exiting.")
        sys.exit(1)

    try:
        colormap = mpl.colormaps.get_cmap(selected_theme.cmap_profile)
    except ValueError:
        log.warning(
            f"Colormap '{selected_theme.cmap_profile}' not in registry."
            " Falling back to 'batlow'."
        )
        colormap = mpl.colormaps.get_cmap("cmc.batlow")

    color_divisor = (num_files - 1) if num_files > 1 else 1.0

    # Hermite spline requires dE/dx. The files contain dE/ds (force w.r.t path).
    # If x is RMSD or Index, dE/dx != dE/ds, so Hermite construction fails.
    if rc_mode in [RCMode.RMSD.value, RCMode.INDEX.value]:
        if spline_method == SplineMethod.HERMITE.value:
            log.warning(
                f"Hermite spline invalid for rc-mode='{rc_mode}' "
                "(forces are w.r.t path length). Switching to standard cubic spline."
            )
            spline_method = SplineMethod.SPLINE.value

    smoothing_params = SmoothingParams(
        window_length=savgol_window, polyorder=savgol_order
    )

    plot_function = (
        lambda ax, pd, c, a, z: plot_energy_path(
            ax, pd, c, a, z, method=spline_method, smoothing=smoothing_params
        )
        if plot_mode == PlotMode.ENERGY.value
        else lambda ax, pd, c, a, z: plot_eigenvalue_path(
            ax, pd, c, a, z, grid_color=selected_theme.gridcolor
        )
    )
    y_data_column = 2 if plot_mode == PlotMode.ENERGY.value else 4

    image_pos_reactant = InsetImagePos(*draw_reactant)
    image_pos_saddle = InsetImagePos(*draw_saddle)
    image_pos_product = InsetImagePos(*draw_product)

    # --- Plotting Loop ---
    for idx, file_path in enumerate(file_paths_to_plot):
        try:
            path_data = np.loadtxt(file_path, skiprows=1).T
        except (ValueError, IndexError) as e:
            log.warning(
                f"Skipping invalid or empty file [yellow]{file_path.name}[/yellow]: {e}"
            )
            continue

        if rc_mode == RCMode.RMSD.value and rmsd_rc is not None:
            # Check dimensions first
            if len(rmsd_rc) != path_data.shape[1]:
                log.warning(f"Skipping {file_path.name}: Dimension mismatch.")
                continue
            path_data[1] = rmsd_rc

        elif rc_mode == RCMode.INDEX.value:
            # Replace X with integer indices: 0, 1, 2...
            num_images = path_data.shape[1]
            path_data[1] = np.arange(num_images)

        elif normalize_rc and rc_mode != RCMode.INDEX.value:
            rc = path_data[1]
            if rc.max() > 0:
                path_data[1] = rc / rc.max()

        rc_for_insets = path_data[1]
        y_for_insets = path_data[y_data_column]

        is_last_file = idx == num_files - 1
        is_first_file = idx == 0

        if highlight_last and is_last_file:
            color, alpha, zorder = selected_theme.highlight_color, 1.0, 20
            plot_function(ax, path_data, color, alpha, zorder)
            if atoms_list and plot_structures != "none":
                plot_structure_insets(
                    ax,
                    atoms_list,
                    rc_for_insets,
                    y_for_insets,
                    y_for_insets,
                    plot_structures,
                    plot_mode,
                    draw_reactant=image_pos_reactant,
                    draw_saddle=image_pos_saddle,
                    draw_product=image_pos_product,
                    zoom_ratio=zoom_ratio,
                    ase_rotation=ase_rotation,
                    arrow_head_length=arrow_head_length,
                    arrow_head_width=arrow_head_width,
                    arrow_tail_width=arrow_tail_width,
                )
        else:
            color = colormap(idx / color_divisor)
            alpha = 1.0 if is_first_file else 0.5
            zorder = 10 if is_first_file else 5
            plot_function(ax, path_data, color, alpha, zorder)

    # --- Add Colorbar ---
    sm = plt.cm.ScalarMappable(
        cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max(1, num_files - 1))
    )
    cbar = fig.colorbar(sm, ax=ax, label="Optimization Step")
    cbar.ax.yaxis.set_tick_params(color=selected_theme.textcolor)
    cbar.outline.set_edgecolor(selected_theme.edgecolor)
    if rc_mode == RCMode.INDEX.value:
        # Force a major tick for every single image
        ax.set_xticks(np.arange(num_images))
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    # --- Add Additional Structure ---
    if additional_atoms_data and rc_mode == RCMode.RMSD.value:
        additional_atoms, add_rmsd_r, _ = additional_atoms_data
        log.info(f"Highlighting additional structure at RMSD_R = {add_rmsd_r:.3f} Å")
        ax.axvline(
            add_rmsd_r,
            color=selected_theme.gridcolor,
            linestyle=":",
            linewidth=2,
            zorder=90,
        )
        if plot_structures != "none":
            y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_pos = ax.get_ylim()[0] + 0.9 * y_span
            plot_single_inset(
                ax,
                additional_atoms,
                add_rmsd_r,
                y_pos,
                xybox=(image_pos_saddle.x, image_pos_saddle.y),
                rad=image_pos_saddle.rad,
                zoom=zoom_ratio,
                ase_rotation=ase_rotation,
                arrow_head_length=arrow_head_length,
                arrow_head_width=arrow_head_width,
                arrow_tail_width=arrow_tail_width,
            )


if __name__ == "__main__":
    main()
