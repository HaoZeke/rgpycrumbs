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
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import click
import cmcrameri.cm  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
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


# --- Enumerations, Constants, and Themes ---

# --- Color Definitions ---
RUHI_COLORS = {
    "coral": "#FF655D",
    "sunshine": "#F1DB4B",
    "teal": "#004D40",
    "sky": "#1E88E5",
    "magenta": "#D81B60",
}


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
    facecolor="gray",
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
    facecolor=RUHI_COLORS["teal"],
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


class PlotMode(Enum):
    """Defines the primary data to be plotted (Y-axis or color)."""

    ENERGY = "energy"
    EIGENVALUE = "eigenvalue"


class SplineMethod(Enum):
    """Defines the interpolation method for profile plots."""

    HERMITE = "hermite"  # Cubic Hermite spline (uses derivatives/forces)
    SPLINE = "spline"  # Standard cubic spline (no derivatives)


DEFAULT_INPUT_PATTERN = "neb_*.dat"


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
    ax,
    atoms,
    x_coord,
    y_coord,
    xybox=(15.0, 60.0),
    rad=0.0,
    zoom=0.4,
    arrow_head_length=0.4,
    arrow_head_width=0.4,
    arrow_tail_width=0.1,
):
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
    arrow_head_length: float = 0.4,
    arrow_head_width: float = 0.4,
    arrow_tail_width: float = 0.1,
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


def plot_eigenvalue_path(ax, path_data, color, alpha, zorder, grid_color="white"):
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

    Z = griddata((rmsd_r, rmsd_p), z_data, (X, Y), method="cubic")

    try:
        colormap = mpl.colormaps.get_cmap(cmap)
    except ValueError:
        log.warning(f"Colormap '{cmap}' not in registry. Falling back to 'batlow'.")
        colormap = mpl.colormaps.get_cmap("cmc.batlow")

    ax.contourf(X, Y, Z, levels=20, cmap=colormap, alpha=0.75, zorder=10)


def setup_plot_aesthetics(ax, title, xlabel, ylabel):
    """Applies labels, limits, and other plot aesthetics."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
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
    default="surface",
    help="For landscape plot: 'path' (only 1D path) or 'surface' (interpolated 2D surface).",
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
    theme,
    cmap_profile,
    cmap_landscape,
    facecolor,
    fontsize_base,
    figsize,
    fig_height,
    aspect_ratio,
    dpi,
    zoom_ratio,
    arrow_head_length,
    arrow_head_width,
    arrow_tail_width,
    highlight_last,
    spline_method,
    savgol_window,
    savgol_order,
    draw_reactant,
    draw_saddle,
    draw_product,
):
    """Main entry point for NEB plot script."""
    # --- Theme Selection & Overrides ---
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
    # ---
    setup_global_theme(selected_theme)

    # --- Dependency Checks ---
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

    # --- Parameter Setup ---
    smoothing_params = SmoothingParams(
        window_length=savgol_window, polyorder=savgol_order
    )
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
    # --- Figure Size Logic ---
    final_figsize = None
    if fig_height is not None and aspect_ratio is not None:
        # New method: height and aspect ratio
        width_in = fig_height * aspect_ratio
        final_figsize = (width_in, fig_height)
        log.info(
            f'Using aspect ratio: height=[bold cyan]{fig_height:.2f}"[/bold cyan], '
            f"aspect=[bold cyan]{aspect_ratio:.2f}[/bold cyan] "
            f'-> width=[bold cyan]{width_in:.2f}"[/bold cyan]'
        )
    elif fig_height is not None or aspect_ratio is not None:
        # Error: Only one was provided
        log.error(
            "[bold red]Error:[/bold red] --fig-height and --aspect-ratio must be used together."
        )
        log.error(f"Falling back to default figsize: {figsize}")
        final_figsize = figsize
    else:
        # Old method: --figsize
        final_figsize = figsize
        log.info(
            f'Using figsize: width=[bold cyan]{figsize[0]:.2f}"[/bold cyan], '
            f'height=[bold cyan]{figsize[1]:.2f}"[/bold cyan]'
        )

    fig, ax = plt.subplots(figsize=final_figsize, dpi=dpi)
    apply_plot_theme(ax, selected_theme)

    # --- LANDSCAPE PLOT ---
    if plot_type == PlotType.LANDSCAPE.value:
        ira_instance = ira_mod.IRA()
        rmsd_r, rmsd_p = calculate_landscape_coords(atoms_list, ira_instance)

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
                errmsg = f"Structure count ({len(atoms_list)}) != data point count ({len(z_data)}) in {final_dat_file.name}"
                raise ValueError(errmsg)
        except Exception as e:
            log.error(f"Failed to load or parse {final_dat_file.name}: {e}")
            sys.exit(1)

        xlabel = xlabel or r"RMSD from Reactant ($\AA$)"
        ylabel = ylabel or r"RMSD from Product ($\AA$)"
        z_label = (
            "Relative Energy (eV)"
            if plot_mode == PlotMode.ENERGY.value
            else r"Lowest Eigenvalue (eV/$\AA^2$)"
        )
        if title == "NEB Path":
            title = "NEB Landscape"

        if landscape_mode == "surface":
            plot_interpolated_landscape(
                ax, rmsd_r, rmsd_p, z_data, selected_theme.cmap_landscape
            )

        plot_landscape_path(
            ax, rmsd_r, rmsd_p, z_data, selected_theme.cmap_landscape, z_label
        )

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
                zoom_ratio=zoom_ratio,
                arrow_head_length=arrow_head_length,
                arrow_head_width=arrow_head_width,
                arrow_tail_width=arrow_tail_width,
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
                plot_single_inset(
                    ax,
                    additional_atoms,
                    add_rmsd_r,
                    add_rmsd_p,
                    xybox=(image_pos_saddle.x, image_pos_saddle.y),
                    rad=image_pos_saddle.rad,
                    zoom=zoom_ratio,
                    arrow_head_length=arrow_head_length,
                    arrow_head_width=arrow_head_width,
                    arrow_tail_width=arrow_tail_width,
                )

        setup_plot_aesthetics(ax, title, xlabel, ylabel)

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
            normalize_rc = False

        xlabel = xlabel or default_xlabel
        ylabel = ylabel or (
            "Relative Energy (eV)"
            if plot_mode == PlotMode.ENERGY.value
            else r"Lowest EigenValue (eV/$\AA^2$)"
        )

        all_file_paths = load_paths(input_pattern)
        file_paths_to_plot = all_file_paths[start:end]
        num_files = len(file_paths_to_plot)
        if num_files == 0:
            log.error("The specified start/end range resulted in zero files. Exiting.")
            sys.exit(1)

        try:
            colormap = mpl.colormaps.get_cmap(selected_theme.cmap_profile)
        except ValueError:
            log.warning(
                f"Colormap '{selected_theme.cmap_profile}' not in registry. Falling back to 'batlow'."
            )
            colormap = mpl.colormaps.get_cmap("cmc.batlow")

        color_divisor = (num_files - 1) if num_files > 1 else 1.0

        plot_function = (
            lambda ax, pd, c, a, z: plot_energy_path(
                ax, pd, c, a, z, method=spline_method, smoothing=smoothing_params
            )
            if plot_mode == PlotMode.ENERGY.value
            else lambda ax, pd, c, a, z: plot_eigenvalue_path(
                ax,
                pd,
                c,
                a,
                z,
                grid_color=selected_theme.gridcolor,
            )
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

            if rc_mode == RCMode.RMSD.value and rmsd_rc is not None:
                if rmsd_rc.shape[0] != path_data.shape[1]:
                    log.warning(
                        f"Skipping [yellow]{file_path.name}[/yellow]: Mismatch in image count between .con ({rmsd_rc.shape[0]}) and .dat ({path_data.shape[1]})."
                    )
                    continue
                path_data[1] = rmsd_rc
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
                        arrow_head_length=arrow_head_length,
                        arrow_head_width=arrow_head_width,
                        arrow_tail_width=arrow_tail_width,
                    )
            else:
                color = colormap(idx / color_divisor)
                alpha = 1.0 if is_first_file else 0.5
                zorder = 10 if is_first_file else 5
                plot_function(ax, path_data, color, alpha, zorder)

        setup_plot_aesthetics(ax, title, xlabel, ylabel)
        if rc_mode == RCMode.PATH.value and normalize_rc:
            ax.set_xlim(0, 1)

        if additional_atoms and rc_mode == RCMode.RMSD.value:
            log.info(
                f"Highlighting additional structure at RMSD_R = {add_rmsd_r:.3f} Å"
            )
            ax.axvline(
                add_rmsd_r,
                color=selected_theme.gridcolor,  # Use grid color
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
                    arrow_head_length=arrow_head_length,
                    arrow_head_width=arrow_head_width,
                    arrow_tail_width=arrow_tail_width,
                )

        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max(1, num_files - 1))
        )
        cbar = fig.colorbar(sm, ax=ax, label="Optimization Step")
        cbar.ax.yaxis.set_tick_params(color=selected_theme.textcolor)
        cbar.outline.set_edgecolor(selected_theme.edgecolor)

    # --- Finalize ---
    if output_file:
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


if __name__ == "__main__":
    main()
