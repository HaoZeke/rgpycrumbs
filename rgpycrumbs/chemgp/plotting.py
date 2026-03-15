# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""ChemGP plotting functions.

This module provides pure plotting functions for ChemGP data visualization.
All functions are format-agnostic and delegate to chemparseplot for actual
plotting. No file I/O, no CLI logic - pure functions only.

.. versionadded:: 1.7.0
    Extracted from chemgp.plt_gp to standalone module.
"""

from pathlib import Path

import numpy as np
from typing import Any

from rgpycrumbs._aux import ensure_import

# Lazy imports for optional dependencies
_pd = None
_plt = None
_chemparseplot = None


def _get_pd():
    """Get pandas module."""
    global _pd
    if _pd is None:
        _pd = ensure_import("pandas")
    return _pd


def _get_plt():
    """Get matplotlib.pyplot module."""
    global _plt
    if _plt is None:
        _plt = ensure_import("matplotlib.pyplot")
    return _plt


def _get_chemparseplot_plot_chemgp():
    """Get chemparseplot.plot.chemgp module."""
    global _chemparseplot
    if _chemparseplot is None:
        _chemparseplot = ensure_import("chemparseplot.plot.chemgp")
    return _chemparseplot

from functools import lru_cache

def safe_plot(func):
    """Decorator for graceful error handling in plot generation.
    
    Catches common errors and logs helpful messages.
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            log.error("Input file not found: %s", e.filename)
            raise
        except KeyError as e:
            log.error("Missing HDF5 group or key: %s", e)
            raise
        except Exception as e:
            log.error("Plot generation failed: %s", type(e).__name__)
            raise
    return wrapper



# MB surfaces need [-200, 50], LEPS need [-5, 5]
_CLAMP_PRESETS = {
    "mb": (-200.0, 50.0, 25.0),  # (lo, hi, contour_step)
    "leps": (-5.0, 5.0, 0.5),
}


@lru_cache(maxsize=128)
def detect_clamp(filename: str):
    """Detect energy clamping preset from filename.
    
    Returns
    -------
    tuple
        (clamp_lo, clamp_hi, contour_step) or (None, None, None)
    """
    stem = filename.lower()
    for prefix, (lo, hi, step) in _CLAMP_PRESETS.items():
        if prefix in stem:
            return lo, hi, step
    return None, None, None


def save_plot(fig: Any, output: Path, dpi: int) -> None:
    """Save a plotnine ggplot or matplotlib Figure to file."""
    plt = _get_plt()
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, plt.Figure):
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        # plotnine ggplot
        fig.save(str(output), dpi=dpi, verbose=False)


def plot_convergence(df: Any, output: Path, *, x: str = "oracle_calls", y=None, color: str = "method", conv_tol=None, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot force/energy convergence vs oracle calls."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    # Auto-detect y column
    if y is None:
        y = "force_norm"
        for candidate in ["ci_force", "max_fatom", "max_force"]:
            if candidate in df.columns:
                y = candidate
                break

    fig = plot_chemgp.plot_convergence_curve(
        df, x=x, y=y, color=color, conv_tol=conv_tol, width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_surface(gx: np.ndarray, gy: np.ndarray, data: np.ndarray, output: Path, *, paths=None, points=None, clamp_lo=None, clamp_hi=None, contour_step=None, levels=None, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot 2D PES contour."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    if levels is None and clamp_lo is not None and clamp_hi is not None:
        levels = np.linspace(clamp_lo, clamp_hi, 25)

    fig = plot_chemgp.plot_surface_contour(
        gx, gy, data, paths=paths, points=points,
        clamp_lo=clamp_lo, clamp_hi=clamp_hi, levels=levels,
        contour_step=contour_step, width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_gp_quality(grids: dict, true_e: np.ndarray, xc: np.ndarray | None, yc: np.ndarray | None, output: Path, *, clamp_lo=None, clamp_hi=None, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot GP surrogate quality progression."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_gp_progression(
        grids, true_e, xc, yc,
        clamp_lo=clamp_lo, clamp_hi=clamp_hi, width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_rff(df: Any, output: Path, *, exact_e_mae: float = 0.0, exact_g_mae: float = 0.0, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot RFF approximation quality vs exact GP."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_rff_quality(
        df, exact_e_mae=exact_e_mae, exact_g_mae=exact_g_mae,
        width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_nll(gx: np.ndarray, gy: np.ndarray, nll_data: np.ndarray, output: Path, *, grid_grad_norm=None, optimum=None, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot MAP-NLL landscape."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_nll_landscape(
        gx, gy, nll_data,
        grid_grad_norm=grid_grad_norm, optimum=optimum,
        width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_sensitivity(x_vals: np.ndarray, y_true: np.ndarray, panels: dict, output: Path, *, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot hyperparameter sensitivity grid."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_hyperparameter_sensitivity(
        x_vals, y_true, panels, width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_trust(x_slice: np.ndarray, e_true: np.ndarray, e_pred: np.ndarray, e_std: np.ndarray, in_trust: np.ndarray, output: Path, *, train_x=None, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot trust region illustration."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_trust_region(
        x_slice, e_true, e_pred, e_std, in_trust,
        train_x=train_x, width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_variance(gx: np.ndarray, gy: np.ndarray, energy: np.ndarray, var_data: np.ndarray, output: Path, *, train_points=None, stationary=None, clamp_lo=None, clamp_hi=None, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot GP variance overlaid on PES."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_variance_overlay(
        gx, gy, energy, var_data,
        train_points=train_points, stationary=stationary if stationary else None,
        clamp_lo=clamp_lo, clamp_hi=clamp_hi, width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_fps(selected_x: np.ndarray, selected_y: np.ndarray, pruned_x: np.ndarray, pruned_y: np.ndarray, output: Path, *, width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot FPS subset visualization."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_fps_projection(
        selected_x, selected_y, pruned_x, pruned_y,
        width=width, height=height,
    )
    save_plot(fig, output, dpi)


def plot_profile(df: Any, output: Path, *, x: str = "image", y: str = "energy", color: str = "method", width: float = 7.0, height: float = 5.0, dpi: int = 300) -> None:
    """Plot NEB energy profile."""
    plot_chemgp = _get_chemparseplot_plot_chemgp()
    
    fig = plot_chemgp.plot_energy_profile(
        df, x=x, y=y, color=color, width=width, height=height,
    )
    save_plot(fig, output, dpi)
