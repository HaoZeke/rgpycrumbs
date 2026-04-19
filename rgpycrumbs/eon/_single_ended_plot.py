"""Shared plotting helpers for single-ended eOn visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from chemparseplot.parse.projection import compute_projection_basis, project_to_sd
from chemparseplot.plot.neb import plot_structure_strip
from chemparseplot.plot.optimization import plot_convergence_panel
from chemparseplot.plot.structs import (
    convert_energy,
    convert_energy_curvature,
    eigenvalue_axis_label,
    energy_axis_label,
)
from chemparseplot.plot.theme import apply_axis_theme
from matplotlib.gridspec import GridSpec

OVERLAY_COLORS = ["#004D40", "#FF655D", "#3F51B5", "#FF9800", "#9C27B0", "#009688"]


def create_landscape_axes(*, dpi: int, has_strip: bool, theme, base_size: float = 5.37):
    """Create a landscape figure with an optional structure strip axis."""

    fig = plt.figure(figsize=(base_size, base_size + (1.5 if has_strip else 0)), dpi=dpi)
    if has_strip:
        gs = GridSpec(2, 1, height_ratios=[1, 0.3], hspace=0.15, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1])
        if theme:
            apply_axis_theme(ax_strip, theme)
    else:
        ax = fig.add_subplot(111)
        ax_strip = None

    if theme:
        apply_axis_theme(ax, theme)
    return fig, ax, ax_strip


def project_landscape_path(rmsd_a, rmsd_b, *, project_path: bool, basis=None):
    """Project a single-ended landscape path into display coordinates."""

    if not project_path:
        return rmsd_a, rmsd_b, None
    if basis is None:
        basis = compute_projection_basis(rmsd_a, rmsd_b)
    plot_x, plot_y = project_to_sd(rmsd_a, rmsd_b, basis)
    return plot_x, plot_y, basis


def annotate_endpoint(ax, x: float, y: float, label: str, *, boxed: bool):
    """Annotate an endpoint consistently on optimization landscapes."""

    kwargs = {
        "fontsize": 10,
        "fontweight": "bold",
        "ha": "center",
        "va": "bottom",
        "zorder": 60,
    }
    if boxed:
        kwargs.update(
            {
                "xytext": (0, 6),
                "textcoords": "offset points",
                "bbox": {
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            }
        )
    ax.annotate(label, (x, y), **kwargs)


def default_strip_zoom(structs) -> float:
    """Scale strip zoom gently with atom count."""

    max_atoms = max(len(s) for s in structs) if structs else 10
    return max(0.25, 0.8 * (20 / max(max_atoms, 20)) ** 0.3)


def render_endpoint_strip(
    ax_strip,
    structs,
    labels,
    *,
    strip_zoom,
    rotation,
    theme,
    strip_renderer,
    strip_spacing,
    strip_dividers,
    perspective_tilt,
    xyzrender_config,
):
    """Render the standard endpoint strip for single-ended plots."""

    zoom = strip_zoom if strip_zoom is not None else default_strip_zoom(structs)
    plot_structure_strip(
        ax_strip,
        structs,
        labels,
        zoom=zoom,
        rotation=rotation,
        theme_color=theme.textcolor if theme else "black",
        renderer=strip_renderer,
        col_spacing=strip_spacing,
        show_dividers=strip_dividers,
        perspective_tilt=perspective_tilt,
        xyzrender_config=xyzrender_config,
    )


def save_landscape_figure(fig, output: Path, *, dpi: int, has_strip: bool) -> None:
    """Save optimization landscapes without tight-layout strip warnings."""

    if not has_strip:
        fig.tight_layout()
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(str(output), dpi=dpi)
    plt.close(fig)


def save_standard_figure(fig, output: Path, *, dpi: int) -> None:
    """Save a standard figure with tight layout."""

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_ended_profile(
    trajs,
    labels,
    output: Path,
    dpi: int,
    *,
    energy_unit: str,
    energy_column: str,
    title: str,
    eigen_column: str | None = None,
) -> None:
    """Plot shared single-ended optimization profiles."""

    has_eigen = bool(eigen_column) and any(
        eigen_column in t.dat_df.columns for t in trajs
    )
    fig, axes = plt.subplots(
        1, 2 if has_eigen else 1, figsize=(10 if has_eigen else 5.37, 4), dpi=dpi
    )
    if not has_eigen:
        axes = [axes]

    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        dat = traj.dat_df
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        iters = dat["iteration"].to_numpy()
        energies = convert_energy(dat[energy_column].to_numpy(), energy_unit)
        axes[0].plot(
            iters, energies, "o-", color=color, markersize=4, linewidth=1.5, label=lbl
        )

        if has_eigen and eigen_column and eigen_column in dat.columns:
            eigenvalues = convert_energy_curvature(
                dat[eigen_column].to_numpy(), energy_unit
            )
            axes[1].plot(
                iters,
                eigenvalues,
                "s-",
                color=color,
                markersize=3,
                linewidth=1.2,
                label=lbl,
            )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(energy_axis_label(energy_unit))
    axes[0].set_title(title)
    if len(trajs) > 1:
        axes[0].legend(frameon=False)

    if has_eigen:
        axes[1].axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(eigenvalue_axis_label(energy_unit))
        axes[1].set_title("Eigenvalue vs Iteration")
        if len(trajs) > 1:
            axes[1].legend(frameon=False)

    save_standard_figure(fig, output, dpi=dpi)


def plot_single_ended_convergence(trajs, labels, output: Path, dpi: int) -> None:
    """Plot shared convergence panels for single-ended optimizers."""

    fig, (ax_force, ax_step) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        plot_convergence_panel(ax_force, ax_step, traj.dat_df, color=color)
        ax_force.plot([], [], color=color, label=lbl)
    if len(trajs) > 1:
        ax_force.legend(frameon=False)
    save_standard_figure(fig, output, dpi=dpi)
