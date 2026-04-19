"""Shared plotting helpers for single-ended eOn visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from chemparseplot.parse.projection import compute_projection_basis, project_to_sd
from chemparseplot.plot.neb import plot_structure_strip
from chemparseplot.plot.theme import apply_axis_theme
from matplotlib.gridspec import GridSpec


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
