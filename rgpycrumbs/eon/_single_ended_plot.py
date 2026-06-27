"""Compatibility re-exports for single-ended eOn plotting helpers.

The public single-ended plotting surface now lives in
``chemparseplot.plot.optimization``. This module remains as a thin wrapper so
existing rgpycrumbs imports continue to resolve while the CLI layer is thinned.
"""

from chemparseplot.plot.optimization import (
    OVERLAY_COLORS,
    annotate_endpoint,
    create_landscape_axes,
    default_strip_zoom,
    plot_single_ended_convergence,
    plot_single_ended_profile,
    project_landscape_path,
    render_endpoint_strip,
    save_landscape_figure,
    save_standard_figure,
)

__all__ = [
    "OVERLAY_COLORS",
    "annotate_endpoint",
    "create_landscape_axes",
    "default_strip_zoom",
    "plot_single_ended_convergence",
    "plot_single_ended_profile",
    "project_landscape_path",
    "render_endpoint_strip",
    "save_landscape_figure",
    "save_standard_figure",
]
