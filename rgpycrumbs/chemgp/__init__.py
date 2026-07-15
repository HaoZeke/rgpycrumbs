# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ChemGP plotting and I/O utilities (compatibility re-exports).

Parsing and plotting live in **chemparseplot**. This package re-exports them
for backward compatibility. Prefer::

    from chemparseplot.plot.chemgp import plot_surface_contour
    from chemparseplot.parse.chemgp_hdf5 import read_h5_points

The CLI entry point remains ``plot_gp.py`` (PEP 723, dispatched by
``rgpycrumbs.cli``).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "rgpycrumbs.chemgp library re-exports are deprecated; import from "
    "chemparseplot.plot.chemgp / chemparseplot.parse.chemgp_hdf5 instead. "
    "The plot_gp CLI under rgpycrumbs remains supported.",
    DeprecationWarning,
    stacklevel=2,
)

from chemparseplot.parse.chemgp_hdf5 import (
    read_h5_grid,
    read_h5_metadata,
    read_h5_path,
    read_h5_points,
    read_h5_table,
)
from chemparseplot.plot.chemgp import (
    detect_clamp,
    plot_convergence_curve,
    plot_energy_profile,
    plot_fps_projection,
    plot_gp_progression,
    plot_hyperparameter_sensitivity,
    plot_nll_landscape,
    plot_rff_quality,
    plot_surface_contour,
    plot_trust_region,
    plot_variance_overlay,
    save_plot,
)

# Backward-compatible aliases for old names
plot_convergence = plot_convergence_curve
plot_fps = plot_fps_projection
plot_gp_quality = plot_gp_progression
plot_nll = plot_nll_landscape
plot_profile = plot_energy_profile
plot_rff = plot_rff_quality
plot_sensitivity = plot_hyperparameter_sensitivity
plot_surface = plot_surface_contour
plot_trust = plot_trust_region
plot_variance = plot_variance_overlay

__all__ = [
    "detect_clamp",
    "plot_convergence",
    "plot_convergence_curve",
    "plot_energy_profile",
    "plot_fps",
    "plot_fps_projection",
    "plot_gp_progression",
    "plot_gp_quality",
    "plot_hyperparameter_sensitivity",
    "plot_nll",
    "plot_nll_landscape",
    "plot_profile",
    "plot_rff",
    "plot_rff_quality",
    "plot_sensitivity",
    "plot_surface",
    "plot_surface_contour",
    "plot_trust",
    "plot_trust_region",
    "plot_variance",
    "plot_variance_overlay",
    "read_h5_grid",
    "read_h5_metadata",
    "read_h5_path",
    "read_h5_points",
    "read_h5_table",
    "save_plot",
]
