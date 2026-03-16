# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""ChemGP plotting and I/O utilities.

Parsing and plotting functions live in chemparseplot. This module
re-exports them for backward compatibility. The CLI entry point
is plot_gp.py (PEP 723 script dispatched by rgpycrumbs.cli).
"""

from chemparseplot.parse.chemgp_hdf5 import (
    read_h5_grid,
    read_h5_metadata,
    read_h5_path,
    read_h5_points,
    read_h5_table,
)
from chemparseplot.plot.chemgp import (
    detect_clamp,
    plot_convergence,
    plot_fps,
    plot_gp_quality,
    plot_nll,
    plot_profile,
    plot_rff,
    plot_sensitivity,
    plot_surface,
    plot_trust,
    plot_variance,
    save_plot,
)

__all__ = [
    "read_h5_grid",
    "read_h5_metadata",
    "read_h5_path",
    "read_h5_points",
    "read_h5_table",
    "detect_clamp",
    "plot_convergence",
    "plot_fps",
    "plot_gp_quality",
    "plot_nll",
    "plot_profile",
    "plot_rff",
    "plot_sensitivity",
    "plot_surface",
    "plot_trust",
    "plot_variance",
    "save_plot",
]
