# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""ChemGP plotting and I/O utilities.

This module provides functions for reading ChemGP HDF5 data and generating
publication-quality plots.

.. versionadded:: 1.7.0
    Refactored from monolithic plt_gp.py to modular structure.
"""

from rgpycrumbs.chemgp.hdf5_io import (
    read_h5_grid,
    read_h5_metadata,
    read_h5_path,
    read_h5_points,
    read_h5_table,
)
from rgpycrumbs.chemgp.plotting import (
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
    # Utilities
    "detect_clamp",
    # Plotting functions
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
    "read_h5_grid",
    "read_h5_metadata",
    "read_h5_path",
    "read_h5_points",
    # HDF5 I/O
    "read_h5_table",
    "save_plot",
]
