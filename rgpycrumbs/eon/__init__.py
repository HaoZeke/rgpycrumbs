# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""eOn tools: CLI scripts and library plot entry points.

Library (no Click argv)::

    from rgpycrumbs.eon import plot_neb, plot_min, plot_saddle

    plot_neb(plot_type="profile", con_file="neb.con", output_file="1D.png")
    plot_min(job_dir="minimization_run", plot_type="landscape", output="min.png")
    plot_saddle(job_dir="saddle_run", plot_type="profile", output="sad.png")

CLI remains ``rgpycrumbs eon plt-{neb,min,saddle}`` (same pipelines).

.. versionadded:: 1.7.0
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "plot_min",
    "plot_min_from_settings",
    "plot_neb",
    "plot_neb_from_settings",
    "plot_saddle",
    "plot_saddle_from_settings",
]

_LAZY = {
    "plot_neb": ("rgpycrumbs.eon.plt_neb", "plot_neb"),
    "plot_neb_from_settings": ("rgpycrumbs.eon.plt_neb", "plot_neb_from_settings"),
    "plot_min": ("rgpycrumbs.eon.plt_min", "plot_min"),
    "plot_min_from_settings": ("rgpycrumbs.eon.plt_min", "plot_min_from_settings"),
    "plot_saddle": ("rgpycrumbs.eon.plt_saddle", "plot_saddle"),
    "plot_saddle_from_settings": (
        "rgpycrumbs.eon.plt_saddle",
        "plot_saddle_from_settings",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(name)
