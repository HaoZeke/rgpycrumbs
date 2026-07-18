# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""eOn tools: CLI scripts and library plot entry points.

::

    from rgpycrumbs.eon import plot, plot_neb, plot_min, plot_saddle

    plot(neb, plot_type="profile", output_file="1D.png")  # after neb.compute()
    plot(frames, kind="neb", plot_type="profile", output_file="1D.png")
    plot_neb(plot_type="profile", con_file="neb.con", output_file="1D.png")
    plot_min(job_dir="minimization_run", plot_type="landscape", output="min.png")
    plot_saddle(job_dir="saddle_run", plot_type="profile", output="sad.png")

CLI: ``rgpycrumbs eon plt-{neb,min,saddle}`` (same pipelines).
"""

from __future__ import annotations

from typing import Any

_COMMANDS = ("neb", "min", "saddle")
__all__ = [
    "adapt_plot_source",
    "plot",
    *[name for cmd in _COMMANDS for name in (f"plot_{cmd}", f"plot_{cmd}_from_settings")],
]


def __getattr__(name: str) -> Any:
    if name in {"plot", "adapt_plot_source"}:
        from rgpycrumbs.eon.plot_dispatch import adapt_plot_source, plot

        if name == "plot":
            return plot
        return adapt_plot_source
    for cmd in _COMMANDS:
        if name in {f"plot_{cmd}", f"plot_{cmd}_from_settings"}:
            import importlib

            mod = importlib.import_module(f"rgpycrumbs.eon.plt_{cmd}")
            return getattr(mod, name)
    raise AttributeError(name)
