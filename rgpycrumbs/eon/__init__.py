# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""eOn tools: CLI scripts and library plot entry points.

Library (no Click argv)::

    from rgpycrumbs.eon import plot_neb

    plot_neb(plot_type="profile", con_file="neb.con", output_file="1D.png")

CLI remains ``rgpycrumbs eon plt-neb`` (same pipeline).

.. versionadded:: 1.7.0
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "plot_neb",
    "plot_neb_from_settings",
]


def __getattr__(name: str) -> Any:
    if name in {"plot_neb", "plot_neb_from_settings"}:
        from rgpycrumbs.eon.plt_neb import plot_neb, plot_neb_from_settings

        return plot_neb if name == "plot_neb" else plot_neb_from_settings
    raise AttributeError(name)
