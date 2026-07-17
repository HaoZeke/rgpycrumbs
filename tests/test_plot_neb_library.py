# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Library API for eOn NEB plots (no Click argv)."""

from __future__ import annotations

import pytest

from tests._optional_imports import has_module_spec

pytestmark = pytest.mark.pure


def test_plot_neb_merge_settings():
    """Library path is merge_plot_settings + plot_neb_from_settings (no argv)."""
    from rgpycrumbs.eon.plot_config import merge_plot_settings

    s = merge_plot_settings(
        "neb",
        cli_overrides={
            "plot_type": "landscape",
            "con_file": "neb.con",
            "output_file": "out.png",
            "surface_type": "grad_imq",
            "landscape_path": "all",
            "project_path": True,
            "title": "NEB-RMSD Surface",
            "figsize": (12, 8),
            "plot_structures": "all",
            "strip_renderer": "xyzrender",
        },
    )
    assert s["plot_type"] == "landscape"
    assert s["surface_type"] == "grad_imq"
    assert s["landscape_path"] == "all"
    assert s["project_path"] is True
    assert s["title"] == "NEB-RMSD Surface"
    assert s["plot_structures"] == "all"


@pytest.mark.skipif(
    not all(has_module_spec(m) for m in ("polars", "chemparseplot", "click")),
    reason="plot_neb import needs plot stack",
)
def test_plot_neb_importable():
    from rgpycrumbs.eon import plot_neb, plot_neb_from_settings

    assert callable(plot_neb)
    assert callable(plot_neb_from_settings)
