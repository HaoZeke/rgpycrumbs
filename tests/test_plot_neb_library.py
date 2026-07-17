# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Library API for eOn plot entry points (no Click argv)."""

from __future__ import annotations

import pytest

from tests._optional_imports import has_module_spec

pytestmark = pytest.mark.pure


def test_run_plot_dispatches_to_runner():
    from rgpycrumbs.eon.plot_config import library_plot, run_plot

    seen: list[dict] = []

    def runner(settings):
        seen.append(settings)
        return settings["plot_type"]

    assert run_plot("min", runner, plot_type="landscape") == "landscape"
    assert seen[0]["plot_type"] == "landscape"
    assert seen[0]["prefix"] == "minimization"

    api = library_plot("saddle", runner)
    assert api.__name__ == "plot_saddle"
    assert api(plot_type="profile") == "profile"
    assert seen[1]["plot_type"] == "profile"


def test_plot_min_saddle_merge_settings():
    from rgpycrumbs.eon.plot_config import merge_plot_settings

    smin = merge_plot_settings(
        "min",
        cli_overrides={"job_dir": "minimization_run", "plot_type": "landscape"},
    )
    assert smin["plot_type"] == "landscape"
    ssad = merge_plot_settings(
        "saddle",
        cli_overrides={"job_dir": "saddle_run", "plot_type": "profile"},
    )
    assert ssad["plot_type"] == "profile"


def test_plot_neb_merge_settings():
    """Library path is merge_plot_settings + runner (no argv)."""
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
    import types

    from rgpycrumbs.eon import plot_neb, plot_neb_from_settings

    assert callable(plot_neb)
    assert callable(plot_neb_from_settings)
    # from_settings must be a plain function (not a Click Command)
    assert isinstance(plot_neb_from_settings, types.FunctionType)
