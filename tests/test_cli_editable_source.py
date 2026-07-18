# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""_find_editable_source must not map site-packages wheels to monorepos."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rgpycrumbs.cli import (
    _find_editable_source,
    _normalize_dist_name,
    _pyproject_project_name,
)


def test_normalize_dist_name():
    assert _normalize_dist_name("eon_schema") == "eon-schema"
    assert _normalize_dist_name("ChemParsePlot") == "chemparseplot"


def test_pyproject_project_name(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "chemparseplot"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    assert _pyproject_project_name(tmp_path / "pyproject.toml") == "chemparseplot"


def test_site_packages_wheel_not_treated_as_editable(tmp_path: Path, monkeypatch):
    """Wheel under .../site-packages must not walk up into eon-akmc monorepo."""
    monorepo = tmp_path / "eOn-pyeon-final"
    monorepo.mkdir()
    (monorepo / "pyproject.toml").write_text(
        '[project]\nname = "eon-akmc"\nversion = "2.16.0"\n',
        encoding="utf-8",
    )
    site = (
        monorepo
        / ".pixi"
        / "envs"
        / "docs-mta"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "chemparseplot"
    )
    site.mkdir(parents=True)
    init = site / "__init__.py"
    init.write_text("# wheel\n", encoding="utf-8")

    fake_spec = SimpleNamespace(
        origin=str(init),
        submodule_search_locations=[str(site)],
    )
    monkeypatch.setattr(
        "rgpycrumbs.cli.importlib.util.find_spec",
        lambda name: fake_spec if name == "chemparseplot" else None,
    )
    assert _find_editable_source("chemparseplot") is None


def test_true_editable_src_layout_matched_by_name(tmp_path: Path, monkeypatch):
    root = tmp_path / "chemparseplot"
    (root / "src" / "chemparseplot").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "chemparseplot"\nversion = "1.9.0"\n',
        encoding="utf-8",
    )
    init = root / "src" / "chemparseplot" / "__init__.py"
    init.write_text("# editable\n", encoding="utf-8")

    fake_spec = SimpleNamespace(
        origin=str(init),
        submodule_search_locations=[str(init.parent)],
    )
    monkeypatch.setattr(
        "rgpycrumbs.cli.importlib.util.find_spec",
        lambda name: fake_spec if name == "chemparseplot" else None,
    )
    assert _find_editable_source("chemparseplot") == root


def test_wrong_pyproject_name_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "eon-akmc"\nversion = "2.16.0"\n',
        encoding="utf-8",
    )
    init = root / "pkg" / "__init__.py"
    init.write_text("# not chemparseplot\n", encoding="utf-8")
    fake_spec = SimpleNamespace(
        origin=str(init),
        submodule_search_locations=[str(init.parent)],
    )
    monkeypatch.setattr(
        "rgpycrumbs.cli.importlib.util.find_spec",
        lambda name: fake_spec if name == "chemparseplot" else None,
    )
    assert _find_editable_source("chemparseplot") is None
