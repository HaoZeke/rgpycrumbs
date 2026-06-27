# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Test that every ensure_import call has a matching _DEPENDENCY_MAP entry.

This test ensures that auto-installation via RGPYCRUMBS_AUTO_DEPS=1
can resolve pip specs for all lazily imported modules. Without a map
entry, ensure_import falls through to a generic ImportError instead of
auto-installing the dependency.

Dependencies intentionally excluded from the map are tested separately.
ira_mod and tblite need pixi; ovito is kept out of the default auto-install
path because it is a heavy optional install.
"""

import ast
from pathlib import Path

import pytest

from rgpycrumbs._aux import _DEPENDENCY_MAP

pytestmark = pytest.mark.pure

PACKAGE_ROOT = Path(__file__).parent.parent / "rgpycrumbs"

# Modules intentionally kept out of _DEPENDENCY_MAP because they are
# pixi-only or heavy enough that we do not auto-install them by default.
EXCLUDED_FROM_MAP = {"ira_mod", "tblite", "ovito", "eon.config"}


def _find_ensure_import_args(root: Path) -> set[str]:
    """Walk all .py files under root and extract ensure_import("X") arguments."""
    modules = set()
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ensure_import"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                modules.add(node.args[0].value)
    return modules


class TestDependencyMapCoverage:
    """Verify _DEPENDENCY_MAP covers all ensure_import call sites."""

    def test_all_ensure_imports_have_map_entries(self) -> None:
        """Every ensure_import("X") call must have X (or its top-level
        package) in _DEPENDENCY_MAP, unless it is a special-case dependency."""
        used = _find_ensure_import_args(PACKAGE_ROOT)
        unmapped = set()
        for mod in used:
            top_level = mod.split(".")[0]
            if top_level in EXCLUDED_FROM_MAP:
                continue
            if mod not in _DEPENDENCY_MAP and top_level not in _DEPENDENCY_MAP:
                unmapped.add(mod)
        assert not unmapped, (
            f"ensure_import calls without _DEPENDENCY_MAP entry: {unmapped}. "
            f"Add entries so auto-install (RGPYCRUMBS_AUTO_DEPS=1) works."
        )

    def test_map_entries_have_valid_pip_specs(self) -> None:
        """Every _DEPENDENCY_MAP entry must have a non-empty pip spec
        and extra name."""
        for mod, (pip_spec, extra) in _DEPENDENCY_MAP.items():
            assert pip_spec, f"{mod}: empty pip spec"
            assert extra, f"{mod}: empty extra name"

    def test_no_duplicate_top_level_with_submodule(self) -> None:
        """If both 'foo' and 'foo.bar' are in the map, they should
        resolve to the same pip package."""
        top_levels: dict[str, str] = {}
        for mod, (pip_spec, _) in _DEPENDENCY_MAP.items():
            top = mod.split(".")[0]
            pkg = pip_spec.split(">=")[0].split("[")[0].strip()
            if top in top_levels:
                assert top_levels[top] == pkg, (
                    f"Inconsistent pip packages for {top}: {top_levels[top]} vs {pkg}"
                )
            else:
                top_levels[top] = pkg


class TestEnsureImportSmoke:
    """Smoke tests for ensure_import with actually available packages."""

    def test_import_numpy(self) -> None:
        """numpy is always available in the test env."""
        from rgpycrumbs._aux import ensure_import

        np = ensure_import("numpy")
        assert hasattr(np, "array")

    def test_import_submodule(self) -> None:
        """Submodule imports should return the submodule, not the parent."""
        from rgpycrumbs._aux import ensure_import

        interp = ensure_import("scipy.interpolate")
        assert hasattr(interp, "CubicSpline")

    def test_missing_module_raises(self) -> None:
        """A module not installed and not in map should raise ImportError."""
        from rgpycrumbs._aux import ensure_import

        with pytest.raises(ImportError, match="not_a_real_package"):
            ensure_import("not_a_real_package")
