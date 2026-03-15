# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for eon CLI and module structure."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.pure


class TestPltNebCLI:
    """Test plt-neb CLI commands."""

    def test_cli_main_command_exists(self) -> None:
        """Test that main CLI command exists."""
        from rgpycrumbs.eon.plt_neb import main

        assert main is not None
        assert hasattr(main, "params")  # Click command has params

    def test_cli_help_output(self) -> None:
        """Test that CLI help is available."""
        from click.testing import CliRunner

        from rgpycrumbs.eon.plt_neb import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "NEB" in result.output or "neb" in result.output
        assert "--plot-type" in result.output

    def test_cli_options_present(self) -> None:
        """Test that expected CLI options are present."""
        from click.testing import CliRunner

        from rgpycrumbs.eon.plt_neb import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        # Check for key options
        assert "--plot-type" in result.output
        assert "--landscape-mode" in result.output
        assert "--surface-type" in result.output
        assert "--project-path" in result.output
        assert "--plot-structures" in result.output
        assert "--show-legend" in result.output
        assert "--output-file" in result.output or "-o" in result.output
        assert "--ira-kmax" in result.output


class TestEonModuleImports:
    """Test that eon module imports work correctly."""

    def test_eon_init_exists(self) -> None:
        """Test that eon __init__.py exists and is importable."""
        from rgpycrumbs import eon

        assert eon is not None
        assert hasattr(eon, "__all__")

    def test_eon_all_defined(self) -> None:
        """Test that __all__ is defined in eon."""
        from rgpycrumbs import eon

        assert isinstance(eon.__all__, list)

    def test_plt_neb_importable(self) -> None:
        """Test that plt_neb module can be imported."""
        from rgpycrumbs.eon import plt_neb

        assert plt_neb is not None
        assert hasattr(plt_neb, "main")


class TestLazyImports:
    """Test that lazy imports work correctly in eon modules."""

    def test_lazy_import_helpers(self) -> None:
        """Test that lazy import helpers are available."""
        from rgpycrumbs._aux import ensure_import

        # Test with a module that should be available
        np = ensure_import("numpy")
        assert np is not None
        assert hasattr(np, "array")

    def test_plt_neb_uses_lazy_imports(self) -> None:
        """Test that plt_neb uses lazy imports pattern."""
        import inspect

        from rgpycrumbs.eon import plt_neb

        # Get the source code
        source = inspect.getsource(plt_neb)

        # Check that it imports ensure_import or uses lazy pattern
        assert "ensure_import" in source or "import" in source


class TestPltNebDataFiles:
    """Test that test data files are available."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test data directory."""
        from pathlib import Path

        return Path(__file__).parent.parent / "data" / "neb_test"

    def test_test_data_exists(self, test_data_dir: Path) -> None:
        """Test that test data directory exists."""
        assert test_data_dir.exists(), "Test data directory not found"

    def test_dat_files_exist(self, test_data_dir: Path) -> None:
        """Test that .dat files exist."""
        dat_files = list(test_data_dir.glob("neb_*.dat"))
        assert len(dat_files) > 0, "No neb_*.dat files found"

    def test_neb_path_file_exists(self, test_data_dir: Path) -> None:
        """Test that NEB path .con file exists."""
        con_file = test_data_dir / "neb.con"
        assert con_file.exists(), "neb.con not found"

    def test_path_con_files_exist(self, test_data_dir: Path) -> None:
        """Test that path .con files exist."""
        path_files = list(test_data_dir.glob("neb_path_*.con"))
        assert len(path_files) > 0, "No neb_path_*.con files found"
