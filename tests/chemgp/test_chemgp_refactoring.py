# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for chemgp plotting and I/O functions."""

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")

pytestmark = pytest.mark.pure


class TestHDF5IO:
    """Test HDF5 I/O functions."""

    @pytest.fixture
    def sample_h5_file(self, tmp_path: Path) -> Path:
        """Create a sample HDF5 file for testing."""
        h5_path = tmp_path / "test.h5"

        with h5py.File(h5_path, "w") as f:
            table_grp = f.create_group("table")
            table_grp.create_dataset("step", data=np.array([1, 2, 3, 4, 5]))
            table_grp.create_dataset("energy", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
            table_grp.create_dataset("force", data=np.array([0.5, 0.4, 0.3, 0.2, 0.1]))

            grid_grp = f.create_group("grids")
            energy_ds = grid_grp.create_dataset("energy", data=np.random.rand(10, 10))
            energy_ds.attrs["x_range"] = [0.0, 1.0]
            energy_ds.attrs["y_range"] = [0.0, 1.0]
            energy_ds.attrs["x_length"] = 10
            energy_ds.attrs["y_length"] = 10

            path_grp = f.create_group("paths")
            path_grp.create_dataset("path1_x", data=np.array([0.1, 0.2, 0.3]))
            path_grp.create_dataset("path1_y", data=np.array([0.4, 0.5, 0.6]))

            points_grp = f.create_group("points")
            points_grp.create_dataset("train_x", data=np.array([0.2, 0.4, 0.6]))
            points_grp.create_dataset("train_y", data=np.array([0.3, 0.5, 0.7]))

            f.attrs["conv_tol"] = 0.01
            f.attrs["n_steps"] = 100

        return h5_path

    def test_read_h5_table(self, sample_h5_file: Path) -> None:
        """Test reading table from HDF5."""
        from rgpycrumbs.chemgp.hdf5_io import read_h5_table

        with h5py.File(sample_h5_file, "r") as f:
            df = read_h5_table(f, "table")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "step" in df.columns
        assert "energy" in df.columns
        assert "force" in df.columns
        assert df["step"].iloc[0] == 1
        assert df["energy"].iloc[-1] == 0.5

    def test_read_h5_grid_with_coords(self, sample_h5_file: Path) -> None:
        """Test reading grid with axis coordinates."""
        from rgpycrumbs.chemgp.hdf5_io import read_h5_grid

        with h5py.File(sample_h5_file, "r") as f:
            data, x_coords, y_coords = read_h5_grid(f, "energy")

        assert data.shape == (10, 10)
        assert x_coords is not None
        assert y_coords is not None
        assert len(x_coords) == 10
        assert len(y_coords) == 10
        assert x_coords[0] == 0.0
        assert x_coords[-1] == 1.0

    def test_read_h5_path(self, sample_h5_file: Path) -> None:
        """Test reading path from HDF5."""
        from rgpycrumbs.chemgp.hdf5_io import read_h5_path

        with h5py.File(sample_h5_file, "r") as f:
            path_data = read_h5_path(f, "path1")

        assert isinstance(path_data, dict)
        assert "path1_x" in path_data
        assert "path1_y" in path_data
        assert len(path_data["path1_x"]) == 3

    def test_read_h5_points(self, sample_h5_file: Path) -> None:
        """Test reading points from HDF5."""
        from rgpycrumbs.chemgp.hdf5_io import read_h5_points

        with h5py.File(sample_h5_file, "r") as f:
            points_data = read_h5_points(f, "train")

        assert isinstance(points_data, dict)
        assert "train_x" in points_data
        assert "train_y" in points_data
        assert len(points_data["train_x"]) == 3

    def test_read_h5_metadata(self, sample_h5_file: Path) -> None:
        """Test reading metadata from HDF5."""
        from rgpycrumbs.chemgp.hdf5_io import read_h5_metadata

        with h5py.File(sample_h5_file, "r") as f:
            metadata = read_h5_metadata(f)

        assert isinstance(metadata, dict)
        assert "conv_tol" in metadata
        assert "n_steps" in metadata
        assert metadata["conv_tol"] == 0.01
        assert metadata["n_steps"] == 100


class TestPlottingFunctions:
    """Test plotting utility functions."""

    def test_detect_clamp_mb(self) -> None:
        """Test clamp detection for Muller-Brown."""
        from rgpycrumbs.chemgp.plotting import detect_clamp

        clamp_lo, clamp_hi, step = detect_clamp("mb_surface.h5")

        assert clamp_lo == -200.0
        assert clamp_hi == 50.0
        assert step == 25.0

    def test_detect_clamp_leps(self) -> None:
        """Test clamp detection for LEPS."""
        from rgpycrumbs.chemgp.plotting import detect_clamp

        clamp_lo, clamp_hi, step = detect_clamp("leps_potential.h5")

        assert clamp_lo == -5.0
        assert clamp_hi == 5.0
        assert step == 0.5

    def test_detect_clamp_unknown(self) -> None:
        """Test clamp detection for unknown filename."""
        from rgpycrumbs.chemgp.plotting import detect_clamp

        clamp_lo, clamp_hi, step = detect_clamp("unknown.h5")

        assert clamp_lo is None
        assert clamp_hi is None
        assert step is None

    def test_save_plot_matplotlib(self, tmp_path: Path) -> None:
        """Test saving matplotlib figure."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from rgpycrumbs.chemgp.plotting import save_plot

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        output_path = tmp_path / "test.pdf"
        save_plot(fig, output_path, dpi=300)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)


class TestCLIRegistration:
    """Test that CLI commands are properly registered."""

    def test_plot_gp_commands_registered(self) -> None:
        """Test that chemgp CLI commands are registered."""
        click = pytest.importorskip("click")
        from click.testing import CliRunner

        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "convergence" in result.output
        assert "surface" in result.output
        assert "quality" in result.output
        assert "rff" in result.output
        assert "nll" in result.output
        assert "sensitivity" in result.output
        assert "trust" in result.output
        assert "variance" in result.output
        assert "fps" in result.output
        assert "profile" in result.output

    def test_cli_command_help(self) -> None:
        """Test that individual CLI commands have help."""
        click = pytest.importorskip("click")
        from click.testing import CliRunner

        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["surface", "--help"])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--clamp-lo" in result.output
        assert "--clamp-hi" in result.output


class TestModuleImports:
    """Test that module imports work correctly."""

    def test_chemgp_init_exports(self) -> None:
        """Test that chemgp __init__.py exports all expected functions."""
        from rgpycrumbs import chemgp

        assert hasattr(chemgp, "__all__")
        assert isinstance(chemgp.__all__, list)

        for name in chemgp.__all__:
            assert hasattr(chemgp, name), f"Missing export: {name}"

    def test_hdf5_io_imports(self) -> None:
        """Test that all HDF5 I/O functions can be imported."""
        from rgpycrumbs.chemgp import (
            read_h5_grid,
            read_h5_metadata,
            read_h5_path,
            read_h5_points,
            read_h5_table,
        )

        assert callable(read_h5_grid)
        assert callable(read_h5_metadata)
        assert callable(read_h5_path)
        assert callable(read_h5_points)
        assert callable(read_h5_table)

    def test_plotting_imports(self) -> None:
        """Test that all plotting functions can be imported."""
        from rgpycrumbs.chemgp import (
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

        assert callable(detect_clamp)
        assert callable(plot_convergence)
        assert callable(plot_fps)
        assert callable(plot_gp_quality)
        assert callable(plot_nll)
        assert callable(plot_profile)
        assert callable(plot_rff)
        assert callable(plot_sensitivity)
        assert callable(plot_surface)
        assert callable(plot_trust)
        assert callable(plot_variance)
        assert callable(save_plot)
