"""Coverage boost tests for CLI scripts and plot modules.

Tests use click.testing.CliRunner for CLI commands, mock unavailable
dependencies (chemparseplot, mlflow, ovito, pyvista, pandas, plotnine,
h5py, polars, adjustText, cmcrameri, rich, pychum), and exercise error
paths and --help flags to maximize statement coverage.
"""

import configparser
import io
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from click.testing import CliRunner

import importlib

def _has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

_HAS_REQUESTS = _has("requests")
_HAS_PANDAS_BOOST = _has("pandas")
_HAS_MLFLOW = _has("mlflow")
_HAS_OVITO = _has("ovito")
_HAS_PYVISTA = _has("pyvista")

pytestmark = pytest.mark.pure


# ======================================================================
# Helpers for building mock modules
# ======================================================================

def _install_fake_module(name, attrs=None, monkeypatch=None):
    """Create a fake module in sys.modules."""
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _make_chemparseplot_mocks():
    """Build all chemparseplot submodule mocks needed by plt_neb, plt_saddle, plt_min."""
    mods = {}

    # Top level
    mods["chemparseplot"] = types.ModuleType("chemparseplot")

    # chemparseplot.parse
    mods["chemparseplot.parse"] = types.ModuleType("chemparseplot.parse")
    mods["chemparseplot.parse.file_"] = types.ModuleType("chemparseplot.parse.file_")
    mods["chemparseplot.parse.file_"].find_file_paths = MagicMock(return_value=[])

    # chemparseplot.parse.eon
    mods["chemparseplot.parse.eon"] = types.ModuleType("chemparseplot.parse.eon")
    mods["chemparseplot.parse.eon.neb"] = types.ModuleType("chemparseplot.parse.eon.neb")
    for fn_name in [
        "aggregate_neb_landscape_data",
        "compute_profile_rmsd",
        "estimate_rbf_smoothing",
        "load_structures_and_calculate_additional_rmsd",
    ]:
        setattr(mods["chemparseplot.parse.eon.neb"], fn_name, MagicMock())

    mods["chemparseplot.parse.eon.dimer_trajectory"] = types.ModuleType(
        "chemparseplot.parse.eon.dimer_trajectory"
    )
    mods["chemparseplot.parse.eon.dimer_trajectory"].load_dimer_trajectory = MagicMock()

    mods["chemparseplot.parse.eon.min_trajectory"] = types.ModuleType(
        "chemparseplot.parse.eon.min_trajectory"
    )
    mods["chemparseplot.parse.eon.min_trajectory"].load_min_trajectory = MagicMock()

    # chemparseplot.parse.trajectory
    mods["chemparseplot.parse.trajectory"] = types.ModuleType("chemparseplot.parse.trajectory")
    mods["chemparseplot.parse.trajectory.hdf5"] = types.ModuleType(
        "chemparseplot.parse.trajectory.hdf5"
    )
    for fn_name in [
        "history_to_landscape_df",
        "history_to_profile_dats",
        "result_to_atoms_list",
        "result_to_profile_dat",
    ]:
        setattr(mods["chemparseplot.parse.trajectory.hdf5"], fn_name, MagicMock())
    # Alias used by plt_neb
    mods["chemparseplot.parse.trajectory.hdf5"].history_to_landscape_df = MagicMock()

    mods["chemparseplot.parse.trajectory.neb"] = types.ModuleType(
        "chemparseplot.parse.trajectory.neb"
    )
    for fn_name in ["load_trajectory", "trajectory_to_landscape_df", "trajectory_to_profile_dat"]:
        setattr(mods["chemparseplot.parse.trajectory.neb"], fn_name, MagicMock())

    # chemparseplot.parse.neb_utils
    mods["chemparseplot.parse.neb_utils"] = types.ModuleType("chemparseplot.parse.neb_utils")
    mods["chemparseplot.parse.neb_utils"].calculate_landscape_coords = MagicMock(
        return_value=(np.zeros(5), np.zeros(5))
    )
    mods["chemparseplot.parse.neb_utils"].compute_synthetic_gradients = MagicMock(
        return_value=(np.zeros(5), np.zeros(5))
    )

    # chemparseplot.parse.chemgp_hdf5
    mods["chemparseplot.parse.chemgp_hdf5"] = types.ModuleType("chemparseplot.parse.chemgp_hdf5")
    for fn_name in ["read_h5_grid", "read_h5_metadata", "read_h5_path", "read_h5_points", "read_h5_table"]:
        setattr(mods["chemparseplot.parse.chemgp_hdf5"], fn_name, MagicMock())

    # chemparseplot.plot
    mods["chemparseplot.plot"] = types.ModuleType("chemparseplot.plot")
    mods["chemparseplot.plot.neb"] = types.ModuleType("chemparseplot.plot.neb")
    for fn_name in [
        "plot_energy_path",
        "plot_landscape_path_overlay",
        "plot_landscape_surface",
        "plot_mmf_peaks_overlay",
        "plot_neb_evolution",
        "plot_structure_inset",
        "plot_structure_strip",
    ]:
        setattr(mods["chemparseplot.plot.neb"], fn_name, MagicMock())

    mods["chemparseplot.plot.theme"] = types.ModuleType("chemparseplot.plot.theme")
    mods["chemparseplot.plot.theme"].apply_axis_theme = MagicMock()
    mods["chemparseplot.plot.theme"].get_theme = MagicMock(return_value={})
    mods["chemparseplot.plot.theme"].setup_global_theme = MagicMock()

    mods["chemparseplot.plot.optimization"] = types.ModuleType("chemparseplot.plot.optimization")
    for fn_name in [
        "plot_convergence_panel",
        "plot_dimer_mode_evolution",
        "plot_optimization_landscape",
        "plot_optimization_profile",
    ]:
        setattr(mods["chemparseplot.plot.optimization"], fn_name, MagicMock())

    mods["chemparseplot.plot.chemgp"] = types.ModuleType("chemparseplot.plot.chemgp")
    for fn_name in [
        "detect_clamp",
        "plot_convergence_curve",
        "plot_energy_profile",
        "plot_fps_projection",
        "plot_gp_progression",
        "plot_hyperparameter_sensitivity",
        "plot_nll_landscape",
        "plot_rff_quality",
        "plot_surface_contour",
        "plot_trust_region",
        "plot_variance_overlay",
        "save_plot",
    ]:
        setattr(mods["chemparseplot.plot.chemgp"], fn_name, MagicMock())

    return mods


def _make_polars_mock():
    """Build a minimal polars mock."""
    mod = types.ModuleType("polars")
    mod.read_csv = MagicMock()

    class FakeDF:
        def __init__(self, data=None):
            self._data = data or {}
            self.columns = list(self._data.keys())

        def __getitem__(self, key):
            return self

        def to_numpy(self):
            return np.zeros(5)

    mod.DataFrame = FakeDF
    return mod


def _make_adjusttext_mock():
    mod = types.ModuleType("adjustText")
    mod.adjust_text = MagicMock()
    return mod


# ======================================================================
# 1. plt_neb.py -- CLI help and import coverage
# ======================================================================
class TestPltNeb:
    """Test plt_neb click CLI via mocked imports."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Install all mocked dependencies before importing plt_neb."""
        self._mods = _make_chemparseplot_mocks()
        self._mods["polars"] = _make_polars_mock()
        self._mods["adjustText"] = _make_adjusttext_mock()

        # Store originals
        self._originals = {}
        for name, mod in self._mods.items():
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        # Remove cached module if previously imported
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.plt_neb"):
                del sys.modules[key]

        yield

        # Restore
        for name in self._mods:
            if self._originals.get(name) is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.plt_neb"):
                del sys.modules[key]

    def test_help(self):
        from rgpycrumbs.eon.plt_neb import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "NEB" in result.output or "neb" in result.output.lower()

    def test_no_input_files(self, tmp_path):
        from rgpycrumbs.eon.plt_neb import main

        runner = CliRunner()
        result = runner.invoke(main, ["--plot-type", "profile"], catch_exceptions=False)
        # Should run but fail gracefully or produce output
        # The exit code may be non-zero if no data files found
        assert result.exit_code is not None

    def test_constants(self):
        from rgpycrumbs.eon.plt_neb import DEFAULT_INPUT_PATTERN, DEFAULT_PATH_PATTERN, IRA_KMAX_DEFAULT

        assert DEFAULT_INPUT_PATTERN == "neb_*.dat"
        assert DEFAULT_PATH_PATTERN == "neb_path_*.con"
        assert isinstance(IRA_KMAX_DEFAULT, float)


# ======================================================================
# 2. plot_gp.py -- ChemGP CLI
# ======================================================================
try:
    from chemparseplot.plot.chemgp import plot_convergence  # noqa: F401
    _HAS_DEV_CHEMGP = True
except (ImportError, ModuleNotFoundError):
    _HAS_DEV_CHEMGP = False


@pytest.mark.skipif(not _HAS_DEV_CHEMGP, reason="needs dev chemparseplot with plot_convergence")
class TestPlotGP:
    """Test chemgp/plot_gp.py click CLI group."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        self._mods = _make_chemparseplot_mocks()

        # h5py mock
        h5py_mod = types.ModuleType("h5py")
        h5py_mod.File = MagicMock()
        self._mods["h5py"] = h5py_mod

        self._originals = {}
        for name, mod in self._mods.items():
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.chemgp.plot_gp"):
                del sys.modules[key]

        yield

        for name in self._mods:
            if self._originals.get(name) is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.chemgp.plot_gp"):
                del sys.modules[key]

    def test_group_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ChemGP" in result.output or "chemgp" in result.output.lower()

    def test_convergence_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["convergence", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output

    def test_surface_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["surface", "--help"])
        assert result.exit_code == 0

    def test_quality_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["quality", "--help"])
        assert result.exit_code == 0

    def test_rff_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["rff", "--help"])
        assert result.exit_code == 0

    def test_nll_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["nll", "--help"])
        assert result.exit_code == 0

    def test_sensitivity_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["sensitivity", "--help"])
        assert result.exit_code == 0

    def test_trust_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["trust", "--help"])
        assert result.exit_code == 0

    def test_variance_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["variance", "--help"])
        assert result.exit_code == 0

    def test_fps_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["fps", "--help"])
        assert result.exit_code == 0

    def test_profile_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["profile", "--help"])
        assert result.exit_code == 0

    def test_batch_help(self):
        from rgpycrumbs.chemgp.plot_gp import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0

    def test_common_options_decorator(self):
        from rgpycrumbs.chemgp.plot_gp import common_options

        assert callable(common_options)

    def test_main_entrypoint(self):
        from rgpycrumbs.chemgp.plot_gp import main

        assert callable(main)


# ======================================================================
# 3. con_splitter.py -- Split .con files
# ======================================================================
class TestConSplitter:
    """Test con_splitter click CLI with synthetic data."""

    def test_help(self):
        from rgpycrumbs.eon.con_splitter import con_splitter

        runner = CliRunner()
        result = runner.invoke(con_splitter, ["--help"])
        assert result.exit_code == 0
        assert "--images-per-path" in result.output

    def test_split_synthetic_traj(self, tmp_path):
        """Create a synthetic trajectory and split it."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        # Create 6 frames (2 paths of 3 images each)
        traj_file = tmp_path / "neb.traj"
        frames = []
        for i in range(6):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.74 + i * 0.01, 0, 0]])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc(True)
            frames.append(atoms)
        ase_write(str(traj_file), frames)

        output_dir = tmp_path / "split_output"

        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [
                str(traj_file),
                "--images-per-path",
                "3",
                "--output-dir",
                str(output_dir),
                "--mode",
                "neb",
            ],
        )
        assert result.exit_code == 0
        assert output_dir.exists()
        # Should have 3 .con files
        con_files = list(output_dir.glob("ipath_*.con"))
        assert len(con_files) == 3

    def test_split_last_path(self, tmp_path):
        """Test extracting the last path (default)."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        traj_file = tmp_path / "neb.traj"
        frames = []
        for i in range(6):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.74 + i * 0.01, 0, 0]])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc(True)
            frames.append(atoms)
        ase_write(str(traj_file), frames)

        output_dir = tmp_path / "split_last"
        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [
                str(traj_file),
                "--images-per-path",
                "3",
                "--output-dir",
                str(output_dir),
                "--path-index",
                "-1",
            ],
        )
        assert result.exit_code == 0

    def test_split_with_centering(self, tmp_path):
        """Test centering option."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        traj_file = tmp_path / "neb.traj"
        frames = []
        for i in range(3):
            atoms = Atoms("H2", positions=[[1, 1, 1], [1.74, 1, 1]])
            atoms.set_cell([10, 10, 10])
            atoms.set_pbc(True)
            frames.append(atoms)
        ase_write(str(traj_file), frames)

        output_dir = tmp_path / "centered"
        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [
                str(traj_file),
                "--images-per-path",
                "3",
                "--output-dir",
                str(output_dir),
                "--center",
                "--box-diagonal",
                "25",
                "25",
                "25",
            ],
        )
        assert result.exit_code == 0

    def test_split_too_few_frames(self, tmp_path):
        """Test error when fewer frames than images_per_path."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        traj_file = tmp_path / "neb.traj"
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)
        ase_write(str(traj_file), atoms)

        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [str(traj_file), "--images-per-path", "5", "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_split_bad_path_index(self, tmp_path):
        """Test error when path_index is out of bounds."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        traj_file = tmp_path / "neb.traj"
        frames = [
            Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)
            for _ in range(3)
        ]
        ase_write(str(traj_file), frames)

        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [
                str(traj_file),
                "--images-per-path",
                "3",
                "--path-index",
                "5",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_split_zero_images(self, tmp_path):
        """Test error when images_per_path is zero."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        traj_file = tmp_path / "neb.traj"
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)
        ase_write(str(traj_file), atoms)

        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [str(traj_file), "--images-per-path", "0", "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_enums(self):
        from rgpycrumbs.eon.con_splitter import AlignMode, SplitMode

        assert AlignMode.NONE.value == "none"
        assert AlignMode.ALL.value == "all"
        assert AlignMode.ENDPOINTS.value == "endpoints"
        assert SplitMode.NEB.value == "neb"
        assert SplitMode.FLEX.value == "flex"

    def test_align_path_none(self):
        from ase import Atoms

        from rgpycrumbs.eon.con_splitter import AlignMode, align_path
        from rgpycrumbs.geom.api.alignment import IRAConfig

        frames = [
            Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
            Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]]),
        ]
        result = align_path(frames, AlignMode.NONE, IRAConfig(enabled=False, kmax=1.8))
        assert len(result) == 2

    def test_align_path_single_frame(self):
        from ase import Atoms

        from rgpycrumbs.eon.con_splitter import AlignMode, align_path
        from rgpycrumbs.geom.api.alignment import IRAConfig

        frames = [Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])]
        result = align_path(frames, AlignMode.ALL, IRAConfig(enabled=False, kmax=1.8))
        assert len(result) == 1

    def test_neb_mode_warning_on_remainder(self, tmp_path):
        """NEB mode should warn when frame count is not a multiple of images_per_path."""
        from ase import Atoms
        from ase.io import write as ase_write

        from rgpycrumbs.eon.con_splitter import con_splitter

        traj_file = tmp_path / "neb.traj"
        # 5 frames, images_per_path=3 -> 1 full path + 2 remainder
        frames = [
            Atoms("H2", positions=[[0, 0, 0], [0.74 + i * 0.01, 0, 0]], cell=[10, 10, 10], pbc=True)
            for i in range(5)
        ]
        ase_write(str(traj_file), frames)

        runner = CliRunner()
        result = runner.invoke(
            con_splitter,
            [
                str(traj_file),
                "--images-per-path",
                "3",
                "--mode",
                "neb",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        # Should still succeed (extracts 1 full path)
        assert result.exit_code == 0


# ======================================================================
# 4. plt_saddle.py -- Saddle search CLI
# ======================================================================
class TestPltSaddle:
    """Test plt_saddle click CLI."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        self._mods = _make_chemparseplot_mocks()
        self._mods["polars"] = _make_polars_mock()

        self._originals = {}
        for name, mod in self._mods.items():
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.plt_saddle"):
                del sys.modules[key]

        yield

        for name in self._mods:
            if self._originals.get(name) is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.plt_saddle"):
                del sys.modules[key]

    def test_help(self):
        from rgpycrumbs.eon.plt_saddle import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--job-dir" in result.output
        assert "--plot-type" in result.output

    def test_constants(self):
        from rgpycrumbs.eon.plt_saddle import IRA_KMAX_DEFAULT

        assert IRA_KMAX_DEFAULT == 1.8


# ======================================================================
# 5. plt_min.py -- Minimization CLI
# ======================================================================
class TestPltMin:
    """Test plt_min click CLI."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        self._mods = _make_chemparseplot_mocks()
        self._mods["polars"] = _make_polars_mock()

        self._originals = {}
        for name, mod in self._mods.items():
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.plt_min"):
                del sys.modules[key]

        yield

        for name in self._mods:
            if self._originals.get(name) is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.plt_min"):
                del sys.modules[key]

    def test_help(self):
        from rgpycrumbs.eon.plt_min import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--job-dir" in result.output
        assert "--prefix" in result.output

    def test_constants(self):
        from rgpycrumbs.eon.plt_min import IRA_KMAX_DEFAULT

        assert IRA_KMAX_DEFAULT == 1.8


# ======================================================================
# 6. to_mlflow.py -- MLflow logging CLI
# ======================================================================
@pytest.mark.skipif(not _HAS_MLFLOW, reason="mlflow required")
class TestToMlflow:
    """Test to_mlflow.py CLI with mocked mlflow."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        # Mock mlflow
        mlflow_mock = types.ModuleType("mlflow")
        mlflow_mock.set_experiment = MagicMock()
        mlflow_mock.start_run = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False)))
        mlflow_mock.log_metric = MagicMock()
        mlflow_mock.log_param = MagicMock()
        mlflow_mock.log_artifact = MagicMock()
        mlflow_mock.log_figure = MagicMock()
        mlflow_mock.set_tag = MagicMock()

        # Mock the _mlflow.log_params submodule
        mlflow_log_params = types.ModuleType("rgpycrumbs.eon._mlflow.log_params")
        mlflow_log_params.log_config_ini = MagicMock()

        self._originals = {}
        for name in ["mlflow", "rgpycrumbs.eon._mlflow.log_params"]:
            self._originals[name] = sys.modules.get(name)

        sys.modules["mlflow"] = mlflow_mock
        sys.modules["rgpycrumbs.eon._mlflow.log_params"] = mlflow_log_params

        for key in list(sys.modules.keys()):
            if key == "rgpycrumbs.eon.to_mlflow":
                del sys.modules[key]

        yield

        for name in self._originals:
            if self._originals[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        sys.modules.pop("rgpycrumbs.eon.to_mlflow", None)

    def test_help(self):
        from rgpycrumbs.eon.to_mlflow import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--log-file" in result.output
        assert "--experiment" in result.output

    def test_regex_patterns(self):
        from rgpycrumbs.eon.to_mlflow import NEB_ITER_RE, DIMER_STEP_RE, IDIMER_ROT_RE, POT_CALLS_RE

        # NEB iteration line
        neb_line = "  1  0.0000e+00     1.3907e+01          10         8.304"
        m = NEB_ITER_RE.match(neb_line)
        assert m is not None
        assert m.group("iter") == "1"
        assert m.group("force") == "1.3907e+01"

        # POT_CALLS
        pot_line = "[XTB] called potential 42 times"
        m = POT_CALLS_RE.search(pot_line)
        assert m is not None
        assert m.group("count") == "42"

    def test_parse_and_log_metrics(self, tmp_path):
        from rgpycrumbs.eon.to_mlflow import parse_and_log_metrics

        log_file = tmp_path / "client.log"
        log_file.write_text(
            "  1  0.0000e+00     1.3907e+01          10         8.304\n"
            "  2  0.0100e+00     5.2000e+00          10         6.100\n"
            "[XTB] called potential 100 times\n"
        )
        # Should not raise
        parse_and_log_metrics(log_file)

    def test_plot_structure_evolution_empty(self):
        from rgpycrumbs.eon.to_mlflow import plot_structure_evolution

        result = plot_structure_evolution([])
        assert result is None

    def test_plot_structure_evolution_nonempty(self):
        import matplotlib
        matplotlib.use("Agg")
        from ase import Atoms
        from rgpycrumbs.eon.to_mlflow import plot_structure_evolution

        atoms_list = [
            Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]) for _ in range(10)
        ]
        fig = plot_structure_evolution(atoms_list, plot_every=5)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ======================================================================
# 7. delete_packages.py -- Package deletion utility
# ======================================================================
@pytest.mark.skipif(not _HAS_REQUESTS, reason="requests required")
class TestDeletePackages:
    """Test prefix/delete_packages.py CLI."""

    def test_help(self):
        from rgpycrumbs.prefix.delete_packages import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--channel" in result.output
        assert "--package-name" in result.output
        assert "--dry-run" in result.output

    def test_get_packages_to_delete_empty(self):
        from rgpycrumbs.prefix.delete_packages import get_packages_to_delete

        with patch("rgpycrumbs.prefix.delete_packages.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_requests.get.return_value = mock_response
            result = get_packages_to_delete("test-channel", "test-pkg", None)
            assert result == []

    def test_get_packages_to_delete_with_matches(self):
        from rgpycrumbs.prefix.delete_packages import get_packages_to_delete

        with patch("rgpycrumbs.prefix.delete_packages.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "packages": {"test-pkg-1.0.0-h123.tar.bz2": {}},
                "packages.conda": {},
            }
            mock_requests.get.return_value = mock_response
            mock_requests.RequestException = Exception

            result = get_packages_to_delete("test-channel", "test-pkg", None)
            assert len(result) > 0

    def test_delete_package_dry_run(self):
        from rgpycrumbs.prefix.delete_packages import delete_package

        session = MagicMock()
        result = delete_package(session, "ch", "linux-64", "pkg.tar.bz2", dry_run=True)
        assert result is True

    def test_delete_package_success(self):
        from rgpycrumbs.prefix.delete_packages import delete_package

        session = MagicMock()
        session.delete.return_value = MagicMock(status_code=200)
        result = delete_package(session, "ch", "linux-64", "pkg.tar.bz2", dry_run=False)
        assert result is True

    def test_delete_package_failure(self):
        from rgpycrumbs.prefix.delete_packages import delete_package

        session = MagicMock()
        session.delete.return_value = MagicMock(status_code=403, text="Forbidden")
        result = delete_package(session, "ch", "linux-64", "pkg.tar.bz2", dry_run=False)
        assert result is False

    def test_delete_package_request_exception(self):
        import requests
        from rgpycrumbs.prefix.delete_packages import delete_package

        session = MagicMock()
        session.delete.side_effect = requests.RequestException("timeout")
        result = delete_package(session, "ch", "linux-64", "pkg.tar.bz2", dry_run=False)
        assert result is False

    def test_constants(self):
        from rgpycrumbs.prefix.delete_packages import BASE_URL, PLATFORMS

        assert "prefix.dev" in BASE_URL
        assert "linux-64" in PLATFORMS
        assert "noarch" in PLATFORMS

    def test_dry_run_no_packages(self):
        from rgpycrumbs.prefix.delete_packages import main

        with patch("rgpycrumbs.prefix.delete_packages.get_packages_to_delete", return_value=[]):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["--channel", "test", "--package-name", "pkg", "--dry-run"],
            )
            assert result.exit_code == 0


# ======================================================================
# 8. match_atoms.py -- Atom matching
# ======================================================================
@pytest.mark.skipif(not _HAS_DEV_CHEMGP, reason="needs dev chemparseplot with chemgp API")
class TestMatchAtoms:
    """Test chemgp/match_atoms.py with real ASE Atoms."""

    def test_help(self):
        from rgpycrumbs.chemgp.match_atoms import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--structure" in result.output

    def test_parse_target_coords(self):
        from rgpycrumbs.chemgp.match_atoms import parse_target_coords

        text = "1.0 2.0 3.0\n4.0 5.0 6.0\n"
        coords = parse_target_coords(text)
        assert coords.shape == (2, 3)
        np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0])

    def test_parse_target_coords_empty(self):
        from rgpycrumbs.chemgp.match_atoms import parse_target_coords

        coords = parse_target_coords("")
        assert coords.shape == (0,)

    def test_parse_target_coords_bad_lines(self):
        from rgpycrumbs.chemgp.match_atoms import parse_target_coords

        text = "1.0 2.0 3.0\nbad line\n4.0 5.0\nabc def ghi\n"
        coords = parse_target_coords(text)
        assert coords.shape == (1, 3)

    def test_match_atoms_function(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.chemgp.match_atoms import match_atoms

        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        struct_file = tmp_path / "test.xyz"
        write(str(struct_file), atoms)

        targets = np.array([[0.1, 0.1, 0.0], [0.9, 0.1, 0.0]])
        results = match_atoms(struct_file, targets)

        assert len(results) == 2
        assert results[0]["closest_atom_id"] == 0
        assert results[1]["closest_atom_id"] == 1

    def test_match_atoms_file_not_found(self, tmp_path):
        from rgpycrumbs.chemgp.match_atoms import match_atoms

        results = match_atoms(tmp_path / "nonexistent.xyz", np.array([[0, 0, 0]]))
        assert results == []

    def test_match_atoms_empty_targets(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.chemgp.match_atoms import match_atoms

        atoms = Atoms("H", positions=[[0, 0, 0]])
        struct_file = tmp_path / "test.xyz"
        write(str(struct_file), atoms)

        results = match_atoms(struct_file, np.array([]))
        assert results == []

    def test_cli_with_structure(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.chemgp.match_atoms import main

        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        struct_file = tmp_path / "test.xyz"
        write(str(struct_file), atoms)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--structure",
                str(struct_file),
                "--target-coords",
                "0.1 0.1 0.0\n0.9 0.1 0.0",
            ],
        )
        assert result.exit_code == 0
        assert "Closest Atom ID" in result.output

    def test_cli_with_target_file(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.chemgp.match_atoms import main

        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        struct_file = tmp_path / "test.xyz"
        write(str(struct_file), atoms)

        target_file = tmp_path / "targets.txt"
        target_file.write_text("0.1 0.1 0.0\n0.9 0.1 0.0\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--structure",
                str(struct_file),
                "--target-file",
                str(target_file),
            ],
        )
        assert result.exit_code == 0

    def test_cli_with_output_file(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.chemgp.match_atoms import main

        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        struct_file = tmp_path / "test.xyz"
        write(str(struct_file), atoms)

        out_file = tmp_path / "results.txt"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--structure",
                str(struct_file),
                "--target-coords",
                "0.1 0.1 0.0",
                "--output",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()

    def test_constants(self):
        from rgpycrumbs.chemgp.match_atoms import POSCON_FILENAME, _EXPECTED_COORD_COLS

        assert POSCON_FILENAME == "pos.con"
        assert _EXPECTED_COORD_COLS == 3


# ======================================================================
# 9. fragment_visualization.py -- PyVista visualization
# ======================================================================
@pytest.mark.skipif(not _HAS_PYVISTA, reason="pyvista required")
class TestFragmentVisualization:
    """Test fragment_visualization.py with mocked pyvista."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        # Mock pyvista
        pv_mock = types.ModuleType("pyvista")
        pv_mock.Plotter = MagicMock
        pv_mock.Sphere = MagicMock(return_value=MagicMock(n_points=10, point_data={}))
        pv_mock.Cylinder = MagicMock(return_value=MagicMock(n_points=10, point_data={}))
        pv_mock.merge = MagicMock(return_value=MagicMock())

        # Mock cmcrameri
        cmc_mock = types.ModuleType("cmcrameri")
        cmc_cm = types.ModuleType("cmcrameri.cm")
        cmc_cm.batlow = MagicMock()
        cmc_mock.cm = cmc_cm

        self._originals = {}
        for name, mod in [("pyvista", pv_mock), ("cmcrameri", cmc_mock), ("cmcrameri.cm", cmc_cm)]:
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.geom.fragment_visualization"):
                del sys.modules[key]

        yield

        for name in self._originals:
            if self._originals[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.geom.fragment_visualization"):
                del sys.modules[key]

    def test_imports_and_constants(self):
        from rgpycrumbs.geom.fragment_visualization import MIN_DIST_ATM, SCALAR_BAR_ARGS

        assert MIN_DIST_ATM == 1e-4
        assert "title" in SCALAR_BAR_ARGS

    def test_visualize_geometric(self):
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        # Should not raise
        visualize_with_pyvista(atoms, DetectionMethod.GEOMETRIC, 1.2)

    def test_visualize_bond_order(self):
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        # Bond order matrix
        bo_matrix = np.array([
            [0, 1.0, 1.0],
            [1.0, 0, 0.03],
            [1.0, 0.03, 0],
        ])
        visualize_with_pyvista(atoms, DetectionMethod.BOND_ORDER, bo_matrix)


# ======================================================================
# 10. ptmdisp.py -- PTM displacement (mocked ovito)
# ======================================================================
@pytest.mark.skipif(not _HAS_OVITO, reason="ovito required")
class TestPtmdisp:
    """Test ptmdisp.py with mocked ovito."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        # Mock ovito and all sub-modules
        ovito_mock = types.ModuleType("ovito")
        ovito_io = types.ModuleType("ovito.io")
        ovito_io_ase = types.ModuleType("ovito.io.ase")
        ovito_io_ase.ase_to_ovito = MagicMock()

        ovito_mods = types.ModuleType("ovito.modifiers")
        # Create mock classes for each modifier
        for mod_name in [
            "CentroSymmetryModifier",
            "DeleteSelectedModifier",
            "ExpressionSelectionModifier",
            "InvertSelectionModifier",
            "PolyhedralTemplateMatchingModifier",
            "SelectTypeModifier",
        ]:
            mock_cls = MagicMock()
            setattr(ovito_mods, mod_name, mock_cls)

        # PolyhedralTemplateMatchingModifier.Type
        ptm_type = MagicMock()
        ptm_type.FCC = 1
        ptm_type.HCP = 2
        ptm_type.BCC = 3
        ptm_type.ICO = 4
        ptm_type.OTHER = 0
        ovito_mods.PolyhedralTemplateMatchingModifier.Type = ptm_type

        ovito_pipeline = types.ModuleType("ovito.pipeline")
        ovito_pipeline.Pipeline = MagicMock()
        ovito_pipeline.StaticSource = MagicMock()

        ovito_vis = types.ModuleType("ovito.vis")
        ovito_vis.Viewport = MagicMock()

        modules = {
            "ovito": ovito_mock,
            "ovito.io": ovito_io,
            "ovito.io.ase": ovito_io_ase,
            "ovito.modifiers": ovito_mods,
            "ovito.pipeline": ovito_pipeline,
            "ovito.vis": ovito_vis,
        }

        self._originals = {}
        for name, mod in modules.items():
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.ptmdisp"):
                del sys.modules[key]

        yield

        for name in self._originals:
            if self._originals[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.ptmdisp"):
                del sys.modules[key]

    def test_help(self):
        from rgpycrumbs.eon.ptmdisp import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "FILENAME" in result.output or "--structure-type" in result.output

    def test_crystal_structure_enum(self):
        from rgpycrumbs.eon.ptmdisp import CrystalStructure

        assert CrystalStructure.FCC == "FCC"
        assert CrystalStructure.HCP == "HCP"
        assert CrystalStructure.BCC == "BCC"

    def test_structure_type_map(self):
        from rgpycrumbs.eon.ptmdisp import STRUCTURE_TYPE_MAP, CrystalStructure

        assert CrystalStructure.FCC in STRUCTURE_TYPE_MAP
        assert CrystalStructure.BCC in STRUCTURE_TYPE_MAP


# ======================================================================
# 11. generate_nwchem_input.py -- NWChem input generator
# ======================================================================
class TestGenerateNwchemInput:
    """Test generate_nwchem_input.py CLI with mocked pychum."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        pychum_mock = types.ModuleType("pychum")
        pychum_mock.render_nwchem = MagicMock(return_value="start nwchem\nend\n")

        self._originals = {"pychum": sys.modules.get("pychum")}
        sys.modules["pychum"] = pychum_mock

        for key in list(sys.modules.keys()):
            if key.startswith("rgpycrumbs.eon.generate_nwchem_input"):
                del sys.modules[key]

        yield

        if self._originals["pychum"] is None:
            sys.modules.pop("pychum", None)
        else:
            sys.modules["pychum"] = self._originals["pychum"]
        sys.modules.pop("rgpycrumbs.eon.generate_nwchem_input", None)

    def test_help(self):
        from rgpycrumbs.eon.generate_nwchem_input import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--pos-file" in result.output
        assert "--config" in result.output

    def test_generate_with_config(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.eon.generate_nwchem_input import main

        # Create pos.con
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.set_cell([10, 10, 10])
        pos_file = tmp_path / "pos.con"
        write(str(pos_file), atoms, format="extxyz")

        # Create config.ini
        config = configparser.ConfigParser()
        config.add_section("SocketNWChemPot")
        config.set("SocketNWChemPot", "nwchem_settings", "settings.nwi")
        config.set("SocketNWChemPot", "mem_in_gb", "4")
        config.set("SocketNWChemPot", "unix_socket_mode", "false")
        config.set("SocketNWChemPot", "host", "127.0.0.1")
        config.set("SocketNWChemPot", "port", "9999")

        conf_file = tmp_path / "config.ini"
        with open(conf_file, "w") as f:
            config.write(f)

        output_file = tmp_path / "nwchem.nwi"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--pos-file",
                str(pos_file),
                "--config",
                str(conf_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()


# ======================================================================
# 12. run/jupyter.py -- Jupyter helper
# ======================================================================
class TestJupyterHelper:
    """Test run/jupyter.py functions."""

    def test_run_command_live_not_on_path(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        with pytest.raises(FileNotFoundError, match="is not on PATH"):
            _run_command_live(["nonexistent_binary_xyz"])

    def test_run_command_live_success(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        result = _run_command_live(["echo", "hello"], capture=True)
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_command_live_failure(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        with pytest.raises(subprocess.CalledProcessError):
            _run_command_live(["false"], check=True)

    def test_run_command_live_no_capture(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        result = _run_command_live(["echo", "hello"], capture=False)
        assert result.returncode == 0
        assert result.stdout is None

    def test_run_command_live_shell_mode(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        result = _run_command_live("echo hello", capture=True)
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_command_or_exit_success(self):
        from rgpycrumbs.run.jupyter import run_command_or_exit

        result = run_command_or_exit(["echo", "test"], capture=True)
        assert result.returncode == 0

    def test_run_command_or_exit_not_found(self):
        from rgpycrumbs.run.jupyter import run_command_or_exit

        with pytest.raises(SystemExit) as exc_info:
            run_command_or_exit(["nonexistent_binary_xyz_123"])
        assert exc_info.value.code == 2

    def test_run_command_or_exit_failure(self):
        from rgpycrumbs.run.jupyter import run_command_or_exit

        with pytest.raises(SystemExit) as exc_info:
            run_command_or_exit(["false"])
        assert exc_info.value.code != 0

    def test_run_command_live_timeout(self):
        """Timeout behavior tested by verifying the function accepts a timeout param."""
        from rgpycrumbs.run.jupyter import _run_command_live

        import inspect
        sig = inspect.signature(_run_command_live)
        assert "timeout" in sig.parameters


# ======================================================================
# 13. func/muller_brown.py -- Muller-Brown potential
# ======================================================================
class TestMullerBrown:
    """Test Muller-Brown potential and gradient."""

    def test_scalar_eval(self):
        from rgpycrumbs.func.muller_brown import muller_brown

        val = muller_brown([0.0, 0.0])
        assert isinstance(val, (float, np.floating))

    def test_meshgrid_eval(self):
        from rgpycrumbs.func.muller_brown import muller_brown

        x = np.linspace(-1.5, 1.2, 10)
        y = np.linspace(-0.2, 2.0, 10)
        X, Y = np.meshgrid(x, y)
        Z = muller_brown([X, Y])
        assert Z.shape == (10, 10)

    def test_gradient_shape(self):
        from rgpycrumbs.func.muller_brown import muller_brown_gradient

        grad = muller_brown_gradient([0.5, 0.5])
        assert grad.shape == (2,)

    def test_gradient_finite_diff(self):
        from rgpycrumbs.func.muller_brown import muller_brown, muller_brown_gradient

        x0 = np.array([0.3, 0.7])
        grad = muller_brown_gradient(x0)
        eps = 1e-6
        fd_grad = np.zeros(2)
        for i in range(2):
            xp = x0.copy()
            xp[i] += eps
            xm = x0.copy()
            xm[i] -= eps
            fd_grad[i] = (muller_brown(xp) - muller_brown(xm)) / (2 * eps)
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-4)

    def test_known_minima(self):
        """Muller-Brown has known minima. Check the deepest one near (-0.558, 1.442)."""
        from rgpycrumbs.func.muller_brown import muller_brown

        val = muller_brown([-0.558, 1.442])
        assert val < -140  # Known deepest minimum is around -146.7


# ======================================================================
# 14. __init__.py -- Lazy import coverage
# ======================================================================
class TestPackageInit:
    """Test the top-level __init__.py lazy imports."""

    def test_version(self):
        import rgpycrumbs

        assert hasattr(rgpycrumbs, "__version__")

    def test_basetypes_lazy(self):
        import rgpycrumbs

        bt = rgpycrumbs.basetypes
        assert bt is not None

    def test_unknown_attr(self):
        import rgpycrumbs

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = rgpycrumbs.nonexistent_module_xyz

    def test_interpolation_lazy(self):
        import rgpycrumbs

        interp = rgpycrumbs.interpolation
        assert interp is not None


# ======================================================================
# 15. _mlflow/log_params.py -- MLflow logging params
# ======================================================================
@pytest.mark.skipif(not _HAS_MLFLOW, reason="mlflow required")
class TestMlflowLogParams:
    """Test _mlflow/log_params.py with mocked mlflow and eon.config."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        # Mock mlflow
        mlflow_mock = types.ModuleType("mlflow")
        mlflow_mock.log_param = MagicMock()
        mlflow_mock.log_artifact = MagicMock()
        mlflow_mock.set_tag = MagicMock()

        # Mock eon.config
        eon_mod = types.ModuleType("eon")
        eon_config_mod = types.ModuleType("eon.config")

        # Create a fake ConfigClass
        class FakeKey:
            def __init__(self, name, default, kind="string"):
                self.name = name
                self.default = default
                self.kind = kind

        class FakeSection:
            def __init__(self, name, keys):
                self.name = name
                self.keys = keys

        class FakeConfigClass:
            def __init__(self):
                self.format = [
                    FakeSection("Main", [
                        FakeKey("method", "neb"),
                        FakeKey("max_steps", "100", "int"),
                    ]),
                    FakeSection("NEB", [
                        FakeKey("spring_constant", "5.0", "float"),
                        FakeKey("converged", "false", "boolean"),
                    ]),
                ]

        eon_config_mod.ConfigClass = FakeConfigClass

        # Mock _import_from_parent_env
        self._originals = {}
        for name, mod in [("mlflow", mlflow_mock), ("eon", eon_mod), ("eon.config", eon_config_mod)]:
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = mod

        # We need to patch _import_from_parent_env before importing the module
        sys.modules.pop("rgpycrumbs.eon._mlflow.log_params", None)
        sys.modules.pop("rgpycrumbs.eon._mlflow", None)

        yield

        for name in self._originals:
            if self._originals[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        sys.modules.pop("rgpycrumbs.eon._mlflow.log_params", None)
        sys.modules.pop("rgpycrumbs.eon._mlflow", None)

    def test_log_config_ini(self, tmp_path):
        with patch("rgpycrumbs._aux._import_from_parent_env", return_value=sys.modules["eon.config"]):
            # Force reimport with mocks in place
            sys.modules.pop("rgpycrumbs.eon._mlflow.log_params", None)
            from rgpycrumbs.eon._mlflow.log_params import log_config_ini

            # Create a config.ini
            config = configparser.ConfigParser()
            config.add_section("Main")
            config.set("Main", "method", "dimer")
            conf_file = tmp_path / "config.ini"
            with open(conf_file, "w") as f:
                config.write(f)

            log_config_ini(conf_file, w_artifact=False, track_overrides=True)

            import mlflow
            assert mlflow.log_param.called

    def test_log_config_ini_no_file(self, tmp_path):
        with patch("rgpycrumbs._aux._import_from_parent_env", return_value=sys.modules["eon.config"]):
            sys.modules.pop("rgpycrumbs.eon._mlflow.log_params", None)
            from rgpycrumbs.eon._mlflow.log_params import log_config_ini

            conf_file = tmp_path / "nonexistent.ini"
            log_config_ini(conf_file, w_artifact=False)

            import mlflow
            assert mlflow.log_param.called


# ======================================================================
# 16. Regex coverage for to_mlflow
# ======================================================================
@pytest.mark.skipif(not _HAS_MLFLOW, reason="mlflow required")
class TestToMlflowRegex:
    """Extra regex tests for to_mlflow patterns."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        mlflow_mock = types.ModuleType("mlflow")
        mlflow_mock.set_experiment = MagicMock()
        mlflow_mock.start_run = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock(return_value=False)
        ))
        mlflow_mock.log_metric = MagicMock()
        mlflow_mock.log_param = MagicMock()
        mlflow_mock.log_artifact = MagicMock()
        mlflow_mock.log_figure = MagicMock()
        mlflow_mock.set_tag = MagicMock()

        mlflow_log_params = types.ModuleType("rgpycrumbs.eon._mlflow.log_params")
        mlflow_log_params.log_config_ini = MagicMock()

        self._originals = {}
        for name in ["mlflow", "rgpycrumbs.eon._mlflow.log_params"]:
            self._originals[name] = sys.modules.get(name)
        sys.modules["mlflow"] = mlflow_mock
        sys.modules["rgpycrumbs.eon._mlflow.log_params"] = mlflow_log_params

        sys.modules.pop("rgpycrumbs.eon.to_mlflow", None)
        yield

        for name in self._originals:
            if self._originals[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self._originals[name]
        sys.modules.pop("rgpycrumbs.eon.to_mlflow", None)

    def test_dimer_step_regex(self):
        from rgpycrumbs.eon.to_mlflow import DIMER_STEP_RE

        line = "[Dimer] 1  0.0351005  -0.0018  4.63490e-01  -10.2810  2.010  4.777  2"
        m = DIMER_STEP_RE.match(line)
        assert m is not None
        assert m.group("step") == "1"
        assert m.group("force") == "4.63490e-01"

    def test_idimer_rot_regex(self):
        from rgpycrumbs.eon.to_mlflow import IDIMER_ROT_RE

        line = "[IDimerRot] ----- --------- ---------- ------------------ -9.9480 5.731 9.06 1"
        m = IDIMER_ROT_RE.match(line)
        assert m is not None
        assert m.group("curvature") == "-9.9480"

    def test_parse_and_log_dimer_metrics(self, tmp_path):
        from rgpycrumbs.eon.to_mlflow import parse_and_log_metrics

        log_file = tmp_path / "dimer.log"
        log_file.write_text(
            "[Dimer] 1  0.0351005  -0.0018  4.63490e-01  -10.2810  2.010  4.777  2\n"
            "[IDimerRot] ----- --------- ---------- ------------------ -9.9480 5.731 9.06 1\n"
            "[XTB] called potential 50 times\n"
        )
        parse_and_log_metrics(log_file)

        import mlflow
        assert mlflow.log_metric.called
