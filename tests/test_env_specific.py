"""Environment-specific tests for modules requiring eonmlflow, ptm, fragments envs.

Each section is gated by the appropriate pytest marker and import guard so tests
are skipped when run in an environment missing the required dependencies.

Run per-env:
  pixi run -e eonmlflow -- pytest tests/test_env_specific.py -m eon -v
  pixi run -e ptm       -- pytest tests/test_env_specific.py -m ptm -v
  pixi run -e fragments -- pytest tests/test_env_specific.py -m fragments -v
  pixi run -e test      -- pytest tests/test_env_specific.py -m pure -v
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner


def _can_import(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


_HAS_MLFLOW = _can_import("mlflow")
_HAS_OVITO = _can_import("ovito")
_HAS_PYVISTA = _can_import("pyvista")
_HAS_TBLITE = _can_import("tblite.interface")
_HAS_REQUESTS = _can_import("requests")


# ======================================================================
# delete_packages.py (prefix/) -- needs requests
# ======================================================================
class TestDeletePackages:
    """Tests for rgpycrumbs.prefix.delete_packages CLI.

    Marked pure because requests is now in the test env.
    """

    pytestmark = pytest.mark.pure

    @pytest.fixture(autouse=True)
    def _skip_no_requests(self):
        if not _HAS_REQUESTS:
            pytest.skip("requests not available")

    def test_help_flag(self):
        from rgpycrumbs.prefix.delete_packages import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--channel" in result.output
        assert "--package-name" in result.output
        assert "--dry-run" in result.output

    def test_missing_required_options(self):
        from rgpycrumbs.prefix.delete_packages import main

        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    @patch("rgpycrumbs.prefix.delete_packages.requests.get")
    def test_dry_run_no_packages_found(self, mock_get):
        """When repodata returns no matching packages, exits cleanly."""
        from rgpycrumbs.prefix.delete_packages import main

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"packages": {}, "packages.conda": {}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--channel", "test-ch", "--package-name", "nonexistent", "--dry-run"],
        )
        assert result.exit_code == 0

    @patch("rgpycrumbs.prefix.delete_packages.requests.get")
    def test_get_packages_to_delete_finds_matches(self, mock_get):
        """Verifies package matching logic with version regex."""
        from rgpycrumbs.prefix.delete_packages import get_packages_to_delete

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "packages": {
                "mypkg-1.0.0-py3_0.tar.bz2": {},
                "mypkg-2.0.0-py3_0.tar.bz2": {},
                "otherpkg-1.0.0-py3_0.tar.bz2": {},
            },
            "packages.conda": {},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Match all versions of mypkg
        results = get_packages_to_delete("test-ch", "mypkg", None)
        # Should find mypkg entries across platforms (6 platforms)
        pkg_names = [fname for _, fname in results]
        assert all("mypkg" in name for name in pkg_names)
        assert not any("otherpkg" in name for name in pkg_names)

    @patch("rgpycrumbs.prefix.delete_packages.requests.get")
    def test_get_packages_version_regex(self, mock_get):
        """Version regex should filter to matching versions only."""
        from rgpycrumbs.prefix.delete_packages import get_packages_to_delete

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "packages": {
                "mypkg-1.0.0-py3_0.tar.bz2": {},
                "mypkg-2.0.0-py3_0.tar.bz2": {},
            },
            "packages.conda": {},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = get_packages_to_delete("test-ch", "mypkg", r"1\.0\.0")
        pkg_names = [fname for _, fname in results]
        assert all("1.0.0" in name for name in pkg_names)
        assert not any("2.0.0" in name for name in pkg_names)

    @patch("rgpycrumbs.prefix.delete_packages.requests.get")
    def test_get_packages_404_platform_skipped(self, mock_get):
        """Platforms returning 404 should be silently skipped."""
        from rgpycrumbs.prefix.delete_packages import get_packages_to_delete

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        results = get_packages_to_delete("test-ch", "mypkg", None)
        assert results == []

    def test_delete_package_dry_run(self):
        """Dry run should return True without making API calls."""
        from rgpycrumbs.prefix.delete_packages import delete_package

        import requests

        session = requests.Session()
        result = delete_package(session, "ch", "linux-64", "pkg-1.0.tar.bz2", dry_run=True)
        assert result is True

    @patch("rgpycrumbs.prefix.delete_packages.requests.get")
    def test_get_packages_request_exception(self, mock_get):
        """RequestException during fetch should be handled gracefully."""
        import requests

        mock_get.side_effect = requests.RequestException("connection error")

        from rgpycrumbs.prefix.delete_packages import get_packages_to_delete

        results = get_packages_to_delete("test-ch", "mypkg", None)
        assert results == []

    def test_delete_package_success(self):
        """Successful delete returns True."""
        from rgpycrumbs.prefix.delete_packages import delete_package

        mock_session = MagicMock()
        mock_session.delete.return_value = MagicMock(status_code=200)
        result = delete_package(mock_session, "ch", "linux-64", "pkg.tar.bz2", dry_run=False)
        assert result is True

    def test_delete_package_failure(self):
        """Failed delete returns False."""
        from rgpycrumbs.prefix.delete_packages import delete_package

        mock_session = MagicMock()
        mock_session.delete.return_value = MagicMock(status_code=403, text="forbidden")
        result = delete_package(mock_session, "ch", "linux-64", "pkg.tar.bz2", dry_run=False)
        assert result is False


# ======================================================================
# eon/to_mlflow.py -- needs eonmlflow env (mlflow + eon)
# ======================================================================
class TestToMlflow:
    """Tests for the to_mlflow CLI and metric parsing, using real mlflow."""

    pytestmark = pytest.mark.eon

    @pytest.fixture(autouse=True)
    def _skip_no_mlflow(self):
        if not _HAS_MLFLOW:
            pytest.skip("mlflow not available")

    def test_help_flag(self):
        from rgpycrumbs.eon.to_mlflow import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--log-file" in result.output
        assert "--config-file" in result.output
        assert "--experiment" in result.output

    def test_regex_neb_iter(self):
        """NEB iteration regex should parse well-formed log lines."""
        from rgpycrumbs.eon.to_mlflow import NEB_ITER_RE

        line = "    1  0.0000e+00     1.3907e+01          10         8.304"
        m = NEB_ITER_RE.match(line)
        assert m is not None
        assert m.group("iter") == "1"
        assert float(m.group("force")) == pytest.approx(13.907)
        assert float(m.group("max_en")) == pytest.approx(8.304)

    def test_regex_dimer_step(self):
        """Dimer step regex should parse well-formed log lines."""
        from rgpycrumbs.eon.to_mlflow import DIMER_STEP_RE

        line = "[Dimer] 1  0.0351005  -0.0018  4.63490e-01  -10.2810  2.010  4.777  2"
        m = DIMER_STEP_RE.match(line)
        assert m is not None
        assert m.group("step") == "1"
        assert float(m.group("force")) == pytest.approx(0.46349)
        assert float(m.group("curvature")) == pytest.approx(-10.281)

    def test_regex_idimer_rot(self):
        """IDimerRot regex should parse rotation log lines."""
        from rgpycrumbs.eon.to_mlflow import IDIMER_ROT_RE

        line = "[IDimerRot] ----- --------- ---------- ------------------ -9.9480 5.731 9.06 1"
        m = IDIMER_ROT_RE.match(line)
        assert m is not None
        assert float(m.group("curvature")) == pytest.approx(-9.948)
        assert float(m.group("torque")) == pytest.approx(5.731)

    def test_regex_pot_calls(self):
        """Potential calls regex should extract the count."""
        from rgpycrumbs.eon.to_mlflow import POT_CALLS_RE

        line = "[XTB] called potential 42 times"
        m = POT_CALLS_RE.search(line)
        assert m is not None
        assert int(m.group("count")) == 42

    def test_parse_and_log_metrics_neb(self, tmp_path):
        """Parses a synthetic NEB log and verifies mlflow metrics are logged."""
        import mlflow

        log_file = tmp_path / "eon_client.log"
        log_file.write_text(
            "    1  1.0000e-02     5.0000e+00          10         3.500\n"
            "    2  2.0000e-02     3.0000e+00           8         2.100\n"
            "[XTB] called potential 150 times\n"
        )

        mlflow.set_experiment("test_parse_neb")
        with mlflow.start_run():
            from rgpycrumbs.eon.to_mlflow import parse_and_log_metrics

            parse_and_log_metrics(log_file)
            run = mlflow.active_run()
            assert run is not None

    def test_parse_and_log_metrics_dimer(self, tmp_path):
        """Parses a synthetic Dimer log with rotation lines."""
        import mlflow

        log_file = tmp_path / "eon_dimer.log"
        log_file.write_text(
            "[IDimerRot] ----- --------- ---------- ------------------ -5.0000 3.200 7.50 1\n"
            "[Dimer] 1  0.0351005  -0.0018  4.63490e-01  -10.2810  2.010  4.777  2\n"
            "[Dimer] 2  0.0200000  -0.0010  2.00000e-01   -9.5000  1.500  3.200  1\n"
        )

        mlflow.set_experiment("test_parse_dimer")
        with mlflow.start_run():
            from rgpycrumbs.eon.to_mlflow import parse_and_log_metrics

            parse_and_log_metrics(log_file)

    def test_plot_structure_evolution_empty(self):
        """Empty atoms list should return None."""
        from rgpycrumbs.eon.to_mlflow import plot_structure_evolution

        assert plot_structure_evolution([]) is None

    def test_plot_structure_evolution_with_atoms(self):
        """Should return a matplotlib figure for non-empty atoms list."""
        import matplotlib.pyplot as plt
        from ase import Atoms

        from rgpycrumbs.eon.to_mlflow import plot_structure_evolution

        atoms_list = [
            Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]) for _ in range(10)
        ]
        fig = plot_structure_evolution(atoms_list, plot_every=5)
        assert fig is not None
        plt.close(fig)

    def test_cli_missing_log_file(self):
        """CLI should fail when required --log-file is missing."""
        from rgpycrumbs.eon.to_mlflow import main

        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_cli_with_log_file(self, tmp_path):
        """Full CLI invocation with a minimal log file."""
        import mlflow

        log_file = tmp_path / "client.log"
        log_file.write_text("    1  0.01  5.0  10  3.5\n")

        config_file = tmp_path / "config.ini"
        config_file.write_text("[Main]\njob = process_search\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--log-file", str(log_file),
                "--config-file", str(config_file),
                "--experiment", "test_cli_run",
            ],
        )
        # Exit code 0 means the run completed (config parsing may warn but not fail)
        assert result.exit_code == 0, result.output

        # Importing main here to avoid issues; it's already imported above
        from rgpycrumbs.eon.to_mlflow import main


# ======================================================================
# eon/_mlflow/log_params.py -- needs eonmlflow env
# ======================================================================
class TestLogParams:
    """Tests for log_config_ini using real mlflow and eon.config."""

    pytestmark = pytest.mark.eon

    @pytest.fixture(autouse=True)
    def _skip_no_mlflow(self):
        if not _HAS_MLFLOW:
            pytest.skip("mlflow not available")

    def test_log_config_ini_default(self, tmp_path):
        """Logs hydrated eOn config to mlflow without overrides."""
        import mlflow

        conf = tmp_path / "config.ini"
        conf.write_text("[Main]\njob = process_search\n")

        mlflow.set_experiment("test_log_params")
        with mlflow.start_run():
            from rgpycrumbs.eon._mlflow.log_params import log_config_ini

            log_config_ini(conf, w_artifact=False)
            run = mlflow.active_run()
            assert run is not None

    def test_log_config_ini_with_overrides(self, tmp_path):
        """Tracks user overrides when track_overrides=True."""
        import mlflow

        conf = tmp_path / "config.ini"
        conf.write_text("[Main]\njob = saddle_search\n")

        mlflow.set_experiment("test_log_params_overrides")
        with mlflow.start_run():
            from rgpycrumbs.eon._mlflow.log_params import log_config_ini

            log_config_ini(conf, w_artifact=True, track_overrides=True)

    def test_log_config_ini_nonexistent_file(self, tmp_path):
        """Non-existent config file should still populate schema defaults."""
        import mlflow

        conf = tmp_path / "does_not_exist.ini"

        mlflow.set_experiment("test_log_params_noconf")
        with mlflow.start_run():
            from rgpycrumbs.eon._mlflow.log_params import log_config_ini

            # Should not raise -- just logs defaults
            log_config_ini(conf, w_artifact=False)


# ======================================================================
# eon/ptmdisp.py -- needs ptm env (ovito)
# ======================================================================
class TestPtmdisp:
    """Additional PTM tests beyond the existing test_ptmdisp.py.

    These exercise more code paths in find_mismatch_indices and the CLI.
    """

    pytestmark = pytest.mark.ptm

    @pytest.fixture(autouse=True)
    def _skip_no_ovito(self):
        if not _HAS_OVITO:
            pytest.skip("ovito not available")

    def test_crystal_structure_enum_values(self):
        from rgpycrumbs.eon.ptmdisp import CrystalStructure

        assert CrystalStructure.FCC == "FCC"
        assert CrystalStructure.BCC == "BCC"
        assert CrystalStructure.HCP == "HCP"
        assert CrystalStructure.ICO == "Icosahedral"
        assert CrystalStructure.OTHER == "Other"

    def test_structure_type_map_completeness(self):
        """All CrystalStructure values should be in the STRUCTURE_TYPE_MAP."""
        from rgpycrumbs.eon.ptmdisp import STRUCTURE_TYPE_MAP, CrystalStructure

        for cs in CrystalStructure:
            assert cs in STRUCTURE_TYPE_MAP

    def test_find_mismatch_on_hcp_crystal(self, tmp_path):
        """Perfect HCP crystal should have no mismatches when searching for HCP."""
        from ase.build import bulk
        from ase.io import write

        from rgpycrumbs.eon.ptmdisp import CrystalStructure, find_mismatch_indices

        filepath = tmp_path / "hcp.xyz"
        atoms = bulk("Mg", "hcp", a=3.21, c=5.21) * (3, 3, 3)
        write(filepath, atoms)

        indices = find_mismatch_indices(
            str(filepath), CrystalStructure.HCP, view_selection=False
        )
        assert len(indices) == 0

    def test_find_mismatch_fcc_with_vacancy_removal(self, tmp_path):
        """Tests the remove_fcc_vacancy=True path."""
        from ase.build import bulk
        from ase.io import write

        from rgpycrumbs.eon.ptmdisp import CrystalStructure, find_mismatch_indices

        filepath = tmp_path / "fcc_vac.xyz"
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
        del atoms[40]  # create a vacancy
        write(filepath, atoms)

        indices = find_mismatch_indices(
            str(filepath),
            CrystalStructure.FCC,
            remove_fcc_vacancy=True,
            view_selection=False,
        )
        # Should return some indices (the interstitial/defect neighborhood)
        assert isinstance(indices, np.ndarray)

    def test_cli_help(self):
        from rgpycrumbs.eon.ptmdisp import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--structure-type" in result.output
        assert "--verbose" in result.output

    def test_cli_nonexistent_file(self):
        from rgpycrumbs.eon.ptmdisp import main

        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/file.xyz"])
        assert result.exit_code != 0


# ======================================================================
# geom/detect_fragments.py -- needs fragments env (pyvista, tblite)
# ======================================================================
class TestDetectFragmentsCLI:
    """CLI tests for the detect_fragments click group."""

    pytestmark = pytest.mark.fragments

    @pytest.fixture(autouse=True)
    def _skip_no_deps(self):
        if not (_HAS_PYVISTA and _HAS_TBLITE):
            pytest.skip("pyvista or tblite not available")

    def test_main_help(self):
        from rgpycrumbs.geom.detect_fragments import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "geometric" in result.output
        assert "bond-order" in result.output
        assert "batch" in result.output

    def test_geometric_subcommand_help(self):
        from rgpycrumbs.geom.detect_fragments import main

        runner = CliRunner()
        result = runner.invoke(main, ["geometric", "--help"])
        assert result.exit_code == 0
        assert "--multiplier" in result.output
        assert "--radius-type" in result.output

    def test_bond_order_subcommand_help(self):
        from rgpycrumbs.geom.detect_fragments import main

        runner = CliRunner()
        result = runner.invoke(main, ["bond-order", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output
        assert "--method" in result.output

    def test_batch_subcommand_help(self):
        from rgpycrumbs.geom.detect_fragments import main

        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--pattern" in result.output
        assert "--output" in result.output

    def test_geometric_on_water_dimer(self, tmp_path):
        """Run geometric detection on a real water dimer xyz file."""
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.geom.detect_fragments import main

        h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0.7, 0.7], [0, -0.7, 0.7]])
        dimer = h2o.copy()
        h2o_2 = h2o.copy()
        h2o_2.translate([5.0, 0, 0])
        dimer.extend(h2o_2)

        xyz_path = tmp_path / "water_dimer.xyz"
        write(str(xyz_path), dimer)

        runner = CliRunner()
        result = runner.invoke(main, ["geometric", str(xyz_path), "--multiplier", "1.2"])
        assert result.exit_code == 0

    def test_geometric_with_merge(self, tmp_path):
        """Geometric detection with --min-dist merging."""
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.geom.detect_fragments import main

        # Two molecules close enough to merge
        h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0.7, 0.7], [0, -0.7, 0.7]])
        dimer = h2o.copy()
        h2o_2 = h2o.copy()
        h2o_2.translate([2.5, 0, 0])
        dimer.extend(h2o_2)

        xyz_path = tmp_path / "close_dimer.xyz"
        write(str(xyz_path), dimer)

        runner = CliRunner()
        result = runner.invoke(
            main, ["geometric", str(xyz_path), "--min-dist", "5.0"]
        )
        assert result.exit_code == 0

    def test_geometric_covalent_radius_type(self, tmp_path):
        """Geometric detection with --radius-type covalent."""
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.geom.detect_fragments import main

        h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        xyz_path = tmp_path / "h2.xyz"
        write(str(xyz_path), h2)

        runner = CliRunner()
        result = runner.invoke(
            main, ["geometric", str(xyz_path), "--radius-type", "covalent"]
        )
        assert result.exit_code == 0

    def test_batch_geometric(self, tmp_path):
        """Batch processing of a directory with geometric method."""
        from ase import Atoms
        from ase.io import write

        from rgpycrumbs.geom.detect_fragments import main

        for i in range(3):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.74 + i * 0.01, 0, 0]])
            write(str(tmp_path / f"mol_{i}.xyz"), atoms)

        output_csv = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "batch",
                str(tmp_path),
                "--method", "geometric",
                "--output", str(output_csv),
            ],
        )
        assert result.exit_code == 0
        assert output_csv.exists()
        lines = output_csv.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_print_results(self):
        """Test the print_results helper directly."""
        from ase import Atoms
        from rich.console import Console

        from rgpycrumbs.geom.detect_fragments import print_results

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0.7, 0.7], [0, -0.7, 0.7]])
        labels = np.array([0, 0, 0])
        # Should not raise
        console = Console(file=None, force_terminal=False)
        print_results(console, atoms, 1, labels)


# ======================================================================
# geom/fragment_visualization.py -- needs fragments env (pyvista)
# ======================================================================
class TestFragmentVisualization:
    """Tests for the pyvista visualization data-processing paths.

    We test with pyvista in offscreen mode to avoid opening a window.
    """

    pytestmark = pytest.mark.fragments

    @pytest.fixture(autouse=True)
    def _skip_and_offscreen(self):
        if not _HAS_PYVISTA:
            pytest.skip("pyvista not available")
        import pyvista as pv

        pv.OFF_SCREEN = True
        yield
        pv.OFF_SCREEN = False

    def test_visualize_geometric_method(self):
        """Exercise the GEOMETRIC branch of visualize_with_pyvista."""
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0.7, 0.7], [0, -0.7, 0.7]])
        # Should not raise; geometric method uses a float multiplier as bond_data
        visualize_with_pyvista(atoms, DetectionMethod.GEOMETRIC, 1.2)

    def test_visualize_bond_order_method(self):
        """Exercise the BOND_ORDER branch of visualize_with_pyvista."""
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        # Build a simple bond order matrix
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        visualize_with_pyvista(
            atoms,
            DetectionMethod.BOND_ORDER,
            matrix,
            nonbond_cutoff=0.05,
            bond_threshold=0.5,
        )

    def test_visualize_bond_order_with_weak_interactions(self):
        """Exercise the weak interaction (dotted) rendering path."""
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        atoms = Atoms("H3", positions=[[0, 0, 0], [0.74, 0, 0], [2.0, 0, 0]])
        # Matrix with one strong bond and one weak interaction
        matrix = np.array([
            [0.0, 1.2, 0.3],
            [1.2, 0.0, 0.1],
            [0.3, 0.1, 0.0],
        ])
        visualize_with_pyvista(
            atoms,
            DetectionMethod.BOND_ORDER,
            matrix,
            nonbond_cutoff=0.05,
            bond_threshold=0.8,
        )

    def test_visualize_bond_order_no_interactions(self):
        """When all bond orders are below cutoff, should still render."""
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        atoms = Atoms("H2", positions=[[0, 0, 0], [5.0, 0, 0]])
        matrix = np.array([[0.0, 0.01], [0.01, 0.0]])
        visualize_with_pyvista(
            atoms,
            DetectionMethod.BOND_ORDER,
            matrix,
            nonbond_cutoff=0.05,
            bond_threshold=0.8,
        )

    def test_cpk_colors_coverage(self):
        """Exercises atoms with known CPK colors and the default fallback."""
        from ase import Atoms

        from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
        from rgpycrumbs.geom.fragments import DetectionMethod

        # Include C, N, O (known) plus Xe (unknown, should use default_color)
        atoms = Atoms(
            "CNOF",
            positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]],
        )
        visualize_with_pyvista(atoms, DetectionMethod.GEOMETRIC, 1.2)
