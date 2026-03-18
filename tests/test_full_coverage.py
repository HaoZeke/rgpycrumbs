"""Comprehensive tests for modules testable in the base test environment.

Covers: parsers, basetypes, interpolation, _aux helpers, cli edge cases,
eon/helpers, func/muller_brown, geom/fragments, geom/ira (mocked),
surfaces/__init__ (mocked jax), and the top-level __init__.
"""

import configparser
import datetime
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.pure


# ======================================================================
# parsers/common.py
# ======================================================================
class TestParsersCommon:
    def test_num_regex_integer(self):
        from rgpycrumbs.parsers.common import _NUM

        assert _NUM.search("42").group() == "42"

    def test_num_regex_negative_float(self):
        from rgpycrumbs.parsers.common import _NUM

        assert _NUM.search("-3.14").group() == "-3.14"

    def test_num_regex_scientific(self):
        from rgpycrumbs.parsers.common import _NUM

        assert _NUM.search("1.5e-10").group() == "1.5e-10"

    def test_num_regex_positive_sign(self):
        from rgpycrumbs.parsers.common import _NUM

        assert _NUM.search("+2.7E+3").group() == "+2.7E+3"

    def test_num_regex_no_leading_digit(self):
        from rgpycrumbs.parsers.common import _NUM

        assert _NUM.search(".5").group() == ".5"

    def test_num_regex_trailing_dot(self):
        from rgpycrumbs.parsers.common import _NUM

        assert _NUM.search("7.").group() == "7."

    def test_num_regex_findall(self):
        from rgpycrumbs.parsers.common import _NUM

        text = "Energy = -3.14 eV, Force = 1.2e-3 Ha/Bohr"
        matches = _NUM.findall(text)
        assert "-3.14" in matches
        assert "1.2e-3" in matches


# ======================================================================
# parsers/bless.py
# ======================================================================
class TestParsersBless:
    def test_bless_log_pattern(self):
        from rgpycrumbs.parsers.bless import BLESS_LOG

        line = "[2024-10-28T18:58:24Z] Starting calculation"
        m = BLESS_LOG.match(line)
        assert m is not None
        assert m.group("timestamp") == "2024-10-28T18:58:24Z"
        assert m.group("logdata") == "Starting calculation"

    def test_bless_log_no_match(self):
        from rgpycrumbs.parsers.bless import BLESS_LOG

        assert BLESS_LOG.match("no brackets here") is None

    def test_bless_time_parsing(self):
        from rgpycrumbs.parsers.bless import BLESS_TIME

        t = BLESS_TIME("2024-10-28T18:58:24Z")
        assert isinstance(t, datetime.datetime)
        assert t.year == 2024
        assert t.month == 10
        assert t.day == 28
        assert t.hour == 18
        assert t.minute == 58
        assert t.second == 24
        assert t.tzinfo is datetime.timezone.utc

    def test_bless_time_difference(self):
        from rgpycrumbs.parsers.bless import BLESS_TIME

        t1 = BLESS_TIME("2024-10-28T18:58:21Z")
        t2 = BLESS_TIME("2024-10-28T18:58:24Z")
        assert (t2 - t1).total_seconds() == 3.0


# ======================================================================
# func/muller_brown.py
# ======================================================================
class TestMullerBrown:
    def test_muller_brown_scalar(self):
        from rgpycrumbs.func.muller_brown import muller_brown

        val = muller_brown([0.0, 0.0])
        assert np.isfinite(val)

    def test_muller_brown_known_minimum(self):
        """The global minimum is near (-0.558, 1.442), energy ~ -146.7."""
        from rgpycrumbs.func.muller_brown import muller_brown

        val = muller_brown([-0.558, 1.442])
        assert val < -140

    def test_muller_brown_vectorized(self):
        from rgpycrumbs.func.muller_brown import muller_brown

        x = np.linspace(-1.5, 1.2, 10)
        y = np.linspace(-0.2, 2.0, 10)
        X, Y = np.meshgrid(x, y)
        Z = muller_brown([X, Y])
        assert Z.shape == (10, 10)

    def test_muller_brown_gradient_shape(self):
        from rgpycrumbs.func.muller_brown import muller_brown_gradient

        grad = muller_brown_gradient([0.0, 0.5])
        assert grad.shape == (2,)

    def test_muller_brown_gradient_finite_diff(self):
        """Numerical gradient should approximately match analytical."""
        from rgpycrumbs.func.muller_brown import muller_brown, muller_brown_gradient

        x0 = np.array([0.3, 0.8])
        grad = muller_brown_gradient(x0)
        h = 1e-5
        for i in range(2):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += h
            xm[i] -= h
            numerical = (muller_brown(xp) - muller_brown(xm)) / (2 * h)
            np.testing.assert_allclose(grad[i], numerical, rtol=1e-4)


# ======================================================================
# eon/helpers.py
# ======================================================================
class TestEonHelpers:
    def test_write_eon_config_file(self, tmp_path):
        from rgpycrumbs.eon.helpers import write_eon_config

        settings = {
            "Main": {"job": "process_search", "temperature": 300},
            "Optimizer": {"converged_force": 0.01},
        }
        out = tmp_path / "config.ini"
        write_eon_config(out, settings)

        assert out.exists()
        config = configparser.ConfigParser()
        config.read(out)
        assert config["Main"]["job"] == "process_search"
        assert config["Main"]["temperature"] == "300"
        assert config["Optimizer"]["converged_force"] == "0.01"

    def test_write_eon_config_preserves_case(self, tmp_path):
        from rgpycrumbs.eon.helpers import write_eon_config

        settings = {"Section": {"CamelCase": "value"}}
        out = tmp_path / "test.ini"
        write_eon_config(out, settings)

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(out)
        assert "CamelCase" in config["Section"]

    def test_write_eon_config_dir_path(self, tmp_path):
        """When given a directory, should write config.ini inside it."""
        from rgpycrumbs.eon.helpers import write_eon_config

        settings = {"Main": {"job": "saddle_search"}}
        write_eon_config(tmp_path, settings)

        assert (tmp_path / "config.ini").exists()


# ======================================================================
# interpolation.py
# ======================================================================
class TestInterpolation:
    def test_spline_interp_basic(self):
        from rgpycrumbs.interpolation import spline_interp

        x = np.linspace(0, 2 * np.pi, 20)
        y = np.sin(x)
        x_fine, y_fine = spline_interp(x, y, num=50)

        assert len(x_fine) == 50
        assert len(y_fine) == 50
        assert x_fine[0] == pytest.approx(x.min())
        assert x_fine[-1] == pytest.approx(x.max())

    def test_spline_interp_accuracy(self):
        from rgpycrumbs.interpolation import spline_interp

        x = np.linspace(0, 2 * np.pi, 30)
        y = np.sin(x)
        x_fine, y_fine = spline_interp(x, y, num=200)

        y_true = np.sin(x_fine)
        np.testing.assert_allclose(y_fine, y_true, atol=0.01)


# ======================================================================
# _aux.py -- additional coverage for uncovered lines
# ======================================================================
class TestAuxHelpers:
    def test_getstrform(self, tmp_path):
        from rgpycrumbs._aux import getstrform

        p = tmp_path / "somefile.txt"
        p.touch()
        result = getstrform(p)
        assert isinstance(result, str)
        assert str(tmp_path) in result

    @patch("rgpycrumbs._aux.subprocess.run")
    @patch("rgpycrumbs._aux.shutil.which", return_value="/usr/bin/git")
    def test_get_gitroot(self, _mock_which, mock_run):
        from rgpycrumbs._aux import get_gitroot

        mock_run.return_value = MagicMock(
            stdout=b"/home/user/project\n", returncode=0
        )
        root = get_gitroot()
        assert root == Path("/home/user/project")

    def test_switchdir(self, tmp_path):
        from rgpycrumbs._aux import switchdir

        original = Path.cwd()
        with switchdir(tmp_path):
            assert Path.cwd() == tmp_path
        assert Path.cwd() == original

    def test_switchdir_restores_on_error(self, tmp_path):
        from rgpycrumbs._aux import switchdir

        original = Path.cwd()
        with pytest.raises(ValueError):  # noqa: PT011
            with switchdir(tmp_path):
                raise ValueError("boom")
        assert Path.cwd() == original

    def test_uv_install_no_installer(self, monkeypatch, tmp_path):
        from rgpycrumbs._aux import _uv_install

        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(RuntimeError, match="Failed to install"):
            _uv_install("fake-package", tmp_path / "target")

    @patch("rgpycrumbs._aux.subprocess.run", side_effect=OSError("fail"))
    @patch("shutil.which", return_value="/usr/bin/uv")
    def test_uv_install_both_fail(self, _mock_which, _mock_run, tmp_path):
        from rgpycrumbs._aux import _uv_install

        with pytest.raises(RuntimeError, match="Failed to install"):
            _uv_install("fake-package", tmp_path / "target")

    def test_uv_install_success(self, monkeypatch, tmp_path):
        from rgpycrumbs._aux import _uv_install

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
        monkeypatch.setattr(
            "rgpycrumbs._aux.subprocess.run",
            MagicMock(return_value=MagicMock(returncode=0)),
        )
        target = tmp_path / "target"
        _uv_install("some-package", target)
        assert target.is_dir()

    def test_import_from_parent_env_no_env_var(self, monkeypatch):
        from rgpycrumbs._aux import _import_from_parent_env

        monkeypatch.delenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", raising=False)
        result = _import_from_parent_env("nonexistent_module_xyz")
        assert result is None

    def test_import_from_parent_env_import_fails(self, monkeypatch):
        from rgpycrumbs._aux import _import_from_parent_env

        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", "/fake/path")
        result = _import_from_parent_env("nonexistent_module_xyz")
        assert result is None


# ======================================================================
# __init__.py -- lazy __getattr__
# ======================================================================
class TestTopLevelInit:
    def test_version_exists(self):
        import rgpycrumbs

        assert hasattr(rgpycrumbs, "__version__")

    def test_basetypes_importable(self):
        import rgpycrumbs.basetypes as bt

        assert hasattr(bt, "SaddleMeasure")

    def test_interpolation_importable(self):
        import rgpycrumbs.interpolation as interp

        assert hasattr(interp, "spline_interp")

    def test_getattr_unknown_raises(self):
        from rgpycrumbs import __getattr__ as ga

        with pytest.raises(AttributeError, match="no attribute"):
            ga("nonexistent_thing")


# ======================================================================
# surfaces/__init__.py -- nystrom_paths_needed, get_surface_model, __getattr__
# ======================================================================
class TestSurfacesInit:
    def test_nystrom_paths_needed_basic(self):
        from rgpycrumbs.surfaces import nystrom_paths_needed

        # 300 inducing, 7 images: ceil(300/7) + 1 = 43 + 1 = 44
        assert nystrom_paths_needed(300, 7) == 44

    def test_nystrom_paths_needed_exact_divide(self):
        from rgpycrumbs.surfaces import nystrom_paths_needed

        # 100 inducing, 10 images: 10 + 1 = 11
        assert nystrom_paths_needed(100, 10) == 11

    def test_nystrom_paths_needed_small(self):
        from rgpycrumbs.surfaces import nystrom_paths_needed

        # 1 inducing, 5 images: max(1, ceil(1/5)) + 1 = 1 + 1 = 2
        assert nystrom_paths_needed(1, 5) == 2

    def test_constants_exported(self):
        from rgpycrumbs.surfaces import (
            NYSTROM_N_INDUCING_DEFAULT,
            NYSTROM_THRESHOLD,
        )

        assert NYSTROM_THRESHOLD == 1000
        assert NYSTROM_N_INDUCING_DEFAULT == 300

    def test_getattr_unknown_raises(self):
        import rgpycrumbs.surfaces as surf

        with pytest.raises(AttributeError, match="no attribute"):
            _ = surf.NonExistentClass

    def test_get_surface_model_triggers_jax_import(self):
        """get_surface_model should attempt jax import for gradient models."""
        import rgpycrumbs.surfaces as surf

        with patch.object(surf, "__getattr__") as mock_getattr:
            mock_getattr.side_effect = ImportError("no jax")
            with pytest.raises(ImportError):
                surf.get_surface_model("grad_matern")

    def test_lazy_imports_mapping(self):
        from rgpycrumbs.surfaces import _LAZY_IMPORTS, _JAX_SUBMODULES

        # All gradient modules require jax
        for name, target in _LAZY_IMPORTS.items():
            if "gradient" in target.lower() or target == "standard":
                assert target in _JAX_SUBMODULES


# ======================================================================
# geom/fragments.py
# ======================================================================
class TestFragments:
    def test_build_graph_single_atom(self):
        from rgpycrumbs.geom.fragments import build_graph_and_find_components

        n_comp, labels = build_graph_and_find_components(1, [], [])
        assert n_comp == 1
        np.testing.assert_array_equal(labels, [0])

    def test_build_graph_disconnected(self):
        from rgpycrumbs.geom.fragments import build_graph_and_find_components

        # 3 atoms, no bonds
        n_comp, labels = build_graph_and_find_components(3, [], [])
        assert n_comp == 3

    def test_build_graph_connected(self):
        from rgpycrumbs.geom.fragments import build_graph_and_find_components

        # 3 atoms, 0-1 and 1-2 bonded
        n_comp, labels = build_graph_and_find_components(3, [0, 1], [1, 2])
        assert n_comp == 1

    def test_build_graph_two_fragments(self):
        from rgpycrumbs.geom.fragments import build_graph_and_find_components

        # 4 atoms: 0-1 bonded, 2-3 bonded
        n_comp, labels = build_graph_and_find_components(4, [0, 2], [1, 3])
        assert n_comp == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_find_fragments_geometric_water_dimer(self):
        """Two water molecules well separated should be two fragments."""
        from ase import Atoms

        from rgpycrumbs.geom.fragments import find_fragments_geometric

        # Water 1 at origin
        w1 = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        # Water 2 far away
        w2 = Atoms(
            "OH2", positions=[[10, 10, 10], [10.96, 10, 10], [9.76, 10.93, 10]]
        )
        combined = w1 + w2
        combined.set_cell([20, 20, 20])
        combined.set_pbc(False)

        n_comp, labels = find_fragments_geometric(combined, 1.2)
        assert n_comp == 2

    def test_find_fragments_geometric_empty(self):
        from ase import Atoms

        from rgpycrumbs.geom.fragments import find_fragments_geometric

        atoms = Atoms()
        n, labels = find_fragments_geometric(atoms, 1.2)
        assert n == 0
        assert len(labels) == 0

    def test_find_fragments_geometric_covalent_radii(self):
        """Test with radius_type='covalent'."""
        from ase import Atoms

        from rgpycrumbs.geom.fragments import find_fragments_geometric

        # Single H2 molecule
        h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        h2.set_cell([10, 10, 10])
        h2.set_pbc(False)
        n, labels = find_fragments_geometric(h2, 1.2, radius_type="covalent")
        assert n == 1

    def test_merge_fragments_single(self):
        from ase import Atoms

        from rgpycrumbs.geom.fragments import merge_fragments_by_distance

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        n, labels = merge_fragments_by_distance(atoms, 1, np.array([0, 0]), 5.0)
        assert n == 1

    def test_merge_fragments_by_distance(self):
        from ase import Atoms

        from rgpycrumbs.geom.fragments import merge_fragments_by_distance

        # Two "fragments" close together should merge
        atoms = Atoms("H4", positions=[[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [2.0, 0, 0]])
        labels = np.array([0, 0, 1, 1])
        n, new_labels = merge_fragments_by_distance(atoms, 2, labels, 3.0)
        assert n == 1
        assert len(np.unique(new_labels)) == 1

    def test_merge_fragments_no_merge(self):
        from ase import Atoms

        from rgpycrumbs.geom.fragments import merge_fragments_by_distance

        atoms = Atoms("H4", positions=[[0, 0, 0], [0.5, 0, 0], [100, 0, 0], [100.5, 0, 0]])
        labels = np.array([0, 0, 1, 1])
        n, new_labels = merge_fragments_by_distance(atoms, 2, labels, 3.0)
        assert n == 2

    def test_detection_method_enum(self):
        from rgpycrumbs.geom.fragments import DetectionMethod

        assert DetectionMethod.GEOMETRIC == "geometric"
        assert DetectionMethod.BOND_ORDER == "bond-order"


# ======================================================================
# geom/ira.py -- with mocked ira_mod
# ======================================================================
class TestIRAModule:
    def test_ira_comp_dataclass(self):
        """IRAComp can be imported (ira_mod import at module level will fail,
        so we test the dataclass by importing from a fresh mock context)."""
        # We need to mock ira_mod at the module level before importing
        mock_ira = MagicMock()
        with patch.dict(sys.modules, {"ira_mod": mock_ira}):
            # Force reimport
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            comp = mod.IRAComp(
                rot=np.eye(3),
                trans=np.zeros(3),
                perm=np.array([0, 1, 2]),
                hd=0.1,
            )
            assert comp.hd == 0.1
            assert comp.rot.shape == (3, 3)

    def test_incomparable_structures_error(self):
        mock_ira = MagicMock()
        with patch.dict(sys.modules, {"ira_mod": mock_ira}):
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            with pytest.raises(ValueError):
                raise mod.IncomparableStructuresError("test error")

    def test_perform_ira_match_different_lengths(self):
        from ase import Atoms

        mock_ira = MagicMock()
        with patch.dict(sys.modules, {"ira_mod": mock_ira}):
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            atm1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
            atm2 = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
            with pytest.raises(mod.IncomparableStructuresError):
                mod._perform_ira_match(atm1, atm2)

    def test_perform_ira_match_different_types(self):
        from ase import Atoms

        mock_ira = MagicMock()
        with patch.dict(sys.modules, {"ira_mod": mock_ira}):
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            atm1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
            atm2 = Atoms("He2", positions=[[0, 0, 0], [1, 0, 0]])
            with pytest.raises(mod.IncomparableStructuresError):
                mod._perform_ira_match(atm1, atm2)

    def test_is_ira_pair_incompatible_returns_false(self):
        from ase import Atoms

        mock_ira = MagicMock()
        with patch.dict(sys.modules, {"ira_mod": mock_ira}):
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            atm1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
            atm2 = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
            assert mod.is_ira_pair(atm1, atm2) is False

    def test_do_ira_returns_iracomp(self):
        from ase import Atoms

        mock_ira_mod = MagicMock()
        mock_instance = MagicMock()
        mock_instance.match.return_value = (
            np.eye(3),
            np.zeros(3),
            np.array([0, 1]),
            0.05,
        )
        mock_ira_mod.IRA.return_value = mock_instance

        with patch.dict(sys.modules, {"ira_mod": mock_ira_mod}):
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            atm1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
            atm2 = Atoms("H2", positions=[[0.1, 0, 0], [1.1, 0, 0]])
            result = mod.do_ira(atm1, atm2)
            assert isinstance(result, mod.IRAComp)
            assert result.hd == 0.05

    def test_calculate_rmsd(self):
        from ase import Atoms

        mock_ira_mod = MagicMock()
        mock_instance = MagicMock()
        mock_instance.match.return_value = (
            np.eye(3),
            np.zeros(3),
            np.array([0, 1]),
            0.01,
        )
        mock_ira_mod.IRA.return_value = mock_instance

        with patch.dict(sys.modules, {"ira_mod": mock_ira_mod}):
            import importlib

            if "rgpycrumbs.geom.ira" in sys.modules:
                del sys.modules["rgpycrumbs.geom.ira"]
            mod = importlib.import_module("rgpycrumbs.geom.ira")

            atm1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
            atm2 = Atoms("H2", positions=[[0.1, 0, 0], [1.1, 0, 0]])
            rmsd = mod.calculate_rmsd(atm1, atm2)
            assert rmsd > 0
            assert np.isfinite(rmsd)


# ======================================================================
# geom/api/alignment.py -- data classes and non-IRA paths
# ======================================================================
class TestAlignmentAPI:
    def test_ira_config_defaults(self):
        from rgpycrumbs.geom.api.alignment import IRAConfig

        config = IRAConfig()
        assert config.enabled is False
        assert config.kmax == 1.8

    def test_alignment_method_enum(self):
        from rgpycrumbs.geom.api.alignment import AlignmentMethod

        assert AlignmentMethod.ASE_PROCRUSTES is not None
        assert AlignmentMethod.IRA_PERMUTATION is not None
        assert AlignmentMethod.NONE is not None

    def test_alignment_result_used_ira(self):
        from ase import Atoms

        from rgpycrumbs.geom.api.alignment import (
            AlignmentMethod,
            AlignmentResult,
        )

        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        r1 = AlignmentResult(atoms=atoms, method=AlignmentMethod.IRA_PERMUTATION)
        assert r1.used_ira is True

        r2 = AlignmentResult(atoms=atoms, method=AlignmentMethod.ASE_PROCRUSTES)
        assert r2.used_ira is False

    def test_align_structure_robust_ase_fallback(self):
        """Without IRA, should fall back to ASE Procrustes."""
        from ase import Atoms

        from rgpycrumbs.geom.api.alignment import (
            AlignmentMethod,
            IRAConfig,
            align_structure_robust,
        )

        ref = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        mobile = Atoms("H2", positions=[[0.1, 0.1, 0], [1.1, 0.1, 0]])

        config = IRAConfig(enabled=False)
        result = align_structure_robust(ref, mobile, config)
        assert result.method == AlignmentMethod.ASE_PROCRUSTES
        assert result.used_ira is False

    def test_ira_match_inputs_dataclass(self):
        from rgpycrumbs.geom.api.alignment import IRAMatchInputs

        inputs = IRAMatchInputs(
            ref_count=2,
            ref_numbers=np.array([1, 1]),
            ref_positions=np.zeros((2, 3)),
            mobile_count=2,
            mobile_numbers=np.array([1, 1]),
            mobile_positions=np.ones((2, 3)),
            kmax=1.8,
        )
        assert inputs.ref_count == 2
        assert inputs.kmax == 1.8

    def test_ira_match_results_dataclass(self):
        from rgpycrumbs.geom.api.alignment import IRAMatchResults

        results = IRAMatchResults(
            rotation=np.eye(3),
            translation=np.zeros(3),
            permutation=np.array([0, 1]),
            hausdorff_dist=0.01,
        )
        assert results.hausdorff_dist == 0.01


# ======================================================================
# cli.py -- additional edge cases
# ======================================================================
class TestCLIEdgeCases:
    def test_get_scripts_nonexistent_folder(self):
        from rgpycrumbs.cli import _get_scripts_in_folder

        result = _get_scripts_in_folder("nonexistent_folder_xyz")
        assert result == []

    def test_get_scripts_excludes_library_modules(self, tmp_path, monkeypatch):
        from rgpycrumbs.cli import _get_scripts_in_folder

        # Create a fake folder with scripts and library modules
        folder = tmp_path / "test_group"
        folder.mkdir()
        (folder / "real_script.py").touch()
        (folder / "helpers.py").touch()  # should be excluded
        (folder / "utils.py").touch()  # should be excluded
        (folder / "_private.py").touch()  # should be excluded
        (folder / "__init__.py").touch()  # should be excluded
        (folder / "cli_something.py").touch()  # should strip cli_ prefix

        monkeypatch.setattr("rgpycrumbs.cli.PACKAGE_ROOT", tmp_path)
        result = _get_scripts_in_folder("test_group")
        assert "real_script" in result
        assert "something" in result  # cli_ prefix stripped
        assert "helpers" not in result
        assert "utils" not in result
        assert "_private" not in result
        assert "__init__" not in result

    def test_main_version_option(self):
        runner = CliRunner()
        from rgpycrumbs.cli import main

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_dispatch_script_not_found(self):
        """_dispatch should exit 1 when script file does not exist."""
        from rgpycrumbs.cli import _dispatch

        with pytest.raises(SystemExit) as exc_info:
            _dispatch("nonexistent_group", "nonexistent_script", ())
        assert exc_info.value.code == 1

    @patch("rgpycrumbs.cli.subprocess.run", side_effect=FileNotFoundError())
    def test_dispatch_uv_not_found(self, _mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        # Make script path exist
        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        with pytest.raises(SystemExit) as exc_info:
            _dispatch("group", "script", ())
        assert exc_info.value.code == 1

    @patch(
        "rgpycrumbs.cli.subprocess.run",
        side_effect=subprocess.CalledProcessError(42, "cmd"),
    )
    def test_dispatch_calledprocesserror(self, _mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        with pytest.raises(SystemExit) as exc_info:
            _dispatch("group", "script", ())
        assert exc_info.value.code == 42

    @patch(
        "rgpycrumbs.cli.subprocess.run",
        side_effect=KeyboardInterrupt(),
    )
    def test_dispatch_keyboard_interrupt(self, _mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        with pytest.raises(SystemExit) as exc_info:
            _dispatch("group", "script", ())
        assert exc_info.value.code == 130


# ======================================================================
# basetypes.py -- verify all types (already 100% but good to exercise)
# ======================================================================
class TestBasetypesExtended:
    def test_nebiter(self):
        from rgpycrumbs.basetypes import nebiter, nebpath

        path = nebpath(norm_dist=0.5, arc_dist=1.2, energy=-10.5)
        it = nebiter(iteration=3, nebpath=path)
        assert it.iteration == 3
        assert it.nebpath.energy == -10.5

    def test_dimer_opt(self):
        from rgpycrumbs.basetypes import DimerOpt

        d = DimerOpt()
        assert d.saddle == "dimer"
        assert d.rot == "lbfgs"

        d2 = DimerOpt(saddle="lanczos", rot="cg", trans="fire")
        assert d2.saddle == "lanczos"

    def test_spin_id(self):
        from rgpycrumbs.basetypes import SpinID

        s = SpinID(mol_id=1, spin="singlet")
        assert s.mol_id == 1
        assert s.spin == "singlet"

    def test_mol_geom(self):
        from rgpycrumbs.basetypes import MolGeom

        m = MolGeom(pos=np.array([[0, 0, 0]]), energy=-1.5, forces=np.array([[0.1, 0, 0]]))
        assert m.energy == -1.5

    def test_saddle_measure_defaults(self):
        from rgpycrumbs.basetypes import SaddleMeasure

        s = SaddleMeasure()
        assert s.pes_calls == 0
        assert s.success is False
        assert s.method == "not run"
        assert np.isnan(s.saddle_energy)
        assert s.termination_status == "not set"


# ======================================================================
# geom/analysis.py -- test with real ASE atoms
# ======================================================================
class TestGeomAnalysis:
    def test_analyze_single_molecule(self):
        from ase import Atoms

        from rgpycrumbs.geom.analysis import analyze_structure

        # Simple H2 molecule
        h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        h2.set_cell([10, 10, 10])
        h2.set_pbc(True)

        dist_mat, bond_mat, fragments, centroid_dist, corrected = analyze_structure(h2)

        assert dist_mat.shape == (2, 2)
        assert bond_mat.shape == (2, 2)
        assert len(fragments) == 1  # single molecule
        assert len(fragments[0]) == 2

    def test_analyze_two_fragments(self):
        from ase import Atoms

        from rgpycrumbs.geom.analysis import analyze_structure

        # Two H2 molecules far apart
        atoms = Atoms(
            "H4",
            positions=[[0, 0, 0], [0.74, 0, 0], [50, 50, 50], [50.74, 50, 50]],
        )
        atoms.set_cell([100, 100, 100])
        atoms.set_pbc(False)

        dist_mat, bond_mat, fragments, centroid_dist, corrected = analyze_structure(atoms)

        assert len(fragments) == 2
        assert centroid_dist.shape == (2, 2)
        assert len(corrected) == 1  # one pair of fragments
        # corrected tuple: (min_dist, sym_i, sym_j, covrad_sum, corrected_dist)
        assert len(corrected[0]) == 5
