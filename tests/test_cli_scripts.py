# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for CLI entry points and importable modules.

Tests that need cross-repo dev branches or heavyweight optional deps
(pypotlib, ovito) are guarded with skipif. They run in the pixi_envs workspace
where all repos are editable installs.
"""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tests._optional_imports import optional_import_available
from tests._optional_imports import has_module_spec

pytestmark = pytest.mark.pure

# Skip only for genuinely absent optional stacks.
_HAS_CHEMGP = all(
    has_module_spec(mod)
    for mod in ("chemparseplot", "matplotlib", "pandas", "plotnine", "h5py")
)
_HAS_PYPOTLIB = has_module_spec("pypotlib")
_HAS_XTS_MB = all(has_module_spec(mod) for mod in ("cmcrameri", "matplotlib"))


class TestMainCLI:
    def test_help(self):
        from rgpycrumbs.cli import main

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "eon" in result.output

    def test_eon_subgroup(self):
        from rgpycrumbs.cli import main

        result = CliRunner().invoke(main, ["eon", "--help"])
        assert result.exit_code == 0

    def test_version(self):
        import rgpycrumbs

        assert hasattr(rgpycrumbs, "__version__")


class TestPep723DispatcherCli:
    @pytest.mark.parametrize(
        "rel_path",
        [
            "chemgp/match_atoms.py",
            "chemgp/plot_gp.py",
            "eon/con_splitter.py",
            "eon/generate_nwchem_input.py",
            "eon/plt_min.py",
            "eon/plt_neb.py",
            "eon/plt_saddle.py",
            "eon/ptmdisp.py",
            "eon/to_mlflow.py",
            "geom/detect_fragments.py",
            "orca/generate_orca_input.py",
            "plumed/direct_reconstruction.py",
            "prefix/delete_packages.py",
        ],
    )
    def test_dispatched_scripts_share_python_floor(self, rel_path):
        script = Path(__file__).resolve().parent.parent / "rgpycrumbs" / rel_path
        text = script.read_text()
        assert '# requires-python = ">=3.11"' in text

    @pytest.mark.parametrize(
        ("argv", "expected_script"),
        [
            (["eon", "plt-neb", "--help"], "eon/plt_neb.py"),
            (["eon", "plt-saddle", "--help"], "eon/plt_saddle.py"),
            (["eon", "plt-min", "--help"], "eon/plt_min.py"),
            (["eon", "generate-nwchem-input", "--help"], "eon/generate_nwchem_input.py"),
        ],
    )
    @patch("rgpycrumbs.cli.subprocess.run")
    def test_help_routes_through_dispatcher(self, mock_run, argv, expected_script, monkeypatch):
        from rgpycrumbs.cli import main

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)

        result = CliRunner().invoke(main, argv)
        assert result.exit_code == 0

        command = mock_run.call_args.args[0]
        assert command[:2] == ["uv", "run"]
        assert command[2].endswith(expected_script)
        assert command[-1] == "--help"

    @patch("rgpycrumbs.cli.subprocess.run")
    def test_missing_job_dir_is_forwarded_to_script(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import main

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)

        result = CliRunner().invoke(main, ["eon", "plt-min"])
        assert result.exit_code == 0

        command = mock_run.call_args.args[0]
        assert command[:2] == ["uv", "run"]
        assert command[2].endswith("eon/plt_min.py")


@pytest.mark.skipif(not _HAS_XTS_MB, reason="xts muller-brown plotting stack missing")
class TestMullerBrownXts:
    def test_muller_brown(self):
        import numpy as np

        from rgpycrumbs.func.muller_brown import muller_brown

        val, grad = muller_brown(np.array([0.0, 0.0]))
        assert isinstance(val, float)
        assert grad.shape == (2,)

    def test_mb_import_has_no_side_effects(self, monkeypatch):
        import rgpycrumbs.xts.saddle.mb as mb_mod

        def fail_meshgrid(*_args, **_kwargs):
            msg = "mb import should not build surface grids"
            raise AssertionError(msg)

        def fail_surface(*_args, **_kwargs):
            msg = "mb import should not evaluate Muller-Brown surface"
            raise AssertionError(msg)

        monkeypatch.setattr("numpy.meshgrid", fail_meshgrid)
        monkeypatch.setattr("rgpycrumbs.func.muller_brown.muller_brown", fail_surface)
        reloaded = importlib.reload(mb_mod)
        assert hasattr(reloaded, "_surface_grid")


@pytest.mark.skipif(not _HAS_PYPOTLIB, reason="pypotlib not installed")
class TestCuH2Xts:

    def test_cuh2(self):
        from rgpycrumbs.xts.saddle.cuh2 import cuh2_potential

        assert callable(cuh2_potential)

    def test_cuh2_import_has_no_side_effects(self, monkeypatch):
        import rgpycrumbs.xts.saddle.cuh2 as cuh2_mod

        def fail_read(*_args, **_kwargs):
            msg = "cuh2 import should not read structures"
            raise AssertionError(msg)

        def fail_grid(*_args, **_kwargs):
            msg = "cuh2 import should not build plotting grids"
            raise AssertionError(msg)

        monkeypatch.setattr("ase.io.read", fail_read)
        monkeypatch.setattr("rgpycrumbs.xts.cuh2.datgen.get_from_gitroot_con", fail_grid)
        reloaded = importlib.reload(cuh2_mod)
        assert callable(reloaded.cuh2_potential)


@pytest.mark.skipif(
    not _HAS_CHEMGP,
    reason="chemparseplot chemgp not importable",
)
class TestChemGPMatchAtoms:
    def test_import(self):
        from rgpycrumbs.chemgp import match_atoms

        assert hasattr(match_atoms, "match_atoms")


class TestOptionalImportGuards:
    def test_missing_third_party_returns_false(self, monkeypatch):
        real_import = importlib.import_module

        def fake_import(name, package=None):
            if name == "chemparseplot.synthetic_optional":
                raise ModuleNotFoundError("missing pandas", name="pandas")
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        assert optional_import_available("chemparseplot.synthetic_optional") is False

    def test_first_party_breakage_raises(self, monkeypatch):
        real_import = importlib.import_module

        def fake_import(name, package=None):
            if name == "rgpycrumbs.synthetic_broken":
                raise ModuleNotFoundError("broken first-party import", name=name)
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with pytest.raises(ModuleNotFoundError):
            optional_import_available("rgpycrumbs.synthetic_broken")


class TestPackageInit:
    def test_basetypes(self):
        from rgpycrumbs.basetypes import DimerOpt, SaddleMeasure

        assert SaddleMeasure is not None
        assert DimerOpt is not None

    def test_interpolation(self):
        from rgpycrumbs.interpolation import spline_interp

        assert callable(spline_interp)

    def test_parsers(self):
        from rgpycrumbs.parsers.bless import BLESS_LOG, BLESS_TIME
        from rgpycrumbs.parsers.common import _NUM

        assert BLESS_LOG is not None
        assert callable(BLESS_TIME)
        assert _NUM is not None

    def test_eon_helpers(self):
        from rgpycrumbs.eon.helpers import write_eon_config

        assert callable(write_eon_config)
