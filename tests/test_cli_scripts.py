# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for CLI entry points and importable modules.

Tests that need cross-repo dev branches or conda-only deps (pypotlib,
ovito) are guarded with skipif. They run in the pixi_envs workspace
where all repos are editable installs.
"""

import importlib

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.pure


def _can_import(module_name):
    """Check if a module is importable without triggering full import chains."""
    try:
        importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


# Evaluate these once, catching any cascading import errors
_HAS_CHEMPARSEPLOT_NEB = _can_import("chemparseplot.plot.neb")
_HAS_DIMER_TRAJ = _can_import("chemparseplot.parse.eon.dimer_trajectory")
_HAS_CHEMGP = _can_import("chemparseplot.plot.chemgp")
_HAS_PYPOTLIB = _can_import("pypotlib")
_HAS_XTS_MB = _can_import("rgpycrumbs.xts.saddle.mb")


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


@pytest.mark.skipif(
    not _HAS_CHEMPARSEPLOT_NEB,
    reason="chemparseplot not installed",
)
class TestPltNebCLI:
    def _import_main(self):
        try:
            from rgpycrumbs.eon.plt_neb import main

            return main
        except ImportError:
            pytest.skip("plt_neb import failed (missing dep)")

    def test_help(self):
        main = self._import_main()
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--plot-type" in result.output


@pytest.mark.skipif(
    not _HAS_DIMER_TRAJ,
    reason="chemparseplot dev branch not installed",
)
class TestPltSaddleCLI:
    def _import_main(self):
        try:
            from rgpycrumbs.eon.plt_saddle import main

            return main
        except ImportError:
            pytest.skip("plt_saddle import failed")

    def test_help(self):
        main = self._import_main()
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--job-dir" in result.output

    def test_missing_job_dir(self):
        main = self._import_main()
        result = CliRunner().invoke(main, [])
        assert result.exit_code != 0


@pytest.mark.skipif(
    not _HAS_DIMER_TRAJ,
    reason="chemparseplot dev branch not installed",
)
class TestPltMinCLI:
    def _import_main(self):
        try:
            from rgpycrumbs.eon.plt_min import main

            return main
        except ImportError:
            pytest.skip("plt_min import failed")

    def test_help(self):
        main = self._import_main()
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_missing_job_dir(self):
        main = self._import_main()
        result = CliRunner().invoke(main, [])
        assert result.exit_code != 0


class TestGenerateNWChemCLI:
    def test_help(self):
        try:
            from rgpycrumbs.eon.generate_nwchem_input import main
        except ImportError:
            pytest.skip("generate_nwchem_input import failed")
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0


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
        try:
            from rgpycrumbs.chemgp import match_atoms

            assert hasattr(match_atoms, "match_atoms")
        except ImportError:
            pytest.skip("chemgp import chain failed")


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
