# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for CLI entry points and importable modules.

Tests that need cross-repo dev branches or heavyweight optional deps
(pypotlib, ovito) are guarded with skipif. They run in the pixi_envs workspace
where all repos are editable installs.
"""

import os
import importlib
import subprocess
import sys
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
    def test_help_routes_through_dispatcher(
        self, mock_run, argv, expected_script, monkeypatch
    ):
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

    @pytest.mark.parametrize(
        "rel_path",
        ["eon/plt_neb.py", "eon/plt_min.py", "eon/plt_saddle.py"],
    )
    def test_eon_plot_scripts_run_directly(self, rel_path):
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "rgpycrumbs" / rel_path
        chemparseplot_root = repo_root.parent / "chemparseplot"

        env = os.environ.copy()
        pythonpath = str(repo_root)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath = f"{pythonpath}:{chemparseplot_root}:{existing_pythonpath}"
        else:
            pythonpath = f"{pythonpath}:{chemparseplot_root}"
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stdout


class TestSharedRenderCli:
    def test_eon_plotters_share_render_option_contract(self):
        from rgpycrumbs.eon.plt_min import main as plt_min_main
        from rgpycrumbs.eon.plt_neb import main as plt_neb_main
        from rgpycrumbs.eon.plt_saddle import main as plt_saddle_main

        expected = {
            "strip_renderer",
            "xyzrender_config",
            "strip_spacing",
            "strip_dividers",
            "rotation",
            "perspective_tilt",
        }

        commands = (plt_min_main, plt_saddle_main, plt_neb_main)
        for command in commands:
            param_names = {param.name for param in command.params}
            assert expected.issubset(param_names)

    def test_single_ended_help_uses_dual_strip_divider_flag(self):
        from rgpycrumbs.eon.plt_min import main as plt_min_main
        from rgpycrumbs.eon.plt_saddle import main as plt_saddle_main

        for command in (plt_min_main, plt_saddle_main):
            result = CliRunner().invoke(command, ["--help"])
            assert result.exit_code == 0
            assert "--strip-dividers / --no-strip-dividers" in result.output


class TestSingleEndedPlotHelpers:
    def test_project_landscape_path_reuses_basis(self, monkeypatch):
        from rgpycrumbs.eon import _single_ended_plot as helper_mod

        calls = []

        def _fake_compute_projection_basis(*_args):
            calls.append("compute")
            return "basis"

        def _fake_project_to_sd(_a, _b, basis):
            calls.append(("project", basis))
            return [1.0, 2.0], [3.0, 4.0]

        monkeypatch.setattr(
            helper_mod, "compute_projection_basis", _fake_compute_projection_basis
        )
        monkeypatch.setattr(helper_mod, "project_to_sd", _fake_project_to_sd)

        x, y, basis = helper_mod.project_landscape_path(
            [0.0, 1.0], [1.0, 0.0], project_path=True
        )
        assert basis == "basis"
        assert calls == ["compute", ("project", "basis")]

        calls.clear()
        x2, y2, basis2 = helper_mod.project_landscape_path(
            [0.0, 1.0], [1.0, 0.0], project_path=True, basis="reused"
        )
        assert basis2 == "reused"
        assert calls == [("project", "reused")]
        assert x == x2 and y == y2

    def test_save_landscape_figure_skips_tight_layout_with_strip(
        self, monkeypatch, tmp_path
    ):
        from rgpycrumbs.eon import _single_ended_plot as helper_mod

        class _FakeFigure:
            def __init__(self):
                self.tight_layout_calls = 0
                self.saved = []

            def tight_layout(self):
                self.tight_layout_calls += 1

            def savefig(self, *args, **kwargs):
                self.saved.append((args, kwargs))

        fake_fig = _FakeFigure()
        monkeypatch.setattr(helper_mod.plt, "close", lambda _fig: None)

        helper_mod.save_landscape_figure(
            fake_fig, tmp_path / "strip.pdf", dpi=100, has_strip=True
        )
        assert fake_fig.tight_layout_calls == 0
        assert "bbox_inches" not in fake_fig.saved[0][1]

        fake_fig = _FakeFigure()
        monkeypatch.setattr(helper_mod.plt, "close", lambda _fig: None)
        helper_mod.save_landscape_figure(
            fake_fig, tmp_path / "plain.pdf", dpi=100, has_strip=False
        )
        assert fake_fig.tight_layout_calls == 1
        assert fake_fig.saved[0][1]["bbox_inches"] == "tight"

    def test_plot_single_ended_profile_handles_optional_eigen_column(
        self, monkeypatch, tmp_path
    ):
        import numpy as np

        from rgpycrumbs.eon import _single_ended_plot as helper_mod

        class _Column:
            def __init__(self, values):
                self._values = np.asarray(values)

            def to_numpy(self):
                return self._values

        class _Frame:
            def __init__(self, columns):
                self._columns = {key: _Column(val) for key, val in columns.items()}

            @property
            def columns(self):
                return list(self._columns)

            def __getitem__(self, key):
                return self._columns[key]

        trajs = [
            type(
                "Traj",
                (),
                {"dat_df": _Frame({"iteration": [0, 1], "delta_e": [0.0, 1.0]})},
            )(),
            type(
                "Traj",
                (),
                {
                    "dat_df": _Frame(
                        {
                            "iteration": [0, 1],
                            "delta_e": [0.0, 1.0],
                            "eigenvalue": [-1.0, -0.5],
                        }
                    )
                },
            )(),
        ]

        called = {}

        def _fake_save(fig, output, *, dpi):
            called["axes"] = len(fig.axes)

        monkeypatch.setattr(helper_mod, "save_standard_figure", _fake_save)
        helper_mod.plot_single_ended_profile(
            trajs,
            ["a", "b"],
            tmp_path / "profile.pdf",
            100,
            energy_unit="eV",
            energy_column="delta_e",
            title="Energy vs Iteration",
            eigen_column="eigenvalue",
        )
        assert called["axes"] == 2

    def test_plot_single_ended_convergence_adds_overlay_legend(
        self, monkeypatch, tmp_path
    ):
        from rgpycrumbs.eon import _single_ended_plot as helper_mod

        calls = []

        def _fake_panel(ax_force, ax_step, dat_df, *, color):
            calls.append(color)

        monkeypatch.setattr(helper_mod, "plot_convergence_panel", _fake_panel)
        monkeypatch.setattr(
            helper_mod, "save_standard_figure", lambda fig, output, *, dpi: None
        )

        trajs = [
            type("Traj", (), {"dat_df": object()})(),
            type("Traj", (), {"dat_df": object()})(),
        ]
        helper_mod.plot_single_ended_convergence(
            trajs, ["one", "two"], tmp_path / "conv.pdf", 100
        )
        assert len(calls) == 2


class TestSingleEndedCliHelpers:
    def test_default_output_path_prefers_explicit_value(self, tmp_path):
        from rgpycrumbs.eon._single_ended_cli import default_output_path

        explicit = tmp_path / "out.pdf"
        assert default_output_path("min", "profile", explicit) == explicit
        assert default_output_path("min", "profile", None) == Path("min_profile.pdf")

    def test_overlay_labels_pads_missing_entries(self, tmp_path):
        from rgpycrumbs.eon._single_ended_cli import overlay_labels

        job_dirs = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
        assert overlay_labels(job_dirs, []) == ["a", "b", "c"]
        assert overlay_labels(job_dirs, ["foo"]) == ["foo", "b", "c"]

    def test_load_trajectories_logs_consistently(self, tmp_path):
        from rgpycrumbs.eon._single_ended_cli import load_trajectories

        logged = []

        def _logger(message, *args):
            logged.append(message % args)

        result = load_trajectories(
            [tmp_path / "job1", tmp_path / "job2"],
            lambda path: {"path": path.name},
            log_info=_logger,
            noun="trajectory",
            detail=lambda traj: f"loaded {traj['path']}",
        )
        assert [item["path"] for item in result] == ["job1", "job2"]
        assert logged == [
            f"Loaded trajectory from {tmp_path / 'job1'} (loaded job1)",
            f"Loaded trajectory from {tmp_path / 'job2'} (loaded job2)",
        ]


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
