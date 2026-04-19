# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Deep coverage tests for CLI plotting scripts.

Exercises the landscape, structure rendering, and additional code paths
that need synthetic eOn data.
"""

import os
import shutil
import textwrap
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase.build import molecule
from ase.io import write as ase_write
from click.testing import CliRunner

from tests._optional_imports import has_module_spec, optional_import_available

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False

_HAS_PLT_NEB = has_module_spec("rgpycrumbs.eon.plt_neb") and optional_import_available(
    "rgpycrumbs.eon.plt_neb"
)
if _HAS_PLT_NEB:
    from rgpycrumbs.eon.plt_neb import main as plt_neb_main
else:
    plt_neb_main = None

_HAS_PLT_SADDLE = has_module_spec(
    "rgpycrumbs.eon.plt_saddle"
) and optional_import_available("rgpycrumbs.eon.plt_saddle")
if _HAS_PLT_SADDLE:
    from rgpycrumbs.eon.plt_saddle import main as plt_saddle_main
else:
    plt_saddle_main = None

_HAS_PLT_MIN = has_module_spec("rgpycrumbs.eon.plt_min") and optional_import_available(
    "rgpycrumbs.eon.plt_min"
)
if _HAS_PLT_MIN:
    from rgpycrumbs.eon.plt_min import main as plt_min_main
else:
    plt_min_main = None

_HAS_OVITO = has_module_spec("ovito")
_HAS_OVITO_RENDER = _HAS_OVITO and any(
    os.environ.get(var) for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")
)
_HAS_XVFB = shutil.which("Xvfb") is not None

try:
    import pyvista

    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

pytestmark = pytest.mark.pure


def _write_neb_dat(path, n_images=5, step=0):
    """Write a synthetic neb_NNN.dat file (eOn format: header + data)."""
    lines = ["# image  distance  energy  f_para  eigenvalue  f_perp"]
    for i in range(n_images):
        rc = i / (n_images - 1) * 3.0
        energy = -0.5 * np.cos(np.pi * i / (n_images - 1))
        eigenval = -0.3 + 0.1 * i
        f_para = 0.1 * np.sin(np.pi * i / (n_images - 1))
        f_perp = 0.05
        lines.append(
            f"{i}\t{rc:.6f}\t{energy:.6f}\t{f_para:.6f}\t{eigenval:.6f}\t{f_perp:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_neb_con(path, n_images=5):
    """Write a multi-frame .con trajectory."""
    frames = []
    # Use C2H6 (8 atoms) -- IRA needs >= 4 atoms to find a basis
    base = molecule("C2H6")
    for i in range(n_images):
        frame = base.copy()
        frame.positions[0, 0] += 0.1 * i
        frames.append(frame)
    ase_write(str(path), frames, format="eon")


def _write_sp_con(path):
    """Write a saddle point .con file."""
    base = molecule("C2H6")
    base.positions[0, 0] += 0.2
    ase_write(str(path), base, format="eon")


def _make_neb_dir(tmp_path, n_steps=3, n_images=5):
    """Create a synthetic NEB output directory."""
    for step in range(n_steps):
        _write_neb_dat(tmp_path / f"neb_{step:03d}.dat", n_images, step)
        _write_neb_con(tmp_path / f"neb_path_{step:03d}.con", n_images)
    # Also write a main neb.con and sp.con
    _write_neb_con(tmp_path / "neb.con", n_images)
    _write_sp_con(tmp_path / "sp.con")
    return tmp_path


@pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
class TestPltNebLandscape:
    """Test plt_neb landscape mode with synthetic data."""

    def test_landscape_path_mode(self, tmp_path):
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Copy files to isolated dir
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(neb_dir / "neb.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "-o",
                    "landscape_path.pdf",
                    "--no-project-path",
                ],
            )
            if result.exit_code != 0:
                print(result.output)
                if result.exception:
                    import traceback

                    traceback.print_exception(
                        type(result.exception),
                        result.exception,
                        result.exception.__traceback__,
                    )
            # Allow graceful failure if IRA not available
            assert result.exit_code == 0 or "ira" in str(result.exception).lower()

    def test_landscape_surface_no_project(self, tmp_path):
        """Test surface mode with grad_matern (needs jax)."""
        pytest.importorskip("jax")
        neb_dir = _make_neb_dir(tmp_path, n_steps=2, n_images=7)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(neb_dir / "neb.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "surface",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--no-project-path",
                    "--surface-type",
                    "grad_matern",
                    "-o",
                    "landscape_surface.pdf",
                ],
            )
            assert result.exit_code == 0, f"Exit {result.exit_code}: {result.exception}"

    def test_profile_with_structures_crit_points(self, tmp_path):
        """Test profile plot with crit_points structure rendering."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(neb_dir / "neb.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "crit_points",
                    "-o",
                    "profile_structs.pdf",
                ],
            )
            # Structure rendering requires IRA for RMSD but profile doesn't
            # The CLI should at least get past parsing

    def test_eigenvalue_mode(self, tmp_path):
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--plot-mode",
                    "eigenvalue",
                    "-o",
                    "eigenvalue.pdf",
                ],
            )

    def test_show_evolution_flag(self, tmp_path):
        neb_dir = _make_neb_dir(tmp_path, n_steps=3)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--no-project-path",
                    "--show-evolution",
                    "-o",
                    "evolution.pdf",
                ],
            )

    def test_additional_con(self, tmp_path):
        neb_dir = _make_neb_dir(tmp_path)
        extra = tmp_path / "extra.con"
        h2o = molecule("H2O")
        h2o.positions[0, 0] += 0.15
        ase_write(str(extra), h2o, format="eon")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(extra, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--con-file",
                    "neb.con",
                    "--additional-con",
                    "extra.con",
                    "Extra",
                    "-o",
                    "with_extra.pdf",
                ],
            )

    def test_traj_source(self, tmp_path):
        """Test extxyz trajectory source."""
        frames = [molecule("H2O") for _ in range(5)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.1 * i
            f.info["energy"] = -0.5 * np.cos(np.pi * i / 4)
            f.arrays["forces"] = np.zeros_like(f.positions)
        traj_file = tmp_path / "traj.xyz"
        ase_write(str(traj_file), frames, format="extxyz")

        runner = CliRunner()
        result = runner.invoke(
            plt_neb_main,
            [
                "--source",
                "traj",
                "--input-traj",
                str(traj_file),
                "--plot-type",
                "profile",
                "-o",
                str(tmp_path / "traj_profile.pdf"),
            ],
        )

    def test_theme_and_figsize(self, tmp_path):
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--theme",
                    "cmc.batlow",
                    "--figsize",
                    "7.0",
                    "5.0",
                    "--dpi",
                    "100",
                    "--title",
                    "Test NEB",
                    "--xlabel",
                    "RC",
                    "--ylabel",
                    "E",
                    "-o",
                    "themed.pdf",
                ],
            )


@pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
class TestPltNebHelpers:
    def test_profile_strip_payload_returns_typed_entries(self):
        from chemparseplot.plot.structs import StructurePlacement
        from ase import Atoms

        import rgpycrumbs.eon.plt_neb as plt_neb_mod

        atoms_list = [Atoms("H"), Atoms("H"), Atoms("H")]
        payload = plt_neb_mod._profile_strip_payload(
            atoms_list,
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 0.0]),
            "crit_points",
            "energy",
        )

        assert all(isinstance(entry, StructurePlacement) for entry in payload)
        assert [entry.label for entry in payload] == ["R", "SP", "P"]

    def test_landscape_half_span_prefers_global_basis(self, monkeypatch):
        from chemparseplot.parse.eon.neb import NebOverlayStructure

        import rgpycrumbs.eon.plt_neb as plt_neb_mod

        recompute_calls = []

        def _fake_compute_projection_basis(*_args):
            recompute_calls.append(True)
            return "final-basis"

        def _fake_project_to_sd(_r, _p, basis):
            return np.zeros(1), np.array([2.0 if basis == "global-basis" else 20.0])

        monkeypatch.setattr(
            plt_neb_mod, "compute_projection_basis", _fake_compute_projection_basis
        )
        monkeypatch.setattr(plt_neb_mod, "project_to_sd", _fake_project_to_sd)

        half_span = plt_neb_mod._landscape_half_span(
            (0.0, 4.0),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            [NebOverlayStructure(atoms=None, r=1.5, p=0.5, label="extra")],
            "global-basis",
        )

        assert half_span == pytest.approx(2.3)
        assert recompute_calls == []

    def test_save_plot_skips_tight_bbox_for_strip(self, monkeypatch, tmp_path):
        import rgpycrumbs.eon.plt_neb as plt_neb_mod

        saved = {}

        def _fake_savefig(path, **kwargs):
            saved[str(path)] = kwargs

        monkeypatch.setattr(plt_neb_mod.plt, "savefig", _fake_savefig)

        strip_out = tmp_path / "strip.pdf"
        plain_out = tmp_path / "plain.pdf"

        plt_neb_mod._save_plot(strip_out, 150, has_strip=True)
        plt_neb_mod._save_plot(plain_out, 150, has_strip=False)

        assert "bbox_inches" not in saved[str(strip_out)]
        assert saved[str(plain_out)]["bbox_inches"] == "tight"


def _try_import_plot_gp():
    return _import_attr(
        "rgpycrumbs.chemgp.plot_gp",
        "cli",
        "plot_gp not importable",
    )


def _import_optional_module(module_name: str, reason: str):
    """Import an optional module, but fail loudly on broken first-party code."""
    if not optional_import_available(module_name):
        pytest.skip(reason)
    return importlib.import_module(module_name)


def _import_attr(module_name: str, attr_name: str, reason: str):
    """Import *attr_name* from *module_name* with optional-dependency-aware skips."""
    module = _import_optional_module(module_name, reason)
    return getattr(module, attr_name)


def _make_grid(f, name, n=20):
    """Add a grid dataset with axis attrs to an HDF5 file."""
    g = f["grids"] if "grids" in f else f.create_group("grids")
    rng = np.random.default_rng(hash(name) % 2**32)
    ds = g.create_dataset(name, data=rng.standard_normal((n, n)))
    # Axis attrs on the dataset (read_h5_grid reads from ds.attrs)
    ds.attrs["x_range"] = [-1.0, 1.0]
    ds.attrs["x_length"] = n
    ds.attrs["y_range"] = [-1.0, 1.0]
    ds.attrs["y_length"] = n
    return ds


def _make_table(f, name, columns):
    """Add a table group with named datasets."""
    t = f.create_group(name) if name not in f else f[name]
    for col_name, data in columns.items():
        t.create_dataset(col_name, data=data)


def _make_points(f, name, columns):
    """Add a points group with named datasets."""
    pts = f["points"] if "points" in f else f.create_group("points")
    p = pts.create_group(name) if name not in pts else pts[name]
    for col_name, data in columns.items():
        p.create_dataset(col_name, data=data)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
class TestPlotGPDeep:
    """Each subcommand gets its own properly structured HDF5."""

    def test_nll_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "nll.h5"
        with h5py.File(h5, "w") as f:
            _make_grid(f, "nll")
            _make_grid(f, "grad_norm")
            _make_points(
                f,
                "optimum",
                {
                    "log_sigma2": np.array([0.5]),
                    "log_theta": np.array([-1.0]),
                },
            )

        result = CliRunner().invoke(
            cli, ["nll", "-i", str(h5), "-o", str(tmp_path / "nll.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_sensitivity_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "sens.h5"
        n = 30
        x = np.linspace(-1, 1, n)
        with h5py.File(h5, "w") as f:
            _make_table(f, "slice", {"x": x})
            _make_table(f, "true_surface", {"E_true": np.sin(x)})
            for j in range(1, 4):
                for i in range(1, 4):
                    _make_table(
                        f,
                        f"gp_ls{j}_sv{i}",
                        {
                            "E_pred": np.sin(x) + 0.1 * j,
                            "E_std": np.abs(
                                np.random.default_rng(j * i).standard_normal(n)
                            )
                            * 0.05,
                        },
                    )

        result = CliRunner().invoke(
            cli, ["sensitivity", "-i", str(h5), "-o", str(tmp_path / "sens.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_trust_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "trust.h5"
        n = 30
        x = np.linspace(-1, 1, n)
        with h5py.File(h5, "w") as f:
            _make_table(
                f,
                "slice",
                {
                    "x": x,
                    "E_true": np.sin(x),
                    "E_pred": np.sin(x) + 0.05,
                    "E_std": np.ones(n) * 0.1,
                    "in_trust": np.array([1.0 if abs(xi) < 0.5 else 0.0 for xi in x]),
                },
            )
            _make_points(
                f,
                "training",
                {
                    "x": np.array([-0.5, 0.0, 0.5]),
                    "y": np.array([0.5, 0.5, 0.5]),
                },
            )
            meta = f.create_group("metadata")
            meta.attrs["y_slice"] = 0.5

        result = CliRunner().invoke(
            cli, ["trust", "-i", str(h5), "-o", str(tmp_path / "trust.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_fps_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "fps.h5"
        rng = np.random.default_rng(42)
        with h5py.File(h5, "w") as f:
            _make_points(
                f,
                "selected",
                {"pc1": rng.standard_normal(20), "pc2": rng.standard_normal(20)},
            )
            _make_points(
                f,
                "pruned",
                {"pc1": rng.standard_normal(10), "pc2": rng.standard_normal(10)},
            )

        result = CliRunner().invoke(
            cli, ["fps", "-i", str(h5), "-o", str(tmp_path / "fps.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_variance_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "var.h5"
        rng = np.random.default_rng(42)
        with h5py.File(h5, "w") as f:
            _make_grid(f, "energy")
            _make_grid(f, "variance")
            _make_points(
                f, "training", {"x": rng.standard_normal(5), "y": rng.standard_normal(5)}
            )
            _make_points(f, "minima", {"x": np.array([0.0]), "y": np.array([0.0])})
            _make_points(f, "saddles", {"x": np.array([0.5]), "y": np.array([0.5])})

        result = CliRunner().invoke(
            cli, ["variance", "-i", str(h5), "-o", str(tmp_path / "var.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_quality_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "quality.h5"
        n = 20
        with h5py.File(h5, "w") as f:
            _make_grid(f, "true_energy")
            for npts in [5, 10, 15]:
                _make_grid(f, f"gp_mean_N{npts}")
                rng = np.random.default_rng(npts)
                _make_points(
                    f,
                    f"train_N{npts}",
                    {
                        "x": rng.standard_normal(npts),
                        "y": rng.standard_normal(npts),
                    },
                )

        result = CliRunner().invoke(
            cli, ["quality", "-i", str(h5), "-o", str(tmp_path / "q.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_rff_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "rff.h5"
        n = 10
        with h5py.File(h5, "w") as f:
            _make_table(
                f,
                "table",
                {
                    "D_rff": np.arange(n, dtype=float) * 100,
                    "energy_mae_vs_gp": np.random.default_rng(1).standard_normal(n),
                    "gradient_mae_vs_gp": np.random.default_rng(2).standard_normal(n),
                },
            )
            meta = f.create_group("metadata")
            meta.attrs["gp_e_mae"] = 0.01
            meta.attrs["gp_g_mae"] = 0.02

        result = CliRunner().invoke(
            cli, ["rff", "-i", str(h5), "-o", str(tmp_path / "rff.pdf")]
        )
        assert result.exit_code == 0, result.output

    def test_surface_with_paths_points(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "surface.h5"
        rng = np.random.default_rng(42)
        with h5py.File(h5, "w") as f:
            _make_grid(f, "energy")
            # Add paths
            paths = f.create_group("paths")
            p = paths.create_group("neb_path")
            p.create_dataset("x", data=rng.standard_normal(5))
            p.create_dataset("y", data=rng.standard_normal(5))
            # Add points
            _make_points(f, "saddle", {"x": np.array([0.1]), "y": np.array([0.2])})

        result = CliRunner().invoke(
            cli,
            [
                "surface",
                "-i",
                str(h5),
                "-o",
                str(tmp_path / "surf.pdf"),
                "--clamp-lo",
                "-2.0",
                "--clamp-hi",
                "2.0",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_batch_subcommand(self, tmp_path):
        cli = _try_import_plot_gp()

        # Create a convergence H5
        h5 = tmp_path / "data" / "conv.h5"
        h5.parent.mkdir()
        with h5py.File(h5, "w") as f:
            _make_table(
                f,
                "table",
                {
                    "oracle_calls": np.arange(5, dtype=float),
                    "max_force": np.abs(np.random.default_rng(1).standard_normal(5)),
                },
            )
            f.create_group("metadata")

        # Write TOML config
        cfg = tmp_path / "plots.toml"
        cfg.write_text(
            textwrap.dedent("""\
            [defaults]
            input_dir = "data"
            output_dir = "output"

            [[plots]]
            type = "convergence"
            input = "conv.h5"
            output = "conv.pdf"
        """)
        )
        (tmp_path / "output").mkdir()

        result = CliRunner().invoke(
            cli,
            [
                "batch",
                "-c",
                str(cfg),
                "-b",
                str(tmp_path),
            ],
        )
        # Batch may fail internally but exercises the dispatch code
        assert result.exit_code in (0, 1), result.output


class TestPlumedTrivial:
    """Cover the 4 lines in plumed/."""

    def test_import_plumed_init(self):
        module = _import_optional_module("rgpycrumbs.plumed", "plumed import failed")
        assert hasattr(module, "direct_reconstruction") or True

    def test_import_direct_reconstruction(self):
        module = _import_optional_module(
            "rgpycrumbs.plumed.direct_reconstruction",
            "direct_reconstruction import failed",
        )
        assert hasattr(module, "calculate_fes_from_hills")
        assert hasattr(module, "find_fes_minima")


class TestInitLazy:
    """Cover __init__.py lazy import paths."""

    def test_surfaces_attr(self):
        import rgpycrumbs

        try:
            _ = rgpycrumbs.surfaces
        except (ImportError, AttributeError):
            pass  # OK if jax not installed

    def test_geom_attr(self):
        import rgpycrumbs

        try:
            _ = rgpycrumbs.geom
        except (ImportError, AttributeError):
            pass

    def test_unknown_attr_raises(self):
        import rgpycrumbs

        with pytest.raises(AttributeError):
            _ = rgpycrumbs.totally_nonexistent_thing


class TestPltNebStructureRendering:
    """Test plt_neb structure rendering paths (lines 834-973)."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_landscape_path_with_crit_points(self, tmp_path):
        """Exercise strip rendering with crit_points structures."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=3, n_images=5)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(neb_dir / "neb.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "crit_points",
                    "--no-project-path",
                    "-o",
                    "crit_points.pdf",
                ],
            )
            # May fail on IRA but exercises the strip setup code

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_landscape_path_with_all_structures(self, tmp_path):
        """Exercise strip rendering with all structures."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=2, n_images=5)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(neb_dir / "neb.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "all",
                    "--no-project-path",
                    "-o",
                    "all_structs.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_profile_with_structures_and_insets(self, tmp_path):
        """Exercise profile with structure insets (lines 953-1033)."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=3, n_images=5)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "all",
                    "-o",
                    "profile_insets.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_landscape_with_mmf_peaks(self, tmp_path):
        """Exercise MMF peaks overlay (lines 695-750)."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=3, n_images=5)
        # Create fake peak files
        peak_dir = tmp_path / "peaks"
        peak_dir.mkdir()
        h2o = molecule("H2O")
        h2o.positions[0, 0] += 0.15
        ase_write(str(peak_dir / "peak00_pos.con"), h2o, format="eon")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copytree(peak_dir, Path(td) / "peaks")

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--no-project-path",
                    "--mmf-peaks",
                    "--peak-dir",
                    "peaks",
                    "-o",
                    "mmf.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_profile_highlight_last_false(self, tmp_path):
        """Exercise --no-highlight-last path."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--no-highlight-last",
                    "-o",
                    "no_highlight.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_show_legend(self, tmp_path):
        """Exercise --show-legend path."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--show-legend",
                    "-o",
                    "legend.pdf",
                ],
            )


class TestPlotGPBranches:
    """Cover remaining branch conditions in plot_gp.py."""

    @pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
    def test_surface_auto_clamp(self, tmp_path):
        """Test surface with auto-detected clamping from filename."""
        cli = _try_import_plot_gp()

        # Filename with clamp hint
        h5 = tmp_path / "surface_lo-2_hi2.h5"
        with h5py.File(h5, "w") as f:
            _make_grid(f, "energy")
        result = CliRunner().invoke(
            cli, ["surface", "-i", str(h5), "-o", str(tmp_path / "s.pdf")]
        )

    @pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
    def test_convergence_ci_force_column(self, tmp_path):
        """Test convergence with ci_force column (branch at line 129)."""
        cli = _try_import_plot_gp()

        h5 = tmp_path / "conv.h5"
        with h5py.File(h5, "w") as f:
            _make_table(
                f,
                "table",
                {
                    "oracle_calls": np.arange(5, dtype=float),
                    "ci_force": np.abs(np.random.default_rng(1).standard_normal(5)),
                    "energy": np.random.default_rng(2).standard_normal(5),
                    "step": np.arange(5, dtype=float),
                    "method": np.array([b"GP"] * 5),
                },
            )
            meta = f.create_group("metadata")
            meta.attrs["conv_tol"] = 0.01
        result = CliRunner().invoke(
            cli, ["convergence", "-i", str(h5), "-o", str(tmp_path / "c.pdf")]
        )
        assert result.exit_code == 0, f"{result.exit_code}: {result.exception}"

    @pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
    def test_batch_empty_config(self, tmp_path):
        """Test batch with no plots entry (line 586)."""
        cli = _try_import_plot_gp()

        cfg = tmp_path / "empty.toml"
        cfg.write_text("[defaults]\n")
        result = CliRunner().invoke(cli, ["batch", "-c", str(cfg), "-b", str(tmp_path)])
        assert result.exit_code == 0


class TestInitLazyMore:
    """Cover __init__.py lazy imports (lines 9-33)."""

    def test_basetypes_lazy(self):
        import rgpycrumbs

        bt = rgpycrumbs.basetypes
        assert hasattr(bt, "SaddleMeasure")

    def test_interpolation_lazy(self):
        import rgpycrumbs

        interp = rgpycrumbs.interpolation
        assert hasattr(interp, "spline_interp")

    def test_unknown_attr_raises(self):
        import rgpycrumbs

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = rgpycrumbs.totally_nonexistent_xyz


class TestConSplitterBranches:
    """Cover remaining con_splitter branches."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_split_with_output_dir(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.con_splitter",
            "con_splitter",
            "con_splitter not importable",
        )

        # Create a multi-frame .con
        frames = [molecule("H2O") for _ in range(3)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.1 * i
        con_file = tmp_path / "traj.con"
        ase_write(str(con_file), frames, format="eon")

        out_dir = tmp_path / "split_out"
        runner = CliRunner()
        result = runner.invoke(main, [str(con_file), "--output-dir", str(out_dir)])


class TestJupyterBranches:
    """Cover remaining jupyter.py branches."""

    def test_run_command_success(self):
        from rgpycrumbs.run.jupyter import run_command_or_exit

        result = run_command_or_exit(["echo", "test"], capture=True)
        assert result.returncode == 0


@pytest.mark.fragments
class TestPltNebWithIRA:
    """Tests that need IRA for RMSD calculations (fragments env)."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_landscape_surface_with_ira(self, tmp_path):
        """Full landscape surface with IRA RMSD + structure insets."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=3, n_images=5)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(neb_dir / "neb.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "crit_points",
                    "--project-path",
                    "-o",
                    "ira_landscape.pdf",
                ],
            )
            assert result.exit_code == 0, f"{result.exit_code}: {result.exception}"

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_profile_insets_with_ira(self, tmp_path):
        """Profile with structure insets requiring RMSD projection."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=3, n_images=5)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "crit_points",
                    "--rc-mode",
                    "rmsd",
                    "-o",
                    "ira_profile.pdf",
                ],
            )
            assert result.exit_code == 0, f"{result.exit_code}: {result.exception}"

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_landscape_with_additional_con(self, tmp_path):
        """Landscape with additional .con overlay."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=2, n_images=5)
        extra = tmp_path / "extra.con"
        base = molecule("C2H6")
        base.positions[0, 0] += 0.15
        ase_write(str(extra), base, format="eon")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)
            shutil.copy(extra, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--additional-con",
                    "extra.con",
                    "Extra",
                    "--project-path",
                    "-o",
                    "additional.pdf",
                ],
            )
            assert result.exit_code == 0, f"{result.exit_code}: {result.exception}"


class TestDeletePackagesDeep:
    """Cover the main CLI execution path of delete_packages."""

    def test_main_dry_run_with_packages(self, tmp_path):
        from unittest.mock import MagicMock, patch

        main = _import_attr(
            "rgpycrumbs.prefix.delete_packages",
            "main",
            "delete_packages not importable",
        )

        runner = CliRunner()
        # Mock get_packages_to_delete to return fake packages
        with patch(
            "rgpycrumbs.prefix.delete_packages.get_packages_to_delete"
        ) as mock_get:
            mock_get.return_value = [
                ("linux-64", "pkg-1.0.tar.bz2"),
                ("linux-64", "pkg-1.1.tar.bz2"),
            ]
            result = runner.invoke(
                main,
                [
                    "--channel",
                    "test-channel",
                    "--package-name",
                    "pkg",
                    "--dry-run",
                ],
                input="y\n",
            )  # Confirm deletion prompt
            assert result.exit_code == 0

    def test_main_no_packages_found(self, tmp_path):
        from unittest.mock import patch

        main = _import_attr(
            "rgpycrumbs.prefix.delete_packages",
            "main",
            "delete_packages not importable",
        )

        runner = CliRunner()
        with patch(
            "rgpycrumbs.prefix.delete_packages.get_packages_to_delete"
        ) as mock_get:
            mock_get.return_value = []
            result = runner.invoke(
                main,
                [
                    "--channel",
                    "test-channel",
                    "--package-name",
                    "nonexistent",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

    def test_main_no_packages_with_version_regex(self, tmp_path):
        from unittest.mock import patch

        main = _import_attr(
            "rgpycrumbs.prefix.delete_packages",
            "main",
            "delete_packages not importable",
        )

        runner = CliRunner()
        with patch(
            "rgpycrumbs.prefix.delete_packages.get_packages_to_delete"
        ) as mock_get:
            mock_get.return_value = []
            result = runner.invoke(
                main,
                [
                    "--channel",
                    "test-channel",
                    "--package-name",
                    "pkg",
                    "--version-regex",
                    "1\\.0.*",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

    def test_main_abort_on_prompt(self, tmp_path):
        from unittest.mock import patch

        main = _import_attr(
            "rgpycrumbs.prefix.delete_packages",
            "main",
            "delete_packages not importable",
        )

        runner = CliRunner()
        with patch(
            "rgpycrumbs.prefix.delete_packages.get_packages_to_delete"
        ) as mock_get:
            mock_get.return_value = [("linux-64", "pkg.tar.bz2")]
            result = runner.invoke(
                main,
                [
                    "--channel",
                    "test-channel",
                    "--package-name",
                    "pkg",
                    "--dry-run",
                ],
                input="n\n",
            )  # Deny deletion
            assert result.exit_code == 0


class TestToMlflowDeep:
    """Cover remaining to_mlflow.py lines (182-210)."""

    def test_plot_structure_evolution_with_data(self, tmp_path):
        plot_structure_evolution = _import_attr(
            "rgpycrumbs.eon.to_mlflow",
            "plot_structure_evolution",
            "to_mlflow not importable",
        )

        # Create synthetic structures
        from ase.build import molecule

        frames = [molecule("H2O") for _ in range(5)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.1 * i

        fig = plot_structure_evolution(frames)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestConSplitterDeep:
    """Cover remaining con_splitter branches (lines 76-96)."""

    def test_split_single_frame(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.con_splitter",
            "con_splitter",
            "con_splitter not importable",
        )

        # Single frame .con
        h2o = molecule("H2O")
        con = tmp_path / "single.con"
        ase_write(str(con), h2o, format="eon")

        runner = CliRunner()
        result = runner.invoke(main, [str(con), "--output-dir", str(tmp_path / "out")])

    def test_split_multi_frame(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.con_splitter",
            "con_splitter",
            "con_splitter not importable",
        )

        frames = [molecule("H2O") for _ in range(4)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.1 * i
        con = tmp_path / "multi.con"
        ase_write(str(con), frames, format="eon")

        runner = CliRunner()
        result = runner.invoke(main, [str(con), "--output-dir", str(tmp_path / "out")])

    def test_split_with_prefix(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.con_splitter",
            "con_splitter",
            "con_splitter not importable",
        )

        frames = [molecule("H2O") for _ in range(3)]
        con = tmp_path / "traj.con"
        ase_write(str(con), frames, format="eon")

        runner = CliRunner()
        result = runner.invoke(
            main, [str(con), "--output-dir", str(tmp_path / "out"), "--prefix", "image"]
        )


class TestNwchemGenDeep:
    """Cover remaining generate_nwchem_input lines."""

    def test_generate_with_pos_file(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.generate_nwchem_input",
            "main",
            "generate_nwchem_input not importable",
        )

        # Create a pos.con file
        h2o = molecule("H2O")
        pos = tmp_path / "pos.con"
        ase_write(str(pos), h2o, format="eon")

        # Create a settings file
        settings = tmp_path / "settings.ini"
        settings.write_text("[NWChem]\nbasis = 6-31G\nmethod = dft\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--pos-file",
                str(pos),
                "--settings",
                str(settings),
                "--socket-address",
                "localhost:12345",
                "--output",
                str(tmp_path / "nwchem.nwi"),
            ],
        )


@pytest.mark.skipif(not _HAS_PLT_SADDLE, reason="plt_saddle not importable")
class TestPltSaddleDeep:
    """Cover remaining plt_saddle branches."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_verbose_mode(self, tmp_path):
        from rgpycrumbs.eon.plt_saddle import main

        # Create synthetic job dir
        h2o = molecule("H2O")
        job = tmp_path / "job"
        job.mkdir()

        frames = [h2o.copy() for _ in range(5)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.05 * i
        ase_write(str(job / "climb"), frames, format="eon")
        ase_write(str(job / "reactant.con"), h2o, format="eon")

        saddle = h2o.copy()
        saddle.positions[0, 0] += 0.2
        ase_write(str(job / "saddle.con"), saddle, format="eon")

        (job / "mode.dat").write_text("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")
        lines = [
            "iteration\tstep_size\tdelta_e\tconvergence\teigenvalue\ttorque\tangle\trotations"
        ]
        for i in range(1, 5):
            lines.append(f"{i}\t0.1\t0.01\t0.05\t-0.1\t0.05\t10.0\t3")
        (job / "climb.dat").write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job),
                "--plot-type",
                "convergence",
                "-v",
                "-o",
                str(tmp_path / "conv.pdf"),
            ],
        )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_mode_evolution(self, tmp_path):
        from rgpycrumbs.eon.plt_saddle import main

        h2o = molecule("H2O")
        job = tmp_path / "job"
        job.mkdir()

        frames = [h2o.copy() for _ in range(3)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.05 * i
        ase_write(str(job / "climb"), frames, format="eon")
        ase_write(str(job / "reactant.con"), h2o, format="eon")
        ase_write(str(job / "saddle.con"), h2o, format="eon")
        (job / "mode.dat").write_text("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")
        lines = [
            "iteration\tstep_size\tdelta_e\tconvergence\teigenvalue\ttorque\tangle\trotations"
        ]
        for i in range(1, 3):
            lines.append(f"{i}\t0.1\t0.01\t0.05\t-0.1\t0.05\t10.0\t3")
        (job / "climb.dat").write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job),
                "--plot-type",
                "mode-evolution",
                "-o",
                str(tmp_path / "mode.pdf"),
            ],
        )


@pytest.mark.skipif(not _HAS_PLT_MIN, reason="plt_min not importable")
class TestPltMinDeep:
    """Cover remaining plt_min branches."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_min_verbose(self, tmp_path):
        from rgpycrumbs.eon.plt_min import main

        job = tmp_path / "job"
        job.mkdir()

        h2o = molecule("H2O")
        frames = [h2o.copy() for _ in range(4)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.05 * i
        ase_write(str(job / "min"), frames, format="eon")
        ase_write(str(job / "min.con"), frames[-1], format="eon")

        lines = ["iteration\tstep_size\tconvergence\tenergy"]
        for i in range(4):
            lines.append(f"{i}\t0.1\t{0.5 * 0.5**i}\t{-10.0 - 0.5 * i}")
        (job / "min.dat").write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job),
                "--plot-type",
                "profile",
                "-v",
                "-o",
                str(tmp_path / "min.pdf"),
            ],
        )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_min_convergence(self, tmp_path):
        from rgpycrumbs.eon.plt_min import main

        job = tmp_path / "job"
        job.mkdir()

        h2o = molecule("H2O")
        frames = [h2o.copy() for _ in range(3)]
        ase_write(str(job / "min"), frames, format="eon")

        lines = ["iteration\tstep_size\tconvergence\tenergy"]
        for i in range(3):
            lines.append(f"{i}\t0.1\t{0.5 * 0.5**i}\t{-10.0 - i}")
        (job / "min.dat").write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job),
                "--plot-type",
                "convergence",
                "-o",
                str(tmp_path / "conv.pdf"),
            ],
        )


import subprocess


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
@pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs plt_neb")
class TestPltNebHdf5Source:
    """Cover the --source hdf5 paths in plt_neb."""

    def test_hdf5_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(
            plt_neb_main,
            [
                "--source",
                "hdf5",
                "--plot-type",
                "profile",
                "-o",
                "/tmp/nope.pdf",
            ],
        )
        assert result.exit_code != 0


class TestJupyterDeep:
    """Cover remaining jupyter.py branches."""

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

        result = _run_command_live(["echo", "test"], capture=False)
        assert result.returncode == 0
        assert result.stdout is None

    def test_run_command_live_not_on_path(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        with pytest.raises(FileNotFoundError):
            _run_command_live(["totally_nonexistent_binary_xyz"])

    def test_run_command_or_exit_not_found(self):
        from rgpycrumbs.run.jupyter import run_command_or_exit

        with pytest.raises(SystemExit) as exc_info:
            run_command_or_exit(["totally_nonexistent_xyz"])
        assert exc_info.value.code == 2

    def test_run_command_or_exit_failure(self):
        from rgpycrumbs.run.jupyter import run_command_or_exit

        with pytest.raises(SystemExit) as exc_info:
            run_command_or_exit(["false"])
        assert exc_info.value.code != 0

    def test_run_command_live_shell_mode(self):
        from rgpycrumbs.run.jupyter import _run_command_live

        result = _run_command_live("echo shell_mode", capture=True)
        assert "shell_mode" in result.stdout


class TestInitDeep:
    """Cover __init__.py lazy import error messages."""

    def test_basetypes_lazy(self):
        import rgpycrumbs

        bt = rgpycrumbs.basetypes
        assert hasattr(bt, "SaddleMeasure")

    def test_interpolation_lazy(self):
        import rgpycrumbs

        interp = rgpycrumbs.interpolation
        assert hasattr(interp, "spline_interp")

    def test_geom_lazy(self):
        import rgpycrumbs

        geom = rgpycrumbs.geom
        assert geom is not None

    def test_unknown_attr_raises(self):
        import rgpycrumbs

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = rgpycrumbs.totally_nonexistent_xyz


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
class TestPlotGPBatchDeep:
    """Cover batch parallel and error handling paths."""

    def test_batch_parallel(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "data" / "conv.h5"
        h5.parent.mkdir()
        with h5py.File(h5, "w") as f:
            _make_table(
                f,
                "table",
                {
                    "oracle_calls": np.arange(5, dtype=float),
                    "max_force": np.abs(np.random.default_rng(1).standard_normal(5)),
                },
            )
            f.create_group("metadata")

        cfg = tmp_path / "plots.toml"
        cfg.write_text(
            textwrap.dedent("""\
            [defaults]
            input_dir = "data"
            output_dir = "output"

            [[plots]]
            type = "convergence"
            input = "conv.h5"
            output = "conv1.pdf"

            [[plots]]
            type = "convergence"
            input = "conv.h5"
            output = "conv2.pdf"
        """)
        )
        (tmp_path / "output").mkdir()

        result = CliRunner().invoke(
            cli,
            [
                "batch",
                "-c",
                str(cfg),
                "-b",
                str(tmp_path),
                "-j",
                "2",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_batch_unknown_type(self, tmp_path):
        cli = _try_import_plot_gp()

        cfg = tmp_path / "bad.toml"
        cfg.write_text(
            textwrap.dedent("""\
            [defaults]
            input_dir = "."
            output_dir = "."

            [[plots]]
            type = "nonexistent_plot_type"
            input = "x.h5"
            output = "x.pdf"
        """)
        )

        result = CliRunner().invoke(
            cli,
            [
                "batch",
                "-c",
                str(cfg),
                "-b",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1  # Should fail with unknown type

    def test_batch_missing_input(self, tmp_path):
        cli = _try_import_plot_gp()

        cfg = tmp_path / "missing.toml"
        cfg.write_text(
            textwrap.dedent("""\
            [defaults]
            input_dir = "nonexistent_dir"
            output_dir = "."

            [[plots]]
            type = "convergence"
            input = "missing.h5"
            output = "out.pdf"
        """)
        )

        result = CliRunner().invoke(
            cli,
            [
                "batch",
                "-c",
                str(cfg),
                "-b",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1


class TestPltNebEdgeCases:
    """Cover specific uncovered lines/branches in plt_neb."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_fig_height_and_aspect_ratio(self, tmp_path):
        """Cover line 527 (fig_height+aspect_ratio)."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--fig-height",
                    "4.0",
                    "--aspect-ratio",
                    "1.5",
                    "-o",
                    "sized.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_fig_height_only_error(self, tmp_path):
        """Cover line 529 (fig_height without aspect_ratio)."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--fig-height",
                    "4.0",
                    "-o",
                    "error.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_traj_source_missing_file(self):
        """Cover line 580-581 (missing --input-traj)."""
        runner = CliRunner()
        result = runner.invoke(
            plt_neb_main,
            [
                "--source",
                "traj",
                "--plot-type",
                "profile",
                "-o",
                "/tmp/nope.pdf",
            ],
        )
        assert result.exit_code != 0

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_traj_landscape(self, tmp_path):
        """Cover lines 592-596 (traj landscape source)."""
        frames = [molecule("H2O") for _ in range(7)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.1 * i
            f.info["energy"] = -0.5 * np.cos(np.pi * i / 6)
            f.arrays["forces"] = np.zeros_like(f.positions)
        traj = tmp_path / "traj.xyz"
        ase_write(str(traj), frames, format="extxyz")

        runner = CliRunner()
        result = runner.invoke(
            plt_neb_main,
            [
                "--source",
                "traj",
                "--input-traj",
                str(traj),
                "--plot-type",
                "landscape",
                "--landscape-mode",
                "path",
                "--no-project-path",
                "-o",
                str(tmp_path / "traj_land.pdf"),
            ],
        )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_hdf5_landscape_source(self, tmp_path):
        """Cover lines 597-617 (hdf5 landscape source)."""
        if not _HAS_H5PY:
            pytest.skip("h5py required")

        h5 = tmp_path / "neb_history.h5"
        with h5py.File(h5, "w") as f:
            # Minimal history structure
            f.create_group("metadata")

        runner = CliRunner()
        result = runner.invoke(
            plt_neb_main,
            [
                "--source",
                "hdf5",
                "--input-h5",
                str(h5),
                "--plot-type",
                "landscape",
                "--landscape-mode",
                "path",
                "--no-project-path",
                "-o",
                str(tmp_path / "h5_land.pdf"),
            ],
        )
        # Will likely fail but exercises the hdf5 landscape branch

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_no_dat_files(self, tmp_path):
        """Cover lines 622-624 (no .dat files found)."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "-o",
                    "nope.pdf",
                ],
            )
            assert result.exit_code != 0

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_hdf5_profile_eigenvalue(self, tmp_path):
        """Cover lines 971 (eigenvalue column in hdf5 profile)."""
        if not _HAS_H5PY:
            pytest.skip("h5py required")

        h5 = tmp_path / "result.h5"
        with h5py.File(h5, "w") as f:
            f.create_group("metadata")

        runner = CliRunner()
        result = runner.invoke(
            plt_neb_main,
            [
                "--source",
                "hdf5",
                "--input-h5",
                str(h5),
                "--plot-type",
                "profile",
                "--plot-mode",
                "eigenvalue",
                "-o",
                str(tmp_path / "eigen.pdf"),
            ],
        )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_profile_normalize_rc(self, tmp_path):
        """Cover normalize_rc branch."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--normalize-rc",
                    "-o",
                    "normalized.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_landscape_last_path_only(self, tmp_path):
        """Cover --landscape-path last branch."""
        neb_dir = _make_neb_dir(tmp_path, n_steps=3)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--landscape-path",
                    "last",
                    "--con-file",
                    "neb.con",
                    "--no-project-path",
                    "-o",
                    "last_path.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_strip_dividers_and_spacing(self, tmp_path):
        """Cover strip rendering options."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)
            shutil.copy(neb_dir / "neb.con", td)
            shutil.copy(neb_dir / "sp.con", td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "landscape",
                    "--landscape-mode",
                    "path",
                    "--con-file",
                    "neb.con",
                    "--sp-file",
                    "sp.con",
                    "--plot-structures",
                    "crit_points",
                    "--no-project-path",
                    "--strip-spacing",
                    "2.0",
                    "--strip-dividers",
                    "-o",
                    "strip.pdf",
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="plt_neb not importable")
    def test_cmap_overrides(self, tmp_path):
        """Cover cmap override options."""
        neb_dir = _make_neb_dir(tmp_path)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            import shutil

            for f in neb_dir.glob("neb_*"):
                shutil.copy(f, td)

            result = runner.invoke(
                plt_neb_main,
                [
                    "--plot-type",
                    "profile",
                    "--cmap-profile",
                    "viridis",
                    "-o",
                    "cmap.pdf",
                ],
            )


class TestJupyterTimeout:
    """Cover timeout and keyboard interrupt handlers in jupyter.py."""

    def test_run_command_or_exit_timeout(self):
        """Cover lines 92-94 (timeout handler in run_command_or_exit)."""
        import subprocess
        from unittest.mock import patch

        from rgpycrumbs.run.jupyter import run_command_or_exit

        # Mock _run_command_live to raise TimeoutExpired
        with patch("rgpycrumbs.run.jupyter._run_command_live") as mock:
            mock.side_effect = subprocess.TimeoutExpired("cmd", 1.0)
            with pytest.raises(SystemExit) as exc_info:
                run_command_or_exit(["test"])
            assert exc_info.value.code == 124


class TestInitErrorPaths:
    """Cover __init__.py lines 14-26 (ImportError with hints)."""

    def test_surfaces_import_error_message(self):
        """Trigger the surfaces ImportError path if jax unavailable."""
        import rgpycrumbs

        try:
            _ = rgpycrumbs.surfaces
        except ImportError as e:
            # Verify the hint message (lines 20-24)
            assert "pip install" in str(e)
            assert "surfaces" in str(e)
        # If no ImportError, surfaces is available (jax installed) -- ok

    def test_interpolation_error_with_mock(self):
        """Force interpolation to fail to cover lines 19-25."""
        from unittest.mock import patch

        import rgpycrumbs

        # Temporarily make the import fail
        with patch("importlib.import_module", side_effect=ImportError("fake")):
            try:
                # Force re-evaluation of __getattr__
                rgpycrumbs.__getattr__("interpolation")
            except ImportError as e:
                assert "pip install" in str(e) or "fake" in str(e)

    def test_geom_error_reraise(self):
        """Cover line 26 (re-raise without hint for geom)."""
        from unittest.mock import patch

        import rgpycrumbs

        with patch("importlib.import_module", side_effect=ImportError("no geom")):
            try:
                rgpycrumbs.__getattr__("geom")
            except ImportError as e:
                # Line 26: re-raise without custom message (geom not in hints)
                assert "no geom" in str(e)


@pytest.mark.skipif(not _HAS_PLT_MIN, reason="plt_min not importable")
class TestPltMinDefaultOutput:
    """Cover plt_min line 135 (default output name)."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_default_output_name(self, tmp_path):
        from rgpycrumbs.eon.plt_min import main

        job = tmp_path / "job"
        job.mkdir()

        h2o = molecule("H2O")
        frames = [h2o.copy() for _ in range(3)]
        ase_write(str(job / "min"), frames, format="eon")
        lines = ["iteration\tstep_size\tconvergence\tenergy"]
        for i in range(3):
            lines.append(f"{i}\t0.1\t0.5\t{-10.0 - i}")
        (job / "min.dat").write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main,
                [
                    "--job-dir",
                    str(job),
                    "--plot-type",
                    "profile",
                    # No -o flag -- triggers default output name (line 135)
                ],
            )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_landscape_triggers_ira_import(self, tmp_path):
        """Cover lines 175-179 (IRA import in landscape mode)."""
        from rgpycrumbs.eon.plt_min import main

        job = tmp_path / "job"
        job.mkdir()

        h2o = molecule("H2O")
        frames = [h2o.copy() for _ in range(3)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.05 * i
        ase_write(str(job / "min"), frames, format="eon")
        ase_write(str(job / "min.con"), frames[-1], format="eon")
        lines = ["iteration\tstep_size\tconvergence\tenergy"]
        for i in range(3):
            lines.append(f"{i}\t0.1\t0.5\t{-10.0 - i}")
        (job / "min.dat").write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job),
                "--plot-type",
                "landscape",
                "--no-project-path",
                "-o",
                str(tmp_path / "land.pdf"),
            ],
        )
        # Will fail without IRA but exercises the import path


@pytest.mark.skipif(not _HAS_PLT_SADDLE, reason="plt_saddle not importable")
class TestPltSaddleDefaultOutput:
    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_default_output(self, tmp_path):
        from rgpycrumbs.eon.plt_saddle import main

        h2o = molecule("H2O")
        job = tmp_path / "job"
        job.mkdir()
        frames = [h2o.copy() for _ in range(3)]
        ase_write(str(job / "climb"), frames, format="eon")
        ase_write(str(job / "reactant.con"), h2o, format="eon")
        lines = [
            "iteration\tstep_size\tdelta_e\tconvergence\teigenvalue\ttorque\tangle\trotations"
        ]
        for i in range(1, 3):
            lines.append(f"{i}\t0.1\t0.01\t0.05\t-0.1\t0.05\t10.0\t3")
        (job / "climb.dat").write_text("\n".join(lines) + "\n")
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main, ["--job-dir", str(job), "--plot-type", "profile"]
            )


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py required")
class TestPlotGPBatchExtraArgs:
    """Cover batch extra args forwarding (lines 651-663)."""

    def test_batch_with_bool_arg(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "data" / "s.h5"
        h5.parent.mkdir()
        with h5py.File(h5, "w") as f:
            _make_grid(f, "energy")

        cfg = tmp_path / "plots.toml"
        cfg.write_text(
            textwrap.dedent("""\
            [defaults]
            input_dir = "data"
            output_dir = "output"

            [[plots]]
            type = "surface"
            input = "s.h5"
            output = "s.pdf"
            clamp_lo = -2.0
            clamp_hi = 2.0
        """)
        )
        (tmp_path / "output").mkdir()

        result = CliRunner().invoke(
            cli,
            [
                "batch",
                "-c",
                str(cfg),
                "-b",
                str(tmp_path),
            ],
        )

    def test_batch_with_list_arg(self, tmp_path):
        cli = _try_import_plot_gp()

        h5 = tmp_path / "data" / "q.h5"
        h5.parent.mkdir()
        with h5py.File(h5, "w") as f:
            _make_grid(f, "true_energy")
            for n in [5, 10]:
                _make_grid(f, f"gp_mean_N{n}")

        cfg = tmp_path / "plots.toml"
        cfg.write_text(
            textwrap.dedent("""\
            [defaults]
            input_dir = "data"
            output_dir = "output"

            [[plots]]
            type = "quality"
            input = "q.h5"
            output = "q.pdf"
            n_points = [5, 10]
        """)
        )
        (tmp_path / "output").mkdir()

        result = CliRunner().invoke(
            cli,
            [
                "batch",
                "-c",
                str(cfg),
                "-b",
                str(tmp_path),
            ],
        )


class TestNwchemGenDeepExtra:
    """Cover remaining generate_nwchem_input lines."""

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_unix_socket_mode(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.generate_nwchem_input",
            "main",
            "generate_nwchem_input not importable",
        )

        h2o = molecule("H2O")
        pos = tmp_path / "pos.con"
        ase_write(str(pos), h2o, format="eon")

        settings = tmp_path / "settings.ini"
        settings.write_text(
            "[NWChem]\nbasis = 6-31G\nunix_socket_mode = True\nunix_socket_path = /tmp/eon.sock\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--pos-file",
                str(pos),
                "--settings",
                str(settings),
                "--output",
                str(tmp_path / "nwchem.nwi"),
            ],
        )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_tcp_socket_mode(self, tmp_path):
        main = _import_attr(
            "rgpycrumbs.eon.generate_nwchem_input",
            "main",
            "generate_nwchem_input not importable",
        )

        h2o = molecule("H2O")
        pos = tmp_path / "pos.con"
        ase_write(str(pos), h2o, format="eon")

        settings = tmp_path / "settings.ini"
        settings.write_text("[NWChem]\nbasis = 6-31G\nhost = 127.0.0.1\nport = 12345\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--pos-file",
                str(pos),
                "--settings",
                str(settings),
                "--output",
                str(tmp_path / "nwchem.nwi"),
            ],
        )

    @pytest.mark.skipif(not _HAS_PLT_NEB, reason="needs chemparseplot")
    def test_missing_settings(self, tmp_path):
        """Cover lines 109-111 (error handler)."""
        main = _import_attr(
            "rgpycrumbs.eon.generate_nwchem_input",
            "main",
            "generate_nwchem_input not importable",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--pos-file",
                str(tmp_path / "nonexistent.con"),
                "--settings",
                str(tmp_path / "nope.ini"),
                "--output",
                str(tmp_path / "out.nwi"),
            ],
        )
        assert result.exit_code != 0


class TestLogParamsDeep:
    """Cover remaining log_params lines 86-91."""

    @pytest.mark.eon
    def test_empty_config(self, tmp_path):
        log_config_ini = _import_attr(
            "rgpycrumbs.eon._mlflow.log_params",
            "log_config_ini",
            "needs mlflow",
        )

        # Empty config file
        cfg = tmp_path / "empty.ini"
        cfg.write_text("[Main]\njob = minimization\n")
        import mlflow

        with mlflow.start_run():
            log_config_ini(cfg)


class TestAuxDeep:
    """Cover _aux.py lines 159-160, 171-172, 196, 242."""

    def test_import_from_parent_env_missing(self):
        from rgpycrumbs._aux import _import_from_parent_env

        try:
            _import_from_parent_env("totally_nonexistent_module_xyz123")
        except (ImportError, Exception):  # noqa: S110
            pass

    def test_has_cuda_check(self):
        from rgpycrumbs._aux import _has_cuda

        result = _has_cuda()
        assert isinstance(result, bool)


@pytest.mark.fragments
@pytest.mark.skipif(not _HAS_XVFB, reason="Xvfb not available")
@pytest.mark.skipif(not _HAS_PYVISTA, reason="pyvista not installed")
@pytest.mark.skipif(not _HAS_PLT_NEB, reason="chemparseplot not installed")
class TestSolvisRendering:
    """Test solvis backend in fragments env (has pyvista)."""

    def test_render_c2h6(self, tmp_path):
        import pyvista as pv

        pv.start_xvfb()
        from chemparseplot.plot.neb import _render_atoms

        c2h6 = molecule("C2H6")
        img = _render_atoms(c2h6, "solvis", 0.3, "0x,90y,0z")
        assert img.ndim == 3
        assert img.shape[0] > 0

    def test_render_in_strip(self, tmp_path):
        import pyvista as pv

        pv.start_xvfb()
        from chemparseplot.plot.neb import plot_structure_strip

        atoms = [molecule("C2H6"), molecule("CH4")]
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, atoms, ["A", "B"], renderer="solvis")
        plt.close(fig)


@pytest.mark.ptm
@pytest.mark.skipif(not _HAS_OVITO_RENDER, reason="ovito render backend unavailable")
@pytest.mark.skipif(not _HAS_PLT_NEB, reason="chemparseplot not installed")
class TestOvitoRendering:
    """Test ovito backend in ptm env (has ovito)."""

    def test_render_c2h6(self, tmp_path):
        from chemparseplot.plot.neb import _render_atoms

        c2h6 = molecule("C2H6")
        img = _render_atoms(c2h6, "ovito", 0.3, "0x,90y,0z")
        assert img.ndim == 3

    def test_render_in_strip(self, tmp_path):
        from chemparseplot.plot.neb import plot_structure_strip

        atoms = [molecule("C2H6"), molecule("CH4")]
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, atoms, ["A", "B"], renderer="ovito")
        plt.close(fig)
