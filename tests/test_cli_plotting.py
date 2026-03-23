# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for CLI plotting scripts with synthetic data.

Exercises the actual plotting code paths of plt_neb, plot_gp, plt_saddle,
and plt_min using minimal synthetic eOn/ChemGP data generated in fixtures.
"""

import importlib
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False

pytestmark = pytest.mark.pure


def _can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


_HAS_CHEMPARSEPLOT_NEB = _can_import("chemparseplot.plot.neb")
_HAS_DIMER_TRAJ = _can_import("chemparseplot.parse.eon.dimer_trajectory")
_HAS_CHEMGP = _can_import("chemparseplot.plot.chemgp")
_HAS_JAX = _can_import("jax")


# ---------------------------------------------------------------------------
# Helpers: synthetic eOn data
# ---------------------------------------------------------------------------


def _write_con_file(path, atoms_list):
    """Write a list of ASE Atoms as a multi-frame eOn .con file."""
    from ase.io import write

    write(str(path), atoms_list, format="eon")


def _make_h2o_images(n_images=5, displacement_scale=0.05):
    """Create n_images of H2O with small displacements for NEB-like paths."""
    from ase.build import molecule

    images = []
    base = molecule("C2H6")
    base.cell = [10, 10, 10]
    base.pbc = True
    rng = np.random.RandomState(42)
    for i in range(n_images):
        atoms = base.copy()
        atoms.positions += (
            rng.randn(*atoms.positions.shape) * displacement_scale * (i + 1)
        )
        images.append(atoms)
    return images


def _write_neb_dat(path, n_images=5, step_index=0):
    """Write a synthetic neb_*.dat file.

    Format: header line then rows with columns:
        index  distance  energy  f_para  eigenvalue
    """
    lines = ["index\tdistance\tenergy\tf_para\teigenvalue\n"]
    rng = np.random.RandomState(step_index)
    for i in range(n_images):
        dist = float(i) / max(n_images - 1, 1)
        # Parabolic energy profile with peak in the middle
        energy = -0.5 * (dist - 0.5) ** 2 + 0.25
        f_para = rng.randn() * 0.01
        eigenvalue = rng.randn() * 0.1
        lines.append(f"{i}\t{dist:.6f}\t{energy:.6f}\t{f_para:.6f}\t{eigenvalue:.6f}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def _make_neb_data(tmpdir, n_steps=3, n_images=5):
    """Create synthetic neb_*.dat and neb_path_*.con files in tmpdir."""
    images = _make_h2o_images(n_images)
    for step in range(n_steps):
        dat_path = tmpdir / f"neb_{step:04d}.dat"
        con_path = tmpdir / f"neb_path_{step:04d}.con"
        _write_neb_dat(dat_path, n_images=n_images, step_index=step)
        # Write slightly displaced images per step
        step_images = []
        for img in images:
            a = img.copy()
            a.positions += np.random.RandomState(step).randn(*a.positions.shape) * 0.001
            step_images.append(a)
        _write_con_file(con_path, step_images)
    return images


def _make_climb_dir(tmpdir, n_frames=10):
    """Create a synthetic dimer/saddle search job directory.

    Returns the job directory path.
    """
    from ase.build import molecule

    job_dir = tmpdir / "climb_job"
    job_dir.mkdir()

    base = molecule("C2H6")
    base.cell = [10, 10, 10]
    base.pbc = True

    rng = np.random.RandomState(99)
    frames = []
    for i in range(n_frames):
        a = base.copy()
        a.positions += rng.randn(*a.positions.shape) * 0.02 * (i + 1)
        frames.append(a)

    # climb (movie file)
    _write_con_file(job_dir / "climb", frames)

    # reactant.con
    _write_con_file(job_dir / "reactant.con", [base])

    # saddle.con (last frame)
    _write_con_file(job_dir / "saddle.con", [frames[-1]])

    # climb.dat (TSV)
    header = "iteration\tstep_size\tconvergence\tdelta_e\teigenvalue\n"
    rows = []
    for i in range(n_frames):
        step_size = 0.1 / (i + 1)
        convergence = 0.5 / (i + 1)
        delta_e = -0.01 * i + rng.randn() * 0.001
        eigenvalue = -0.5 + 0.05 * i
        rows.append(
            f"{i}\t{step_size:.6f}\t{convergence:.6f}\t{delta_e:.6f}\t{eigenvalue:.6f}\n"
        )
    with open(job_dir / "climb.dat", "w") as f:
        f.write(header)
        f.writelines(rows)

    # mode.dat (Nx3 array for the eigenvector)
    mode = rng.randn(len(base), 3)
    np.savetxt(job_dir / "mode.dat", mode)

    return job_dir


def _make_min_dir(tmpdir, n_frames=10, prefix="min"):
    """Create a synthetic minimization job directory.

    Returns the job directory path.
    """
    from ase.build import molecule

    job_dir = tmpdir / "min_job"
    job_dir.mkdir()

    base = molecule("C2H6")
    base.cell = [10, 10, 10]
    base.pbc = True

    rng = np.random.RandomState(77)
    frames = []
    for i in range(n_frames):
        a = base.copy()
        a.positions += rng.randn(*a.positions.shape) * 0.03 * max(n_frames - i, 1)
        frames.append(a)

    # movie file
    _write_con_file(job_dir / prefix, frames)

    # min.con (final structure)
    _write_con_file(job_dir / f"{prefix}.con", [frames[-1]])

    # min.dat (TSV)
    header = "iteration\tstep_size\tconvergence\tenergy\n"
    rows = []
    for i in range(n_frames):
        step_size = 0.1 / (i + 1)
        convergence = 1.0 / (i + 1)
        energy = -10.0 - 0.1 * i + rng.randn() * 0.01
        rows.append(f"{i}\t{step_size:.6f}\t{convergence:.6f}\t{energy:.6f}\n")
    with open(job_dir / f"{prefix}.dat", "w") as f:
        f.write(header)
        f.writelines(rows)

    return job_dir


def _make_chemgp_convergence_h5(path):
    """Create a minimal ChemGP HDF5 file for the convergence subcommand."""
    with h5py.File(str(path), "w") as f:
        n = 20
        rng = np.random.RandomState(10)
        # table group
        tbl = f.create_group("table")
        tbl.create_dataset("oracle_calls", data=np.arange(n))
        tbl.create_dataset("max_fatom", data=rng.rand(n) * 0.5)
        tbl.create_dataset("force_norm", data=rng.rand(n))
        tbl.create_dataset("method", data=np.array(["dimer"] * n, dtype="S10"))
        # metadata
        f.attrs["conv_tol"] = 0.01


def _make_chemgp_surface_h5(path):
    """Create a minimal ChemGP HDF5 file for the surface subcommand."""
    with h5py.File(str(path), "w") as f:
        nx, ny = 20, 20
        rng = np.random.RandomState(11)
        grids = f.create_group("grids")
        ds = grids.create_dataset("energy", data=rng.randn(ny, nx))
        ds.attrs["x_range"] = [0.0, 3.0]
        ds.attrs["x_length"] = nx
        ds.attrs["y_range"] = [0.0, 3.0]
        ds.attrs["y_length"] = ny

        # paths
        paths = f.create_group("paths")
        p1 = paths.create_group("neb_path")
        p1.create_dataset("rAB", data=np.linspace(0.5, 2.5, 10))
        p1.create_dataset("rBC", data=np.linspace(2.5, 0.5, 10))

        # points
        points = f.create_group("points")
        sp = points.create_group("saddle")
        sp.create_dataset("rAB", data=np.array([1.5]))
        sp.create_dataset("rBC", data=np.array([1.5]))


def _make_chemgp_profile_h5(path):
    """Create a minimal ChemGP HDF5 file for the profile subcommand."""
    with h5py.File(str(path), "w") as f:
        n = 7
        tbl = f.create_group("table")
        tbl.create_dataset("image", data=np.arange(n))
        tbl.create_dataset("energy", data=np.array([0.0, 0.2, 0.5, 0.8, 0.5, 0.2, 0.0]))
        tbl.create_dataset("method", data=np.array(["neb"] * n, dtype="S10"))


# ---------------------------------------------------------------------------
# plt_neb tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_CHEMPARSEPLOT_NEB,
    reason="chemparseplot.plot.neb not available",
)
class TestPltNebPlotting:
    """Test actual plotting paths of plt_neb.py."""

    def _import_main(self):
        try:
            from rgpycrumbs.eon.plt_neb import main

            return main
        except ImportError:
            pytest.skip("plt_neb import failed (missing dep)")

    def test_profile_eon_source(self, tmp_path):
        """Test profile plot with eOn .dat source."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=3, n_images=5)
        output = tmp_path / "profile.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Copy data files into the isolated dir
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        # Allow exit code 0 or graceful error; check no crash
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_profile_index_rc_mode(self, tmp_path):
        """Test profile plot with --rc-mode index."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=2, n_images=5)
        output = tmp_path / "profile_idx.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "--rc-mode",
                    "index",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

    def test_profile_eigenvalue_mode(self, tmp_path):
        """Test profile plot with --plot-mode eigenvalue."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=2, n_images=5)
        output = tmp_path / "profile_eigen.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "--plot-mode",
                    "eigenvalue",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

    def test_profile_spline_method(self, tmp_path):
        """Test profile plot with --spline-method spline."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=2, n_images=5)
        output = tmp_path / "profile_spline.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "--spline-method",
                    "spline",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

    def test_profile_normalize_rc(self, tmp_path):
        """Test profile plot with --normalize-rc."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=2, n_images=5)
        output = tmp_path / "profile_norm.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "--normalize-rc",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

    def test_profile_start_end_range(self, tmp_path):
        """Test profile plot with --start/--end slicing."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=3, n_images=5)
        output = tmp_path / "profile_range.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "--start",
                    "0",
                    "--end",
                    "2",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

    def test_profile_no_highlight_last(self, tmp_path):
        """Test profile plot with --no-highlight-last."""
        main = self._import_main()
        _make_neb_data(tmp_path, n_steps=2, n_images=5)
        output = tmp_path / "profile_nohl.png"
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td_path = Path(td)
            import shutil

            for f in tmp_path.glob("neb_*"):
                shutil.copy2(f, td_path / f.name)

            result = runner.invoke(
                main,
                [
                    "--plot-type",
                    "profile",
                    "--source",
                    "eon",
                    "--no-highlight-last",
                    "-o",
                    str(output),
                    "--dpi",
                    "72",
                ],
            )
            plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"


# ---------------------------------------------------------------------------
# plt_saddle tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_DIMER_TRAJ,
    reason="chemparseplot dev branch not installed",
)
class TestPltSaddlePlotting:
    """Test actual plotting paths of plt_saddle.py."""

    def _import_main(self):
        try:
            from rgpycrumbs.eon.plt_saddle import main

            return main
        except ImportError:
            pytest.skip("plt_saddle import failed")

    def test_profile(self, tmp_path):
        """Test saddle profile plot."""
        main = self._import_main()
        job_dir = _make_climb_dir(tmp_path)
        output = tmp_path / "saddle_profile.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "profile",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_convergence(self, tmp_path):
        """Test saddle convergence panel."""
        main = self._import_main()
        job_dir = _make_climb_dir(tmp_path)
        output = tmp_path / "saddle_convergence.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "convergence",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    @pytest.mark.surfaces
    def test_landscape(self, tmp_path):
        pytest.importorskip("jax")
        """Test saddle landscape plot (no IRA, falls back to ASE Procrustes)."""
        main = self._import_main()
        job_dir = _make_climb_dir(tmp_path)
        output = tmp_path / "saddle_landscape.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "landscape",
                "--surface-type",
                "grad_matern",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_mode_evolution(self, tmp_path):
        """Test mode-evolution plot (shows placeholder when per-iter modes unavailable)."""
        main = self._import_main()
        job_dir = _make_climb_dir(tmp_path)
        output = tmp_path / "saddle_mode.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "mode-evolution",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_verbose_flag(self, tmp_path):
        """Test verbose output."""
        main = self._import_main()
        job_dir = _make_climb_dir(tmp_path)
        output = tmp_path / "saddle_verbose.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "profile",
                "-v",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"


# ---------------------------------------------------------------------------
# plt_min tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_DIMER_TRAJ,
    reason="chemparseplot dev branch not installed",
)
class TestPltMinPlotting:
    """Test actual plotting paths of plt_min.py."""

    def _import_main(self):
        try:
            from rgpycrumbs.eon.plt_min import main

            return main
        except ImportError:
            pytest.skip("plt_min import failed")

    def test_profile(self, tmp_path):
        """Test minimization profile plot."""
        main = self._import_main()
        job_dir = _make_min_dir(tmp_path)
        output = tmp_path / "min_profile.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "profile",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_convergence(self, tmp_path):
        """Test minimization convergence panel."""
        main = self._import_main()
        job_dir = _make_min_dir(tmp_path)
        output = tmp_path / "min_convergence.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "convergence",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    @pytest.mark.surfaces
    def test_landscape(self, tmp_path):
        pytest.importorskip("jax")
        """Test minimization landscape plot."""
        main = self._import_main()
        job_dir = _make_min_dir(tmp_path)
        output = tmp_path / "min_landscape.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "landscape",
                "--surface-type",
                "grad_matern",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_verbose_flag(self, tmp_path):
        """Test verbose output."""
        main = self._import_main()
        job_dir = _make_min_dir(tmp_path)
        output = tmp_path / "min_verbose.png"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--job-dir",
                str(job_dir),
                "--plot-type",
                "profile",
                "-v",
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"


# ---------------------------------------------------------------------------
# plot_gp (ChemGP) tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_CHEMGP,
    reason="chemparseplot chemgp not importable",
)
@pytest.mark.skipif(not _HAS_H5PY, reason="h5py required for ChemGP tests")
class TestPlotGPPlotting:
    """Test actual plotting paths of plot_gp.py (ChemGP CLI)."""

    def _import_cli(self):
        try:
            from rgpycrumbs.chemgp.plot_gp import cli

            return cli
        except ImportError:
            pytest.skip("plot_gp import failed (missing dep)")

    def test_convergence(self, tmp_path):
        """Test convergence subcommand."""
        cli = self._import_cli()
        h5_path = tmp_path / "convergence.h5"
        _make_chemgp_convergence_h5(h5_path)
        output = tmp_path / "convergence.pdf"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convergence",
                "-i",
                str(h5_path),
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_surface(self, tmp_path):
        """Test surface subcommand."""
        cli = self._import_cli()
        h5_path = tmp_path / "surface.h5"
        _make_chemgp_surface_h5(h5_path)
        output = tmp_path / "surface.pdf"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "surface",
                "-i",
                str(h5_path),
                "-o",
                str(output),
                "--clamp-lo",
                "-2.0",
                "--clamp-hi",
                "2.0",
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_profile(self, tmp_path):
        """Test profile subcommand."""
        cli = self._import_cli()
        h5_path = tmp_path / "profile.h5"
        _make_chemgp_profile_h5(h5_path)
        output = tmp_path / "profile.pdf"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "profile",
                "-i",
                str(h5_path),
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert output.exists()

    def test_surface_without_clamp(self, tmp_path):
        """Test surface subcommand with auto-detect clamping."""
        cli = self._import_cli()
        # Name the file to not match any clamp pattern
        h5_path = tmp_path / "custom_surface.h5"
        _make_chemgp_surface_h5(h5_path)
        output = tmp_path / "surface_auto.pdf"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "surface",
                "-i",
                str(h5_path),
                "-o",
                str(output),
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

    def test_surface_with_figsize(self, tmp_path):
        """Test surface subcommand with custom width/height."""
        cli = self._import_cli()
        h5_path = tmp_path / "surface_sized.h5"
        _make_chemgp_surface_h5(h5_path)
        output = tmp_path / "surface_sized.pdf"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "surface",
                "-i",
                str(h5_path),
                "-o",
                str(output),
                "-W",
                "10.0",
                "-H",
                "8.0",
                "--clamp-lo",
                "-2.0",
                "--clamp-hi",
                "2.0",
                "--dpi",
                "72",
            ],
        )
        plt.close("all")
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
