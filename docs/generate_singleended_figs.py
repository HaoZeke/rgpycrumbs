#!/usr/bin/env python3
"""Generate tutorial figures for the saddle search and minimization tutorials.

Creates synthetic eOn-style output directories and generates all figures.
Run as: pixi run -e test python docs/generate_singleended_figs.py
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

# Import the write helper from tests
import sys

import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_cli_plotting import _write_con_file

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "source" / "_static" / "tutorial"
OUT.mkdir(parents=True, exist_ok=True)

runner = CliRunner()


def _make_saddle_data(job_dir, n_frames=15):
    """Create synthetic dimer/saddle search data (C2H6 -> distorted)."""
    base = molecule("C2H6")
    base.cell = [10, 10, 10]
    base.pbc = True

    rng = np.random.RandomState(42)
    frames = []
    for i in range(n_frames):
        a = base.copy()
        # Progressive distortion toward saddle
        displacement = rng.randn(*a.positions.shape) * 0.02 * (i + 1)
        # Add systematic drift (mimics dimer walking uphill)
        displacement[:, 0] += 0.01 * i
        a.positions += displacement
        frames.append(a)

    _write_con_file(job_dir / "climb", frames)
    _write_con_file(job_dir / "reactant.con", [base])
    _write_con_file(job_dir / "saddle.con", [frames[-1]])

    header = "iteration\tstep_size\tconvergence\tdelta_e\teigenvalue\n"
    rows = []
    for i in range(n_frames):
        step_size = 0.08 / (i + 1) + 0.002
        convergence = 0.8 / (i + 1) + 0.01
        delta_e = 0.05 * np.log(i + 1) + rng.randn() * 0.005
        eigenvalue = -0.8 + 0.04 * i + rng.randn() * 0.02
        rows.append(
            f"{i}\t{step_size:.6f}\t{convergence:.6f}\t{delta_e:.6f}\t{eigenvalue:.6f}\n"
        )
    with open(job_dir / "climb.dat", "w") as f:
        f.write(header)
        f.writelines(rows)

    # Mode vector
    mode = rng.randn(len(base), 3)
    mode /= np.linalg.norm(mode)
    np.savetxt(job_dir / "mode.dat", mode)
    return job_dir


def _make_min_data(job_dir, n_frames=20, prefix="minimization"):
    """Create synthetic minimization data (C2H6 relaxation)."""
    base = molecule("C2H6")
    base.cell = [10, 10, 10]
    base.pbc = True

    rng = np.random.RandomState(77)
    frames = []
    for i in range(n_frames):
        a = base.copy()
        # Start far from minimum, converge exponentially
        scale = 0.05 * np.exp(-0.15 * i)
        a.positions += rng.randn(*a.positions.shape) * scale
        frames.append(a)

    _write_con_file(job_dir / prefix, frames)
    _write_con_file(job_dir / f"{prefix}.con", [frames[-1]])

    header = "iteration\tstep_size\tconvergence\tenergy\n"
    rows = []
    for i in range(n_frames):
        step_size = 0.1 * np.exp(-0.1 * i) + 0.001
        convergence = 2.0 * np.exp(-0.2 * i) + 0.005
        energy = -10.0 - 0.5 * (1 - np.exp(-0.15 * i)) + rng.randn() * 0.002
        rows.append(f"{i}\t{step_size:.6f}\t{convergence:.6f}\t{energy:.6f}\n")
    with open(job_dir / f"{prefix}.dat", "w") as f:
        f.write(header)
        f.writelines(rows)

    return job_dir


def run(name, cli_main, args):
    """Run a CLI command and save the figure."""
    out_path = str(OUT / f"{name}.png")
    r = runner.invoke(cli_main, [*args, "-o", out_path])
    status = "OK" if r.exit_code == 0 else f"FAIL({r.exit_code})"
    print(f"  {name}: {status}")
    if r.exit_code != 0 and r.exception:
        import traceback

        traceback.print_exception(
            type(r.exception), r.exception, r.exception.__traceback__
        )
    plt.close("all")
    return r.exit_code == 0


print("Generating single-ended tutorial figures...")

tmpdir = Path(tempfile.mkdtemp())

# --- Saddle Search ---
saddle_dir = tmpdir / "saddle_job"
saddle_dir.mkdir()
_make_saddle_data(saddle_dir)

from rgpycrumbs.eon.plt_saddle import main as saddle_main

sd = str(saddle_dir)
run("saddle_profile", saddle_main, ["--job-dir", sd, "--plot-type", "profile"])
run("saddle_convergence", saddle_main, ["--job-dir", sd, "--plot-type", "convergence"])
run(
    "saddle_landscape",
    saddle_main,
    [
        "--job-dir",
        sd,
        "--plot-type",
        "landscape",
        "--surface-type",
        "grad_matern",
        "--project-path",
        "--plot-structures",
        "endpoints",
        "--strip-renderer",
        "xyzrender",
        "--strip-dividers",
    ],
)
run("saddle_mode", saddle_main, ["--job-dir", sd, "--plot-type", "mode-evolution"])

# --- Minimization ---
min_dir = tmpdir / "min_job"
min_dir.mkdir()
_make_min_data(min_dir)

from rgpycrumbs.eon.plt_min import main as min_main

md = str(min_dir)
run("min_profile", min_main, ["--job-dir", md, "--plot-type", "profile"])
run("min_convergence", min_main, ["--job-dir", md, "--plot-type", "convergence"])
run(
    "min_landscape",
    min_main,
    [
        "--job-dir",
        md,
        "--plot-type",
        "landscape",
        "--surface-type",
        "grad_matern",
        "--project-path",
        "--plot-structures",
        "endpoints",
        "--strip-renderer",
        "xyzrender",
        "--strip-dividers",
    ],
)

# Clean up
shutil.rmtree(tmpdir, ignore_errors=True)

n_figs = len(list(OUT.glob("saddle_*.png"))) + len(list(OUT.glob("min_*.png")))
print(f"\nGenerated {n_figs} single-ended figures in {OUT}")
