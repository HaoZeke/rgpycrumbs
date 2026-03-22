#!/usr/bin/env python3
"""Generate tutorial figures for the NEB visualization tutorial.

Run during docs build to produce images embedded in the tutorial.
Output goes to docs/source/_static/tutorial/.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Try headless PyVista
try:
    import pyvista as pv
    pv.start_xvfb()
except Exception:
    pass

import matplotlib.pyplot as plt
from click.testing import CliRunner
from rgpycrumbs.eon.plt_neb import main

ROOT = Path(__file__).parent.parent
DATA = ROOT / "docs" / "orgmode" / "tutorials" / "data" / "diels_alder"
OUT = ROOT / "docs" / "source" / "_static" / "tutorial"
OUT.mkdir(parents=True, exist_ok=True)

runner = CliRunner()
common = [
    "--input-dat-pattern", str(DATA / "neb_*.dat"),
    "--input-path-pattern", str(DATA / "neb_path_*.con"),
]
con = ["--con-file", str(DATA / "neb.con"), "--sp-file", str(DATA / "sp.con")]


def run(name, args):
    out_path = str(OUT / f"{name}.png")
    r = runner.invoke(main, args + ["-o", out_path])
    status = "OK" if r.exit_code == 0 else f"FAIL({r.exit_code})"
    print(f"  {name}: {status}")
    if r.exit_code != 0 and r.exception:
        print(f"    {r.exception}")
    plt.close("all")
    return r.exit_code == 0


print("Generating tutorial figures...")

# 1. Energy profile
run("profile", ["--plot-type", "profile"] + common + con)

# 2. Raw RMSD landscape path
run("landscape_raw", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
] + common + con)

# 3. Projected (s,d) landscape path
run("landscape_projected", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--project-path",
] + common + con)

# 4. GP surface (projected)
run("surface_projected", [
    "--plot-type", "landscape", "--landscape-mode", "surface",
    "--surface-type", "grad_matern", "--project-path",
] + common + con)

# 5. GP surface (raw)
run("surface_raw", [
    "--plot-type", "landscape", "--landscape-mode", "surface",
    "--surface-type", "grad_matern", "--no-project-path",
] + common + con)

# 6. Gallery: xyzrender
run("gallery_xyzrender", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
    "--plot-structures", "crit_points",
    "--strip-renderer", "xyzrender",
    "--strip-spacing", "2.0", "--strip-dividers",
] + common + con)

# 7. Gallery: ASE
run("gallery_ase", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
    "--plot-structures", "crit_points",
    "--strip-renderer", "ase",
    "--strip-spacing", "2.0", "--strip-dividers",
] + common + con)

# 8. Gallery: solvis
run("gallery_solvis", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
    "--plot-structures", "crit_points",
    "--strip-renderer", "solvis",
    "--strip-spacing", "2.0", "--strip-dividers",
] + common + con)

# 9. Perspective tilt comparison
run("no_tilt", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
    "--plot-structures", "crit_points",
    "--strip-dividers", "--perspective-tilt", "0",
] + common + con)

run("tilt_8deg", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
    "--plot-structures", "crit_points",
    "--strip-dividers", "--perspective-tilt", "8",
] + common + con)

# 10. MMF peaks overlay
run("mmf_peaks", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path",
    "--mmf-peaks", "--peak-dir", str(DATA),
] + common + con)

# 11. Band evolution
run("evolution", [
    "--plot-type", "landscape", "--landscape-mode", "path",
    "--no-project-path", "--show-evolution",
    "--con-file", str(DATA / "neb.con"),
] + common)

# 12. Profile with xyzrender insets
run("profile_xyzrender", [
    "--plot-type", "profile",
    "--plot-structures", "crit_points",
    "--strip-renderer", "xyzrender",
] + common + con)

# 13. Surface + xyzrender strip + projected
run("surface_gallery", [
    "--plot-type", "landscape", "--landscape-mode", "surface",
    "--surface-type", "grad_matern", "--project-path",
    "--plot-structures", "crit_points",
    "--strip-renderer", "xyzrender",
    "--strip-spacing", "2.0", "--strip-dividers",
] + common + con)

# 14. Surface + MMF + xyzrender
run("surface_mmf", [
    "--plot-type", "landscape", "--landscape-mode", "surface",
    "--surface-type", "grad_matern", "--project-path",
    "--mmf-peaks", "--peak-dir", str(DATA),
    "--plot-structures", "crit_points",
    "--strip-renderer", "xyzrender",
    "--strip-spacing", "2.0", "--strip-dividers",
] + common + con)

n_figs = len(list(OUT.glob("*.png")))
print(f"\nGenerated {n_figs} figures in {OUT}")
