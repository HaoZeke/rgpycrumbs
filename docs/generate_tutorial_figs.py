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

# --- Post-process RST to inject image directives ---
RST = ROOT / "docs" / "source" / "tutorials" / "neb-visualization.rst"
if RST.exists():
    print(f"Injecting images into {RST}...")
    rst = RST.read_text()

    INJECTIONS = [
        ("Step 1: Energy Profile\n----------------------", "profile"),
        ("Multiple NEB optimization steps", None),  # skip, image before text
        ("Shows the NEB path on raw RMSD", "landscape_raw"),
        ("concerted mechanism.", "landscape_projected"),
        ("contours with dashed variance", None),  # surfaces side-by-side
        ("Left: projected", None),  # skip
        ("Four backends are available", None),
        ("solvis: PyVista", None),
        ("Star markers show where", "mmf_peaks"),
        ("Older bands are drawn", "evolution"),
    ]

    # Simpler approach: inject after specific headings
    IMAGE_MAP = {
        "Step 1: Energy Profile\n----------------------\n": (
            "\n.. image:: /_static/tutorial/profile.png\n   :width: 80%\n\n"
        ),
        "Path overlay (no surface fit)\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n": (
            "\n.. image:: /_static/tutorial/landscape_raw.png\n   :width: 80%\n\n"
        ),
        "Projected (s, d) coordinates\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n": (
            "\n.. image:: /_static/tutorial/landscape_projected.png\n   :width: 80%\n\n"
        ),
        "GP surface fit\n~~~~~~~~~~~~~~\n": (
            "\n.. image:: /_static/tutorial/surface_projected.png\n   :width: 49%\n"
            ".. image:: /_static/tutorial/surface_raw.png\n   :width: 49%\n\n"
        ),
        "Rendering backends\n~~~~~~~~~~~~~~~~~~\n": (
            "\n.. image:: /_static/tutorial/gallery_xyzrender.png\n   :width: 49%\n"
            ".. image:: /_static/tutorial/gallery_ase.png\n   :width: 49%\n\n"
            ".. image:: /_static/tutorial/gallery_solvis.png\n   :width: 60%\n\n"
        ),
        "Perspective tilt\n~~~~~~~~~~~~~~~~\n": (
            "\n.. image:: /_static/tutorial/no_tilt.png\n   :width: 49%\n"
            ".. image:: /_static/tutorial/tilt_8deg.png\n   :width: 49%\n\n"
        ),
        "Step 4: OCI-NEB Peak Overlay\n----------------------------\n": (
            "\n.. image:: /_static/tutorial/mmf_peaks.png\n   :width: 80%\n\n"
        ),
        "Step 5: Band Evolution\n----------------------\n": (
            "\n.. image:: /_static/tutorial/evolution.png\n   :width: 80%\n\n"
        ),
        "Combined Figure\n": (
            "\n.. image:: /_static/tutorial/surface_mmf.png\n   :width: 100%\n\n"
        ),
    }

    for marker, img_rst in IMAGE_MAP.items():
        if marker in rst:
            # Find the end of the heading (after the underline)
            idx = rst.index(marker) + len(marker)
            # Insert after the first blank line following the heading
            next_blank = rst.index("\n\n", idx)
            rst = rst[:next_blank] + "\n" + img_rst + rst[next_blank:]

    RST.write_text(rst)
    print("  Images injected into RST.")
