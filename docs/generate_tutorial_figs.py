#!/usr/bin/env python3
"""Render the NEB visualization tutorial figures.

Each command below is printed in ``docs/orgmode/tutorials/neb-visualization.org``.
The images go to ``docs/source/_static/tutorial/``.

Run as: pixi run -e fragments python docs/generate_tutorial_figs.py
"""

import matplotlib as mpl

mpl.use("Agg")

# solvis renders through PyVista, which needs a virtual display on headless hosts.
try:
    import pyvista as pv

    pv.start_xvfb()
except Exception:
    pass

from figure_commands import DOCS, NEB_INPUTS, render

NEB = "rgpycrumbs eon plt-neb"
STRIP = "--plot-type profile --plot-structures crit_points --strip-dividers"
SURFACE = "--plot-type landscape --landscape-mode surface --surface-type grad_matern"

OUT_DIR = DOCS / "source" / "_static" / "tutorial"
PAGES = {
    "neb-visualization.org": [
        f"{NEB} --plot-type profile {NEB_INPUTS} -o profile.png",
        f"{NEB} --plot-type landscape --landscape-mode path --no-project-path "
        f"{NEB_INPUTS} -o landscape_raw.png",
        f"{NEB} --plot-type landscape --landscape-mode path --project-path "
        f"{NEB_INPUTS} -o landscape_projected.png",
        f"{NEB} {SURFACE} --project-path {NEB_INPUTS} -o surface_projected.png",
        f"{NEB} {STRIP} --strip-renderer xyzrender {NEB_INPUTS} -o gallery_xyzrender.png",
        f"{NEB} {STRIP} --strip-renderer ase {NEB_INPUTS} -o gallery_ase.png",
        f"{NEB} {STRIP} --strip-renderer solvis {NEB_INPUTS} -o gallery_solvis.png",
        f"{NEB} --plot-type landscape --landscape-mode path --no-project-path "
        f"--show-evolution {NEB_INPUTS} -o evolution.png",
        f"{NEB} {SURFACE} --project-path --plot-structures crit_points "
        f"--strip-renderer xyzrender --strip-dividers {NEB_INPUTS} "
        "-o surface_gallery.png",
    ]
}

if __name__ == "__main__":
    render(PAGES, OUT_DIR)
