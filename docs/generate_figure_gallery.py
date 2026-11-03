#!/usr/bin/env python3
"""Render the figure gallery.

Each command below is printed in ``docs/orgmode/tutorials/figure-gallery.org``.
The images go to ``docs/source/_static/gallery/``.

Run as: pixi run -e fragments python docs/generate_figure_gallery.py
"""

import matplotlib as mpl

mpl.use("Agg")

from figure_commands import DOCS, NEB_INPUTS, render

NEB = "rgpycrumbs eon plt-neb"
OUT_DIR = DOCS / "source" / "_static" / "gallery"
PAGES = {
    "figure-gallery.org": [
        f"{NEB} --plot-type profile --energy-unit kJ/mol "
        f"--plot-structures crit_points --xyzrender-config tube {NEB_INPUTS} "
        "-o profile_kj_tube.png",
        f"{NEB} --plot-type landscape --landscape-mode path --project-path "
        f"--plot-structures crit_points {NEB_INPUTS} -o landscape_path_strip.png",
        f"{NEB} --plot-type landscape --landscape-mode surface "
        f"--surface-type grad_imq --project-path {NEB_INPUTS} "
        "-o surface_grad_imq.png",
    ]
}

if __name__ == "__main__":
    render(PAGES, OUT_DIR)
