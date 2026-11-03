#!/usr/bin/env python3
"""Render the saddle search and minimization tutorial figures.

The input is the eOn ethane data in ``docs/orgmode/tutorials/data/ethane/``,
which ``docs/generate_singleended_data.py`` computes. Each command below is
printed in its tutorial page. The images go to ``docs/source/_static/tutorial/``.

Run as: pixi run -e fragments python docs/generate_singleended_figs.py
"""

import matplotlib as mpl

mpl.use("Agg")

from figure_commands import DOCS, render

SADDLE = "rgpycrumbs eon plt-saddle --job-dir data/ethane/saddle_search"
MIN = (
    "rgpycrumbs eon plt-min --job-dir data/ethane/minimization "
    "--prefix minimization.con --label ethane"
)
LANDSCAPE = (
    "--plot-type landscape --surface-type grad_matern --project-path "
    "--plot-structures endpoints"
)
OUT_DIR = DOCS / "source" / "_static" / "tutorial"
PAGES = {
    "saddle-search-visualization.org": [
        f"{SADDLE} --plot-type profile -o saddle_profile.png",
        f"{SADDLE} {LANDSCAPE} -o saddle_landscape.png",
        f"{SADDLE} --plot-type convergence -o saddle_convergence.png",
    ],
    "minimization-visualization.org": [
        f"{MIN} --plot-type profile -o min_profile.png",
        f"{MIN} {LANDSCAPE} -o min_landscape.png",
        f"{MIN} --plot-type convergence -o min_convergence.png",
    ],
}

if __name__ == "__main__":
    render(PAGES, OUT_DIR)
