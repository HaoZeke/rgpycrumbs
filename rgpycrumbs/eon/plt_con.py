#!/usr/bin/env python3
"""Plot general readcon CON (or chemfiles) trajectories.

Energy/force profiles, multi-file overlays, optional structure strips.
Unlike ``plt-min`` / ``plt-neb``, no eOn job directory or sidecar ``.dat``.

.. versionadded:: 1.9.x
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "polars",
#   "ase",
#   "rich",
#   "chemparseplot[neb,plot]>=1.9.12,<2",
#   "readcon>=0.13.1",
#   "rgpycrumbs>=1.9.13",
# ]
# ///

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs eon plt-con")

from chemparseplot.api import load_trajectory, plot_con_overlay, plot_con_overview
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)
log = logging.getLogger("rich")


def _has_forces(traj) -> bool:
    return (
        traj.table is not None
        and not traj.table.is_empty()
        and "fmax" in traj.table.columns
        and traj.table["fmax"].null_count() < traj.table.height
    )


@click.command()
@click.argument(
    "con_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output figure (default: <first_stem>_profile.pdf).",
)
@click.option(
    "--label",
    "labels",
    type=str,
    multiple=True,
    help="Legend label per input (must match count when multi-file).",
)
@click.option(
    "--energy-unit",
    type=click.Choice(["eV", "kcal/mol", "kJ/mol"]),
    default="eV",
    show_default=True,
)
@click.option(
    "--absolute/--relative",
    default=False,
    help="Absolute energies vs ΔE from each trajectory's first frame.",
)
@click.option(
    "--no-forces",
    is_flag=True,
    default=False,
    help="Skip force panel (single-traj mode only).",
)
@click.option(
    "--structures",
    type=click.Choice(["none", "endpoints", "linspace", "all"]),
    default="none",
    show_default=True,
    help="Structure strip mode (single-traj; ase renderer by default).",
)
@click.option(
    "--max-structs",
    type=int,
    default=8,
    show_default=True,
    help="Cap on strip structures for linspace/all.",
)
@click.option(
    "--strip-renderer",
    type=click.Choice(["ase", "xyzrender"]),
    default="ase",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option("--title", type=str, default=None)
def main(
    con_files,
    output,
    labels,
    energy_unit,
    absolute,
    no_forces,
    structures,
    max_structs,
    strip_renderer,
    dpi,
    title,
):
    """Plot CON/chemfiles trajectory profile(s).

    One file: energy (+ forces + optional structure strip).
    Multiple files: energy overlay only.
    """
    paths = list(con_files)
    trajs = [load_trajectory(p) for p in paths]
    for p, traj in zip(paths, trajs):
        n_e = int(sum(1 for v in traj.energies if v == v))
        log.info(
            "Loaded %s: %d frames, energy finite=%d, forces=%s, source=%s",
            p.name,
            traj.n_frames,
            n_e,
            "yes" if _has_forces(traj) else "no",
            traj.source,
        )

    if len(trajs) == 1:
        fig = plot_con_overview(
            trajs[0],
            energy_unit=energy_unit,
            relative=not absolute,
            show_forces=not no_forces,
            structures=structures,
            max_structs=max_structs,
            strip_renderer=strip_renderer,
            title=title or paths[0].name,
        )
    else:
        if labels and len(labels) != len(trajs):
            raise click.UsageError(
                f"--label count ({len(labels)}) must match files ({len(trajs)})"
            )
        fig, ax = plt.subplots(figsize=(5.37, 3.8), constrained_layout=True)
        plot_con_overlay(
            trajs,
            labels=list(labels) if labels else None,
            ax=ax,
            energy_unit=energy_unit,
            relative=not absolute,
            title=title or "CON overlay",
        )
    out = output or paths[0].with_name(f"{paths[0].stem}_profile.pdf")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    log.info("Wrote %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
