#!/usr/bin/env python3
"""Stitch multi-segment NEB paths into ONE band and plot it (1D + 2D).

.. versionadded:: 0.0.3

This CLI assembles several NEB segments that share an absolute-energy axis into a
single continuous reaction band (deduplicating shared junction frames and
removing per-segment minimization offsets), then renders both the 1D energy
profile (with the xyzrender structure strip) and the 2D reaction-valley
landscape on the stitched band by reusing the existing ``plt-neb`` plotter.

The stitching itself lives in :func:`chemparseplot.parse.eon.stitch.stitch_neb_segments`.

Example::

    python -m rgpycrumbs.cli --dev eon plt-neb-stitch \\
        --segment "geo2 -> well:results/02_neb/si3n4_g2g1/neb.con:0:7" \\
        --segment "well -> geo1_R:results/02_neb/si3n4_g2g1_mid/neb.con:0:" \\
        --segment "geo1_R -> N2 loss:results/02_neb/si3n4_cluster/neb.con:0:" \\
        --saddle-override "well -> geo1_R:results/03_saddle/si3n4_g2g1_mid/saddle.con:-29432.165003" \\
        --out-dir results/02_neb/si3n4_resolved \\
        --profile-output results/03_figures/si3n4_resolved-1d-path.png \\
        --landscape-output results/03_figures/si3n4_resolved-2d-landscape.png
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "ase",
#   "chemparseplot>=1.8.0",
#   "readcon>=0.7.0",
# ]
# ///

import logging
import os
import subprocess
import sys
from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - direct script execution without package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs eon plt-neb-stitch")

from chemparseplot.parse.eon.stitch import stitch_neb_segments

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("plt-neb-stitch")

PLT_NEB = Path(__file__).resolve().parent / "plt_neb.py"


def _parse_segment(spec: str) -> tuple[str, str, int | None, int | None]:
    """Parse ``LABEL:CON_PATH:START:END`` (END may be empty for 'to the end')."""
    rest, end = spec.rsplit(":", 1)
    rest, start = rest.rsplit(":", 1)
    label, con = rest.split(":", 1)

    def _idx(token: str) -> int | None:
        token = token.strip()
        return int(token) if token else None

    return label.strip(), con.strip(), _idx(start), _idx(end)


def _parse_override(spec: str) -> tuple[str, str, float]:
    """Parse ``LABEL:CON_PATH:ENERGY`` (absolute saddle energy in eV)."""
    rest, energy = spec.rsplit(":", 1)
    label, con = rest.split(":", 1)
    return label.strip(), con.strip(), float(energy)


def _run_plt_neb(args: list[str]) -> int:
    """Dispatch to the existing plt-neb plotter, inheriting the environment."""
    command = [sys.executable, str(PLT_NEB), *args]
    log.info("Running: %s", " ".join(command))
    proc = subprocess.run(command, env=os.environ.copy(), check=False)  # noqa: S603
    return proc.returncode


@click.command()
@click.option(
    "--segment",
    "segments",
    multiple=True,
    required=True,
    help="Repeatable segment spec 'LABEL:CON_PATH:START:END' (END empty = end).",
)
@click.option(
    "--saddle-override",
    "saddle_overrides",
    multiple=True,
    default=(),
    help="Repeatable 'LABEL:SADDLE_CON:ENERGY_EV' to replace a segment peak.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for the stitched band (neb.con, neb_path_000.con, neb_000.dat, sp.con).",
)
@click.option(
    "--profile-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output PNG for the 1D profile (skipped if omitted).",
)
@click.option(
    "--landscape-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output PNG for the 2D landscape (skipped if omitted).",
)
@click.option(
    "--sp-label",
    default="hidden TS",
    show_default=True,
    help="Legend label for the overlaid saddle (sp.con) on the landscape.",
)
@click.option("--profile-title", default="Stitched NEB path", show_default=True)
@click.option("--landscape-title", default="Reaction valley (stitched)", show_default=True)
@click.option("--plot-structures", default="crit_points", show_default=True)
@click.option("--strip-renderer", default="xyzrender", show_default=True)
@click.option("--facecolor", default="white", show_default=True)
@click.option("--figsize", nargs=2, type=float, default=(7.0, 7.0), show_default=True)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option("--fontsize-base", type=int, default=16, show_default=True)
@click.option("--zoom-ratio", type=float, default=0.4, show_default=True)
@click.option("--rotation", default="auto", show_default=True)
@click.option("--ira-kmax", type=float, default=14.0, show_default=True)
def main(
    segments,
    saddle_overrides,
    out_dir,
    profile_output,
    landscape_output,
    sp_label,
    profile_title,
    landscape_title,
    plot_structures,
    strip_renderer,
    facecolor,
    figsize,
    dpi,
    fontsize_base,
    zoom_ratio,
    rotation,
    ira_kmax,
):
    """Stitch NEB segments and plot the combined 1D profile and 2D landscape."""
    seg_specs = [_parse_segment(s) for s in segments]
    overrides = {}
    for spec in saddle_overrides:
        label, con, energy = _parse_override(spec)
        overrides[label] = (con, energy)

    summary = stitch_neb_segments(seg_specs, out_dir, saddle_overrides=overrides)

    click.echo("")
    click.echo(f"Stitched band: {summary.n_frames} frames in {summary.out_dir}")
    click.echo(f"Segment boundaries (combined index): {summary.boundary_indices}")
    for rec in summary.segments:
        click.echo(
            f"  {rec.label:<22s} frames {rec.start:>2d}..{rec.end:<2d}  "
            f"well {rec.well_energy:+.4f}  peak {rec.peak_energy:+.4f}  "
            f"barrier {rec.barrier:.4f} eV"
        )
    click.echo(
        f"Overall highest point: {summary.highest_energy:+.4f} eV "
        f"at image {summary.highest_index}"
    )
    click.echo("")

    out_dir = Path(out_dir)
    neb_con = out_dir / "neb.con"
    dat_pattern = str(out_dir / "neb_*.dat")
    path_pattern = str(out_dir / "neb_path*.con")
    sp_con = out_dir / "sp.con"
    fig_w, fig_h = figsize

    common = [
        "--con-file", str(neb_con),
        "--plot-structures", plot_structures,
        "--strip-renderer", strip_renderer,
        "--facecolor", facecolor,
        "--input-dat-pattern", dat_pattern,
        "--figsize", str(fig_w), str(fig_h),
        "--dpi", str(dpi),
        "--fontsize-base", str(fontsize_base),
        "--zoom-ratio", str(zoom_ratio),
        "--rotation", rotation,
    ]

    rc = 0
    if profile_output is not None:
        profile_args = [
            *common,
            "--output-file", str(profile_output),
            "--plot-type", "profile",
            "--rc-mode", "path",
            "--title", profile_title,
        ]
        rc |= _run_plt_neb(profile_args)

    if landscape_output is not None:
        landscape_args = [
            *common,
            "--output-file", str(landscape_output),
            "--plot-type", "landscape",
            "--rc-mode", "path",
            "--landscape-mode", "surface",
            "--landscape-path", "all",
            "--surface-type", "grad_imq",
            "--project-path",
            "--show-pts",
            "--strip-dividers",
            "--input-path-pattern", path_pattern,
            "--additional-con", str(sp_con), sp_label,
            "--ira-kmax", str(ira_kmax),
            "--cache-file", str(out_dir / "2dcache.parquet"),
            "--show-legend",
            "--title", landscape_title,
        ]
        rc |= _run_plt_neb(landscape_args)

    sys.exit(rc)


if __name__ == "__main__":
    main()
