#!/usr/bin/env python3
"""Plot ChemGP figures from HDF5 data files.

.. versionadded:: 1.6.0

CLI for generating publication figures from ChemGP HDF5 outputs.
Reads grids, tables, paths, and point sets, then delegates to
chemparseplot for visualization.

HDF5 layout (mirrors Julia common_plot.jl helpers):

- ``grids/<name>``: 2D arrays with attrs x_range, y_range,
  x_length, y_length
- ``table/<name>``: group of same-length 1D arrays
- ``paths/<name>``: point sequences (x, y or rAB, rBC)
- ``points/<name>``: point sets (x, y or pc1, pc2)
- Root attrs: metadata scalars
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "h5py",
#   "polars",
#   "plotnine",
#   "chemparseplot",
#   "rgpycrumbs",
# ]
# ///

import logging
import sys
from pathlib import Path

import click
import h5py
import numpy as np
import polars as pl
from chemparseplot.plot.chemgp import (
    plot_convergence,
    plot_fps,
    plot_nll,
    plot_profile,
    plot_quality,
    plot_rff,
    plot_sensitivity,
    plot_surface,
    plot_trust,
    plot_variance,
)

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# --- HDF5 helpers ---


def h5_read_table(f: h5py.File, name: str = "table") -> pl.DataFrame:
    """Read a group of same-length vectors as a DataFrame."""
    g = f[name]
    cols = {}
    for k in g.keys():
        arr = g[k][()]
        if arr.dtype.kind in {"S", "O"}:
            cols[k] = arr.astype(str).tolist()
        else:
            cols[k] = arr.tolist()
    return pl.DataFrame(cols)


def h5_read_grid(
    f: h5py.File, name: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a 2D grid with optional axis ranges.

    Returns (data, x_coords, y_coords).
    """
    ds = f[f"grids/{name}"]
    data = ds[()]
    x_coords = None
    y_coords = None
    if "x_range" in ds.attrs and "x_length" in ds.attrs:
        lo, hi = ds.attrs["x_range"]
        n = int(ds.attrs["x_length"])
        x_coords = np.linspace(lo, hi, n)
    if "y_range" in ds.attrs and "y_length" in ds.attrs:
        lo, hi = ds.attrs["y_range"]
        n = int(ds.attrs["y_length"])
        y_coords = np.linspace(lo, hi, n)
    return data, x_coords, y_coords


def h5_read_path(f: h5py.File, name: str) -> dict[str, np.ndarray]:
    """Read a path (ordered point sequence)."""
    g = f[f"paths/{name}"]
    return {k: g[k][()] for k in g.keys()}


def h5_read_points(f: h5py.File, name: str) -> dict[str, np.ndarray]:
    """Read a point set."""
    g = f[f"points/{name}"]
    return {k: g[k][()] for k in g.keys()}


def h5_read_metadata(f: h5py.File) -> dict:
    """Read root-level metadata attributes."""
    return {k: f.attrs[k] for k in f.attrs.keys()}


def save_plot(fig, output: Path, dpi: int) -> None:
    """Save a figure to PDF, creating parent dirs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    log.info("Saved: %s", output)


# --- Common click options ---


def common_options(func):
    """Shared options for all subcommands."""
    func = click.option(
        "--input",
        "-i",
        "input_path",
        required=True,
        type=click.Path(exists=True, path_type=Path),
        help="HDF5 data file.",
    )(func)
    func = click.option(
        "--output",
        "-o",
        "output_path",
        required=True,
        type=click.Path(path_type=Path),
        help="Output PDF path.",
    )(func)
    func = click.option(
        "--width",
        "-W",
        default=7.0,
        type=float,
        help="Figure width in inches.",
    )(func)
    func = click.option(
        "--height",
        "-H",
        default=5.0,
        type=float,
        help="Figure height in inches.",
    )(func)
    func = click.option(
        "--dpi",
        default=300,
        type=int,
        help="Output resolution.",
    )(func)
    return func


# --- CLI ---


@click.group()
def cli():
    """ChemGP figure generation from HDF5 data."""


@cli.command()
@common_options
def convergence(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """Force/energy convergence vs oracle calls.

    Reads table/ group with oracle_calls, method, and
    max_fatom or force_norm columns.
    """
    with h5py.File(input_path, "r") as f:
        df = h5_read_table(f, "table")
        meta = h5_read_metadata(f)
    fig = plot_convergence(df, meta, width=width, height=height)
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
@click.option(
    "--grid",
    "-g",
    "grid_name",
    default="energy",
    help="Grid dataset name under grids/.",
)
@click.option(
    "--clamp-lo",
    default=None,
    type=float,
    help="Clamp grid values below this.",
)
@click.option(
    "--clamp-hi",
    default=None,
    type=float,
    help="Clamp grid values above this.",
)
def surface(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
    grid_name: str,
    clamp_lo: float | None,
    clamp_hi: float | None,
):
    """2D PES contour plot.

    Reads grids/<grid_name>, optional paths/neb and
    points/endpoints.
    """
    with h5py.File(input_path, "r") as f:
        data, xc, yc = h5_read_grid(f, grid_name)
        path = None
        if "paths" in f and "neb" in f["paths"]:
            path = h5_read_path(f, "neb")
        endpoints = None
        if "points" in f and "endpoints" in f["points"]:
            endpoints = h5_read_points(f, "endpoints")
        meta = h5_read_metadata(f)
    fig = plot_surface(
        data,
        xc,
        yc,
        path=path,
        endpoints=endpoints,
        meta=meta,
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
@click.option(
    "--n-points",
    multiple=True,
    type=int,
    default=[3, 8, 15, 30],
    help="Training set sizes for panel grid.",
)
def quality(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
    n_points: tuple[int, ...],
):
    """GP surrogate quality progression (multi-panel).

    Reads grids/true_energy, grids/gp_mean_N<n>, and
    points/train_N<n> for each n in --n-points.
    """
    panels = []
    with h5py.File(input_path, "r") as f:
        true_e, xc, yc = h5_read_grid(f, "true_energy")
        for n in n_points:
            gp_e, _, _ = h5_read_grid(f, f"gp_mean_N{n}")
            train = h5_read_points(f, f"train_N{n}")
            panels.append(
                {
                    "n": n,
                    "gp_e": gp_e,
                    "train": train,
                }
            )
    fig = plot_quality(
        true_e,
        xc,
        yc,
        panels=panels,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def rff(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """RFF approximation quality vs exact GP.

    Reads table/ with D_rff, energy/gradient MAE columns,
    and root attrs for exact GP baselines.
    """
    with h5py.File(input_path, "r") as f:
        df = h5_read_table(f, "table")
        meta = h5_read_metadata(f)
    fig = plot_rff(df, meta, width=width, height=height)
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def nll(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """MAP-NLL landscape in hyperparameter space.

    Reads grids/nll, grids/grad_norm, and
    points/optimum.
    """
    with h5py.File(input_path, "r") as f:
        nll_data, xc, yc = h5_read_grid(f, "nll")
        grad_data, _, _ = h5_read_grid(f, "grad_norm")
        optimum = h5_read_points(f, "optimum")
        meta = h5_read_metadata(f)
    fig = plot_nll(
        nll_data,
        grad_data,
        xc,
        yc,
        optimum=optimum,
        meta=meta,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def sensitivity(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """Hyperparameter sensitivity grid (3x3).

    Reads table/slice (x), table/true_surface (E_true),
    and table/gp_ls<j>_sv<i> for each panel.
    """
    with h5py.File(input_path, "r") as f:
        slice_df = h5_read_table(f, "slice")
        true_df = h5_read_table(f, "true_surface")
        panels = {}
        for j in range(1, 4):
            for i in range(1, 4):
                name = f"gp_ls{j}_sv{i}"
                if name in f:
                    panels[(j, i)] = h5_read_table(f, name)
        meta = h5_read_metadata(f)
    fig = plot_sensitivity(
        slice_df,
        true_df,
        panels=panels,
        meta=meta,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def trust(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """Trust region illustration (1D slice).

    Reads table/slice with x, E_true, E_pred, E_std,
    in_trust columns and points/training.
    """
    with h5py.File(input_path, "r") as f:
        df = h5_read_table(f, "slice")
        training = h5_read_points(f, "training")
        meta = h5_read_metadata(f)
    fig = plot_trust(
        df,
        training,
        meta=meta,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def variance(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """GP variance overlaid on PES contour.

    Reads grids/energy, grids/variance, points/training,
    points/minima, points/saddles, and root metadata.
    """
    with h5py.File(input_path, "r") as f:
        energy, xc, yc = h5_read_grid(f, "energy")
        var_data, _, _ = h5_read_grid(f, "variance")
        training = h5_read_points(f, "training")
        minima = None
        if "points" in f and "minima" in f["points"]:
            minima = h5_read_points(f, "minima")
        saddles = None
        if "points" in f and "saddles" in f["points"]:
            saddles = h5_read_points(f, "saddles")
        meta = h5_read_metadata(f)
    fig = plot_variance(
        energy,
        var_data,
        xc,
        yc,
        training=training,
        minima=minima,
        saddles=saddles,
        meta=meta,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def fps(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """FPS subset visualization (PCA scatter).

    Reads points/selected and points/pruned with
    pc1, pc2 datasets.
    """
    with h5py.File(input_path, "r") as f:
        selected = h5_read_points(f, "selected")
        pruned = h5_read_points(f, "pruned")
        meta = h5_read_metadata(f)
    fig = plot_fps(
        selected,
        pruned,
        meta=meta,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
def profile(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """NEB energy profile (image index vs delta E).

    Reads table/ with image, energy, method columns.
    """
    with h5py.File(input_path, "r") as f:
        df = h5_read_table(f, "table")
        meta = h5_read_metadata(f)
    fig = plot_profile(df, meta, width=width, height=height)
    save_plot(fig, output_path, dpi)


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="TOML config listing plots to generate.",
)
@click.option(
    "--base-dir",
    "-b",
    "base_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Base directory for relative paths in config.",
)
@click.option(
    "--dpi",
    default=300,
    type=int,
    help="Output resolution.",
)
def batch(
    config_path: Path,
    base_dir: Path | None,
    dpi: int,
):
    """Generate multiple plots from a TOML config.

    Config format::

        [[plots]]
        input = "leps_minimize.h5"
        output = "leps_minimize_convergence.pdf"
        type = "convergence"

    Optional per-plot keys: width, height, and any
    subcommand-specific options.
    """
    try:
        import tomllib  # noqa: PLC0415
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415

    with open(config_path, "rb") as fp:
        cfg = tomllib.load(fp)

    if base_dir is None:
        base_dir = config_path.parent

    plots = cfg.get("plots", [])
    if not plots:
        log.warning("No [[plots]] entries in %s", config_path)
        return

    cmds = {
        "convergence": convergence,
        "surface": surface,
        "quality": quality,
        "rff": rff,
        "nll": nll,
        "sensitivity": sensitivity,
        "trust": trust,
        "variance": variance,
        "fps": fps,
        "profile": profile,
    }

    n_ok = 0
    n_fail = 0
    for idx, entry in enumerate(plots):
        plot_type = entry.get("type")
        if plot_type not in cmds:
            log.error(
                "Plot %d: unknown type %r, skipping",
                idx,
                plot_type,
            )
            n_fail += 1
            continue

        inp = base_dir / entry["input"]
        out = base_dir / entry["output"]
        w = entry.get("width", 7.0)
        h = entry.get("height", 5.0)
        d = entry.get("dpi", dpi)

        # Build args for the click command
        args = [
            "--input",
            str(inp),
            "--output",
            str(out),
            "--width",
            str(w),
            "--height",
            str(h),
            "--dpi",
            str(d),
        ]

        # Forward extra keys as CLI options
        skip = {
            "type",
            "input",
            "output",
            "width",
            "height",
            "dpi",
        }
        for k, v in entry.items():
            if k in skip:
                continue
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    args.append(flag)
            elif isinstance(v, list):
                for item in v:
                    args.extend([flag, str(item)])
            else:
                args.extend([flag, str(v)])

        log.info(
            "[%d/%d] %s: %s -> %s",
            idx + 1,
            len(plots),
            plot_type,
            inp.name,
            out.name,
        )
        try:
            ctx = click.Context(cmds[plot_type])
            cmds[plot_type].parse_args(ctx, args)
            ctx.invoke(
                cmds[plot_type].callback,
                **ctx.params,
            )
            n_ok += 1
        except Exception:
            log.exception("Plot %d (%s) failed", idx, plot_type)
            n_fail += 1

    log.info("Batch complete: %d ok, %d failed", n_ok, n_fail)
    if n_fail > 0:
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
