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
#   "pandas",
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
import pandas as pd
from chemparseplot.plot.chemgp import (
    plot_convergence_curve,
    plot_energy_profile,
    plot_fps_projection,
    plot_gp_progression,
    plot_hyperparameter_sensitivity,
    plot_nll_landscape,
    plot_rff_quality,
    plot_surface_contour,
    plot_trust_region,
    plot_variance_overlay,
)

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# --- HDF5 helpers ---


def h5_read_table(f: h5py.File, name: str = "table") -> pd.DataFrame:
    """Read a group of same-length vectors as a DataFrame."""
    g = f[name]
    cols = {}
    for k in g.keys():
        arr = g[k][()]
        if arr.dtype.kind in {"S", "O"}:
            cols[k] = arr.astype(str).tolist()
        else:
            cols[k] = arr.tolist()
    return pd.DataFrame(cols)


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
    """Save a plotnine ggplot or matplotlib Figure to PDF."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, plt.Figure):
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        # plotnine ggplot
        fig.save(str(output), dpi=dpi, verbose=False)
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

    # Detect y column: prefer ci_force > max_fatom > max_force > force_norm
    y_col = "force_norm"
    for candidate in ["ci_force", "max_fatom", "max_force"]:
        if candidate in df.columns:
            y_col = candidate
            break

    conv_tol = meta.get("conv_tol", None)
    fig = plot_convergence_curve(
        df,
        x="oracle_calls",
        y=y_col,
        color="method",
        conv_tol=float(conv_tol) if conv_tol is not None else None,
        width=width,
        height=height,
    )
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

        # Collect paths
        paths = None
        if "paths" in f:
            paths = {}
            for pname in f["paths"].keys():
                pdata = h5_read_path(f, pname)
                # paths have x,y or rAB,rBC
                keys = list(pdata.keys())
                paths[pname] = (pdata[keys[0]], pdata[keys[1]])

        # Collect points
        points = None
        if "points" in f:
            points = {}
            for pname in f["points"].keys():
                pdata = h5_read_points(f, pname)
                keys = list(pdata.keys())
                points[pname] = (pdata[keys[0]], pdata[keys[1]])

    # Build meshgrid from coordinates
    if xc is not None and yc is not None:
        gx, gy = np.meshgrid(xc, yc)
    else:
        ny, nx = data.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    fig = plot_surface_contour(
        gx,
        gy,
        data,
        paths=paths,
        points=points,
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
    default=None,
    help="Training set sizes for panel grid.",
)
def quality(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
    n_points: tuple[int, ...] | None,
):
    """GP surrogate quality progression (multi-panel).

    Reads grids/true_energy, grids/gp_mean_N<n>, and
    points/train_N<n> for each n in --n-points.
    """
    with h5py.File(input_path, "r") as f:
        true_e, xc, yc = h5_read_grid(f, "true_energy")

        # Auto-detect n values from grid names if not specified
        if not n_points:
            grid_names = [k for k in f["grids"].keys() if k.startswith("gp_mean_N")]
            n_points = sorted(int(k.replace("gp_mean_N", "")) for k in grid_names)

        grids = {}
        for n in n_points:
            gp_e, _, _ = h5_read_grid(f, f"gp_mean_N{n}")
            grids[n] = {"gp_mean": gp_e}

    fig = plot_gp_progression(
        grids,
        true_e,
        xc,
        yc,
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

    # Rename columns to match chemparseplot API
    rename_map = {}
    if "energy_mae_vs_gp" in df.columns:
        rename_map["energy_mae_vs_gp"] = "energy_mae"
    if "gradient_mae_vs_gp" in df.columns:
        rename_map["gradient_mae_vs_gp"] = "gradient_mae"
    if "D_rff" in df.columns:
        rename_map["D_rff"] = "d_rff"
    if rename_map:
        df = df.rename(columns=rename_map)

    exact_e = float(meta.get("gp_e_mae", 0.0))
    exact_g = float(meta.get("gp_g_mae", 0.0))

    fig = plot_rff_quality(
        df,
        exact_e_mae=exact_e,
        exact_g_mae=exact_g,
        width=width,
        height=height,
    )
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

    Reads grids/nll and points/optimum.
    """
    with h5py.File(input_path, "r") as f:
        nll_data, xc, yc = h5_read_grid(f, "nll")
        opt = h5_read_points(f, "optimum")

    if xc is not None and yc is not None:
        gx, gy = np.meshgrid(xc, yc)
    else:
        ny, nx = nll_data.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    optimum = None
    if "log_sigma2" in opt and "log_theta" in opt:
        optimum = (float(opt["log_sigma2"][0]), float(opt["log_theta"][0]))

    fig = plot_nll_landscape(
        gx,
        gy,
        nll_data,
        optimum=optimum,
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

    Reads slice/x, true_surface/E_true, and gp_ls<j>_sv<i>
    for each panel.
    """
    with h5py.File(input_path, "r") as f:
        slice_df = h5_read_table(f, "slice")
        true_df = h5_read_table(f, "true_surface")
        x_vals = slice_df["x"].to_numpy()
        y_true = true_df["E_true"].to_numpy()

        # Build long-form DataFrame for faceted plot
        rows = []
        for j in range(1, 4):
            for i in range(1, 4):
                name = f"gp_ls{j}_sv{i}"
                if name in f:
                    gp_df = h5_read_table(f, name)
                    y_pred = gp_df["E_pred"].to_numpy()
                    for k in range(len(x_vals)):
                        rows.append(
                            {
                                "x": x_vals[k],
                                "y_true": y_true[k],
                                "y_pred": y_pred[k],
                                "y_lower": 0.0,
                                "y_upper": 0.0,
                                "ell": f"ell={j}",
                                "sigma_f": f"sv={i}",
                            }
                        )

    df = pd.DataFrame(rows)
    fig = plot_hyperparameter_sensitivity(
        df,
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

    Reads slice/ with x, E_true, E_pred, E_std,
    in_trust columns and points/training.
    """
    with h5py.File(input_path, "r") as f:
        slice_df = h5_read_table(f, "slice")
        training = h5_read_points(f, "training")

    # Build DataFrame with columns the plot function expects
    df = slice_df.rename(columns={"E_pred": "y_pred"})
    # Compute confidence bounds
    if "E_std" in slice_df.columns:
        std = slice_df["E_std"]
        df["y_lower"] = df["y_pred"] - 2 * std
        df["y_upper"] = df["y_pred"] + 2 * std

    train_pts = None
    if "x" in training and "y" in training:
        train_pts = (training["x"], training["y"])
    elif "x" in training and "E" in training:
        train_pts = (training["x"], training["E"])

    fig = plot_trust_region(
        df,
        train_points=train_pts,
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
    """GP variance overlaid on PES.

    Reads grids/energy, grids/variance, points/training,
    optional points/minima, points/saddles.
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

    if xc is not None and yc is not None:
        gx, gy = np.meshgrid(xc, yc)
    else:
        ny, nx = energy.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    # Build stationary points dict for the plot function
    stationary = {}
    if minima is not None:
        keys = list(minima.keys())
        for idx in range(len(minima[keys[0]])):
            stationary[f"min{idx}"] = (
                float(minima[keys[0]][idx]),
                float(minima[keys[1]][idx]),
            )
    if saddles is not None:
        keys = list(saddles.keys())
        for idx in range(len(saddles[keys[0]])):
            stationary[f"saddle{idx}"] = (
                float(saddles[keys[0]][idx]),
                float(saddles[keys[1]][idx]),
            )

    train_pts = None
    if training:
        keys = list(training.keys())
        train_pts = (training[keys[0]], training[keys[1]])

    fig = plot_variance_overlay(
        gx,
        gy,
        energy,
        var_data,
        train_points=train_pts,
        stationary=stationary if stationary else None,
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

    fig = plot_fps_projection(
        selected["pc1"],
        selected["pc2"],
        pruned["pc1"],
        pruned["pc2"],
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

    fig = plot_energy_profile(
        df,
        x="image",
        y="energy",
        color="method",
        width=width,
        height=height,
    )
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

        [defaults]
        input_dir = "output"
        output_dir = "output"

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

    defaults = cfg.get("defaults", {})
    input_dir = base_dir / defaults.get("input_dir", ".")
    output_dir = base_dir / defaults.get("output_dir", ".")

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

        inp = input_dir / entry["input"]
        out = output_dir / entry["output"]
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
