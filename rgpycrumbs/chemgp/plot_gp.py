# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "h5py",
#   "pandas",
#   "plotnine",
#   "chemparseplot",
# ]
# ///
"""ChemGP CLI - Plot generation from HDF5 data.

Thin CLI wrapper around chemgp.plotting functions.
All plotting logic delegated to pure functions.

.. versionadded:: 1.7.0
    Refactored from chemgp.plt_gp to thin CLI wrapper.
"""

import logging
from pathlib import Path

import click
import h5py
import numpy as np

from chemparseplot.parse.chemgp_hdf5 import (
    read_h5_grid,
    read_h5_metadata,
    read_h5_path,
    read_h5_points,
    read_h5_table,
)
from chemparseplot.plot.chemgp import (
    detect_clamp,
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
    save_plot,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


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
    pass


@cli.command()
@common_options
def convergence(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
):
    """Force/energy convergence vs oracle calls."""
    with h5py.File(input_path, "r") as f:
        df = read_h5_table(f, "table")
        meta = read_h5_metadata(f)

    conv_tol = meta.get("conv_tol", None)

    # Auto-detect y column
    y = "force_norm"
    for candidate in ["ci_force", "max_fatom", "max_force"]:
        if candidate in df.columns:
            y = candidate
            break

    fig = plot_convergence_curve(
        df,
        y=y,
        conv_tol=float(conv_tol) if conv_tol is not None else None,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
@click.option("--clamp-lo", default=None, type=float)
@click.option("--clamp-hi", default=None, type=float)
@click.option("--contour-step", default=None, type=float)
def surface(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
    clamp_lo: float | None,
    clamp_hi: float | None,
    contour_step: float | None,
):
    """2D PES contour plot."""
    # Auto-detect clamping from filename if not specified
    if clamp_lo is None and clamp_hi is None:
        clamp_lo, clamp_hi, contour_step = detect_clamp(input_path.name)

    with h5py.File(input_path, "r") as f:
        data, xc, yc = read_h5_grid(f, "energy")

        # Collect paths
        paths = None
        if "paths" in f:
            paths = {}
            for pname in f["paths"].keys():
                pdata = read_h5_path(f, pname)
                keys = list(pdata.keys())
                paths[pname] = (pdata[keys[0]], pdata[keys[1]])

        # Collect points
        points = None
        if "points" in f:
            points = {}
            for pname in f["points"].keys():
                pdata = read_h5_points(f, pname)
                keys = list(pdata.keys())
                points[pname] = (pdata[keys[0]], pdata[keys[1]])

    # Build meshgrid from coordinates
    if xc is not None and yc is not None:
        gx, gy = np.meshgrid(xc, yc)
    else:
        ny, nx = data.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    levels = None
    if clamp_lo is not None and clamp_hi is not None:
        levels = np.linspace(clamp_lo, clamp_hi, 25)

    fig = plot_surface_contour(
        gx,
        gy,
        data,
        paths=paths,
        points=points,
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
        levels=levels,
        contour_step=contour_step,
        width=width,
        height=height,
    )
    save_plot(fig, output_path, dpi)


@cli.command()
@common_options
@click.option("--n-points", multiple=True, type=int, default=None)
def quality(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
    n_points: tuple[int, ...] | None,
):
    """GP surrogate quality progression (multi-panel)."""
    # Auto-detect clamping from filename
    clamp_lo, clamp_hi, _ = detect_clamp(input_path.name)
    if clamp_lo is None:
        clamp_lo = -200.0
    if clamp_hi is None:
        clamp_hi = 50.0

    with h5py.File(input_path, "r") as f:
        true_e, xc, yc = read_h5_grid(f, "true_energy")

        # Auto-detect n values from grid names if not specified
        if not n_points:
            grid_names = [k for k in f["grids"].keys() if k.startswith("gp_mean_N")]
            n_points = sorted(int(k.replace("gp_mean_N", "")) for k in grid_names)

        grids = {}
        for n in n_points:
            gp_e, _, _ = read_h5_grid(f, f"gp_mean_N{n}")
            entry = {"gp_mean": gp_e}

            # Read training points if available
            pts_name = f"train_N{n}"
            if "points" in f and pts_name in f["points"]:
                pts = read_h5_points(f, pts_name)
                keys = list(pts.keys())
                entry["train_x"] = pts[keys[0]]
                entry["train_y"] = pts[keys[1]]

            grids[n] = entry

    fig = plot_gp_progression(
        grids,
        true_e,
        xc,
        yc,
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
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
    """RFF approximation quality vs exact GP."""
    with h5py.File(input_path, "r") as f:
        df = read_h5_table(f, "table")
        meta = read_h5_metadata(f)

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
    """MAP-NLL landscape in hyperparameter space."""
    with h5py.File(input_path, "r") as f:
        nll_data, xc, yc = read_h5_grid(f, "nll")
        opt = read_h5_points(f, "optimum")

        # Read gradient norm grid if available
        grad_norm = None
        if "grids" in f and "grad_norm" in f["grids"]:
            grad_norm, _, _ = read_h5_grid(f, "grad_norm")

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
        grid_grad_norm=grad_norm,
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
    """Hyperparameter sensitivity grid (3x3)."""
    with h5py.File(input_path, "r") as f:
        slice_df = read_h5_table(f, "slice")
        true_df = read_h5_table(f, "true_surface")
        x_vals = slice_df["x"].to_numpy()
        y_true = true_df["E_true"].to_numpy()

        panels = {}
        for j in range(1, 4):
            for i in range(1, 4):
                name = f"gp_ls{j}_sv{i}"
                if name in f:
                    gp_df = read_h5_table(f, name)
                    panels[name] = {
                        "E_pred": gp_df["E_pred"].to_numpy(),
                        "E_std": gp_df["E_std"].to_numpy(),
                    }

    fig = plot_hyperparameter_sensitivity(
        x_vals,
        y_true,
        panels,
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
    """Trust region illustration (1D slice)."""
    with h5py.File(input_path, "r") as f:
        slice_df = read_h5_table(f, "slice")
        training = read_h5_points(f, "training")
        meta = read_h5_metadata(f)

    x_slice = slice_df["x"].to_numpy()
    e_true = slice_df["E_true"].to_numpy()
    e_pred = slice_df["E_pred"].to_numpy()
    e_std = slice_df["E_std"].to_numpy()
    in_trust = slice_df["in_trust"].to_numpy()

    # Training x coordinates (filter to nearby slice)
    y_slice = float(meta.get("y_slice", 0.5))
    train_x = training.get("x", np.array([]))
    train_y = training.get("y", np.array([]))
    if len(train_x) > 0 and len(train_y) > 0:
        mask = np.abs(train_y - y_slice) < 0.3
        train_x = train_x[mask]
    else:
        train_x = None

    fig = plot_trust_region(
        x_slice,
        e_true,
        e_pred,
        e_std,
        in_trust,
        train_x=train_x,
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
    """GP variance overlaid on PES."""
    # Auto-detect clamping from filename
    clamp_lo, clamp_hi, _ = detect_clamp(input_path.name)
    if clamp_lo is None:
        clamp_lo = -200.0
    if clamp_hi is None:
        clamp_hi = 50.0

    with h5py.File(input_path, "r") as f:
        energy, xc, yc = read_h5_grid(f, "energy")
        var_data, _, _ = read_h5_grid(f, "variance")
        training = read_h5_points(f, "training")
        minima = None
        if "points" in f and "minima" in f["points"]:
            minima = read_h5_points(f, "minima")
        saddles = None
        if "points" in f and "saddles" in f["points"]:
            saddles = read_h5_points(f, "saddles")

    if xc is not None and yc is not None:
        gx, gy = np.meshgrid(xc, yc)
    else:
        ny, nx = energy.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    # Build stationary points dict
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
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
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
    """FPS subset visualization (PCA scatter)."""
    with h5py.File(input_path, "r") as f:
        selected = read_h5_points(f, "selected")
        pruned = read_h5_points(f, "pruned")

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
    """NEB energy profile (image index vs delta E)."""
    with h5py.File(input_path, "r") as f:
        df = read_h5_table(f, "table")

    fig = plot_energy_profile(df, width=width, height=height)
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
@click.option("--dpi", default=300, type=int, help="Output resolution.")
@click.option(
    "--parallel",
    "-j",
    default=1,
    type=int,
    help="Number of parallel jobs (default: 1).",
)
def batch(
    config_path: Path,
    base_dir: Path | None,
    dpi: int,
    parallel: int,
):
    """Generate multiple plots from a TOML config."""
    import tomllib
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # Map plot types to functions
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

    def generate_single_plot(entry: dict) -> tuple[str, bool, str | None]:
        """Generate a single plot. Returns (output_name, success, error_msg)."""
        plot_type = entry.get("type")
        if plot_type not in cmds:
            return entry.get("output", "unknown"), False, f"Unknown type: {plot_type}"

        out = output_dir / entry["output"]
        w = entry.get("width", 7.0)
        h = entry.get("height", 5.0)
        d = entry.get("dpi", dpi)

        # Build arguments based on plot type
        if plot_type == "landscape":
            src_dir = base_dir / entry.get("source_dir", ".")
            args = [
                "--source-dir",
                str(src_dir),
                "--output",
                str(out),
                "--width",
                str(w),
                "--height",
                str(h),
                "--dpi",
                str(d),
            ]
        else:
            inp = input_dir / entry["input"]
            if not inp.exists():
                return entry["output"], False, f"Input not found: {inp}"
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
        skip = {"type", "input", "output", "width", "height", "dpi", "source_dir"}
        for k, v in entry.items():
            if k in skip:
                continue
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    args.append(flag)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, list):
                        args.append(flag)
                        args.extend(str(x) for x in item)
                    else:
                        args.extend([flag, str(item)])
            else:
                args.extend([flag, str(v)])

        try:
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(cmds[plot_type], args)
            if result.exit_code == 0:
                return entry["output"], True, None
            else:
                return entry["output"], False, result.output
        except Exception as e:
            return entry["output"], False, str(e)

    # Process plots
    n_ok = 0
    n_fail = 0

    if parallel > 1:
        # Parallel processing with progress tracking
        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("[cyan]Generating plots...", total=len(plots))

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(generate_single_plot, entry): entry for entry in plots
                }
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        out_name, success, error = future.result()
                        if success:
                            n_ok += 1
                            log.info("[green][OK][/green] %s", out_name)
                        else:
                            n_fail += 1
                            log.error("[red][FAIL][/red] %s: %s", out_name, error)
                    except Exception as e:
                        n_fail += 1
                        log.error(
                            "[red][FAIL][/red] %s: %s", entry.get("output", "unknown"), e
                        )
                    progress.advance(task)
    else:
        # Sequential processing
        for entry in plots:
            out_name, success, error = generate_single_plot(entry)
            if success:
                n_ok += 1
                log.info("[OK] %s", out_name)
            else:
                n_fail += 1
                log.error("[FAIL] %s: %s", out_name, error)

    log.info("Batch complete: %d ok, %d failed", n_ok, n_fail)
    if n_fail > 0:
        import sys

        sys.exit(1)


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
