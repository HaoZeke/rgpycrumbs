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
import subprocess
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


# --- Auto-detect clamping from filename ---

# MB surfaces need [-200, 50], LEPS need [-5, 5]
_CLAMP_PRESETS = {
    "mb": (-200.0, 50.0, 25.0),  # (lo, hi, contour_step)
    "leps": (-5.0, 5.0, 0.5),
}


def _detect_clamp(filename: str) -> tuple[float | None, float | None, float | None]:
    """Detect energy clamping preset from filename."""
    stem = filename.lower()
    for prefix, (lo, hi, step) in _CLAMP_PRESETS.items():
        if prefix in stem:
            return lo, hi, step
    return None, None, None


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
    """Force/energy convergence vs oracle calls."""
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
        clamp_lo, clamp_hi, contour_step = _detect_clamp(input_path.name)

    with h5py.File(input_path, "r") as f:
        data, xc, yc = h5_read_grid(f, "energy")

        # Collect paths
        paths = None
        if "paths" in f:
            paths = {}
            for pname in f["paths"].keys():
                pdata = h5_read_path(f, pname)
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

    # Build explicit levels if clamping specified
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
    clamp_lo, clamp_hi, _ = _detect_clamp(input_path.name)
    if clamp_lo is None:
        clamp_lo = -200.0
    if clamp_hi is None:
        clamp_hi = 50.0

    with h5py.File(input_path, "r") as f:
        true_e, xc, yc = h5_read_grid(f, "true_energy")

        # Auto-detect n values from grid names if not specified
        if not n_points:
            grid_names = [k for k in f["grids"].keys() if k.startswith("gp_mean_N")]
            n_points = sorted(int(k.replace("gp_mean_N", "")) for k in grid_names)

        grids = {}
        for n in n_points:
            gp_e, _, _ = h5_read_grid(f, f"gp_mean_N{n}")
            entry = {"gp_mean": gp_e}

            # Read training points if available
            pts_name = f"train_N{n}"
            if "points" in f and pts_name in f["points"]:
                pts = h5_read_points(f, pts_name)
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
        df = h5_read_table(f, "table")
        meta = h5_read_metadata(f)

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
        nll_data, xc, yc = h5_read_grid(f, "nll")
        opt = h5_read_points(f, "optimum")

        # Read gradient norm grid if available
        grad_norm = None
        if "grids" in f and "grad_norm" in f["grids"]:
            grad_norm, _, _ = h5_read_grid(f, "grad_norm")

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
        slice_df = h5_read_table(f, "slice")
        true_df = h5_read_table(f, "true_surface")
        x_vals = slice_df["x"].to_numpy()
        y_true = true_df["E_true"].to_numpy()

        panels = {}
        for j in range(1, 4):
            for i in range(1, 4):
                name = f"gp_ls{j}_sv{i}"
                if name in f:
                    gp_df = h5_read_table(f, name)
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
        slice_df = h5_read_table(f, "slice")
        training = h5_read_points(f, "training")
        meta = h5_read_metadata(f)

    x_slice = slice_df["x"].to_numpy()
    e_true = slice_df["E_true"].to_numpy()
    e_pred = slice_df["E_pred"].to_numpy()
    e_std = slice_df["E_std"].to_numpy()
    in_trust = slice_df["in_trust"].to_numpy()

    # Training x coordinates (filter to nearby slice)
    y_slice = float(meta.get("y_slice", 0.5))
    train_x = training.get("x", np.array([]))
    train_y = training.get("y", np.array([]))
    # Keep only training points near the slice
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
    clamp_lo, clamp_hi, _ = _detect_clamp(input_path.name)
    if clamp_lo is None:
        clamp_lo = -200.0
    if clamp_hi is None:
        clamp_hi = 50.0

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
    """NEB energy profile (image index vs delta E)."""
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
    "--dat-pattern",
    default=None,
    type=str,
    help="Glob pattern for .dat files (eON format).",
)
@click.option(
    "--con-pattern",
    default=None,
    type=str,
    help="Glob pattern for .con path files (eON format).",
)
@click.option(
    "--source-dir",
    "-d",
    default=".",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing .dat/.con files.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output PDF path.",
)
@click.option("--width", "-W", default=7.0, type=float)
@click.option("--height", "-H", default=5.0, type=float)
@click.option("--dpi", default=300, type=int)
@click.option(
    "--surface-type",
    default="grad_imq",
    type=str,
    help="Surface interpolation method for plt-neb.",
)
@click.option(
    "--landscape-mode",
    default="surface",
    type=click.Choice(["path", "surface"]),
)
@click.option("--plot-structures", default="none", type=str)
@click.option("--project-path", is_flag=True, default=False)
def landscape(
    dat_pattern: str | None,
    con_pattern: str | None,
    source_dir: Path,
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
    surface_type: str,
    landscape_mode: str,
    plot_structures: str,
    project_path: bool,
):
    """2D NEB reaction landscape via plt-neb (RMSD coordinates)."""
    # Delegate to plt-neb which has the full landscape pipeline
    plt_neb_script = Path(__file__).parent.parent / "eon" / "plt_neb.py"
    cmd = [
        sys.executable,
        str(plt_neb_script),
        "--source",
        "eon",
        "--plot-type",
        "landscape",
        "--landscape-mode",
        landscape_mode,
        "--surface-type",
        surface_type,
        "--plot-structures",
        plot_structures,
        "--figsize",
        str(width),
        str(height),
        "--dpi",
        str(dpi),
        "--output-file",
        str(output_path),
    ]

    if dat_pattern:
        cmd.extend(["--input-dat-pattern", dat_pattern])
    if con_pattern:
        cmd.extend(["--input-path-pattern", con_pattern])
    if project_path:
        cmd.append("--project-path")

    log.info("Delegating to plt-neb: %s", " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd, cwd=str(source_dir), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        log.error("plt-neb failed:\n%s\n%s", result.stdout, result.stderr)
        raise click.ClickException("plt-neb landscape generation failed")
    log.info("Saved: %s", output_path)


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
def batch(
    config_path: Path,
    base_dir: Path | None,
    dpi: int,
):
    """Generate multiple plots from a TOML config."""
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
        "landscape": landscape,
    }

    n_ok = 0
    n_fail = 0
    n_skip = 0
    for idx, entry in enumerate(plots):
        plot_type = entry.get("type")
        if plot_type not in cmds:
            log.error("Plot %d: unknown type %r, skipping", idx, plot_type)
            n_fail += 1
            continue

        out = output_dir / entry["output"]
        w = entry.get("width", 7.0)
        h = entry.get("height", 5.0)
        d = entry.get("dpi", dpi)

        # Landscape type uses source_dir instead of input HDF5
        if plot_type == "landscape":
            src_dir = base_dir / entry.get("source_dir", ".")
            inp_name = entry.get("source_dir", ".")
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
            inp_name = inp.name
            if not inp.exists():
                log.warning("Skipping %s: input %s not found", entry["output"], inp)
                n_skip += 1
                continue
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
                    args.extend([flag, str(item)])
            else:
                args.extend([flag, str(v)])

        log.info(
            "[%d/%d] %s: %s -> %s",
            idx + 1,
            len(plots),
            plot_type,
            inp_name,
            out.name,
        )
        try:
            ctx = click.Context(cmds[plot_type])
            cmds[plot_type].parse_args(ctx, args)
            ctx.invoke(cmds[plot_type].callback, **ctx.params)
            n_ok += 1
        except Exception:
            log.exception("Plot %d (%s) failed", idx, plot_type)
            n_fail += 1

    log.info("Batch complete: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail)
    if n_fail > 0:
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
