"""Compare absolute rate calculations against digitized Arrhenius markers."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "readcon", "click", "numpy", "scipy", "matplotlib", "xyzrender>=0.3", "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@25e35124c335ce7446f3e4a29658a062142bdb1c"]
# ///

import csv
import json
from pathlib import Path

import click
import numpy as np


def draw_thermal_orbit(ax, geometry, ring_file, level_file, label, bounds=(0.02, 0.03, 0.36, 0.46), count=9):
    """Inset the tunnelling orbit as an xyzrender ensemble of its replicas."""
    from ase.io import read

    atoms = read(geometry, format="extxyz")
    level = json.loads(level_file.read_text())
    if ring_file.suffix == ".con":
        import readcon

        frames = list(readcon.iter_con(str(ring_file)))
        if not frames:
            raise ValueError("empty readcon orbit")
        half = frames[0].metadata.get("evaluation_counts", {}).get("mode") == "rate"
        key = "neb_bead" if half else "tau_over_beta"
        frames.sort(key=lambda f: f.metadata[key])
        positions = []
        for frame in frames:
            order = np.argsort([atom.atom_id for atom in frame.atoms])
            if ([frame.atoms[i].symbol for i in order] != atoms.get_chemical_symbols()
                    or not np.isclose(frame.metadata["temperature_K"], level["temperature_K"])):
                raise ValueError("orbit identity or temperature mismatch")
            positions.append(np.asarray(frame.xyz)[order])
        ring = np.stack(positions)
        if half:
            ring = np.concatenate((ring, ring[::-1]))
    else:
        ring = np.asarray(json.loads(ring_file.read_text()), dtype=float).reshape(-1, len(atoms), 3)
    if len(ring) != level["beads"]:
        raise ValueError("inset orbit and normalized level specify different bead counts")
    # One opaque turning-point replica and translucent replicas along the
    # outgoing half of the orbit, drawn by xyzrender.
    from ase import Atoms
    from chemparseplot.plot.kinetics import add_image_inset, render_path_ensemble

    outgoing = ring[: len(ring) // 2 + 1]
    chosen = outgoing[np.unique(np.linspace(0, len(outgoing) - 1, count).round().astype(int))]
    image = render_path_ensemble([Atoms(atoms.get_chemical_symbols(), positions=x) for x in chosen])
    add_image_inset(ax, image, bounds, f"{label} orbit, {level['temperature_K']:.0f} K".strip())


def draw_cost(ax, cost_file, colors):
    """Ranges of surface evaluations per temperature for each method."""
    with cost_file.open() as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        color = colors.get(row["method"], "0.3")
        both = float(row["analytic_hessians_max"]) > 0
        for offset, key, hatch in ((0.18, "gradients", None), (-0.18, "analytic_hessians", "////")):
            low, high = float(row[f"{key}_min"]), float(row[f"{key}_max"])
            if high <= 0:
                continue
            offset = offset if both else 0.0
            ax.barh(i + offset, high - low if high > low else high * 0.04, left=low, height=0.3,
                    color=color if hatch is None else "white", edgecolor=color, hatch=hatch, linewidth=1)
            ax.annotate(f"{low:,.0f}–{high:,.0f}" if high > low else f"{high:,.0f}", (high, i + offset),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize="x-small")
    import textwrap

    ax.set_yticks(range(len(rows)), [textwrap.fill(row["method"], 22) for row in rows], fontsize="small")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(100, 2e4)
    ax.set_xlabel("Surface evaluations per temperature")
    ax.barh(np.nan, 0, color="0.4", label="Gradients, with finite-difference Hessians")
    ax.barh(np.nan, 0, color="white", edgecolor="0.4", hatch="////", label="Analytic Hessians")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize="x-small", frameon=False)


def draw_search_cost(ax, cost_file, cycle):
    """Surface calls spent finding the orbit at each temperature, per method.

    Reads the JSON rows of ``orbit_search_cost.py``: method, temperature_K,
    search_gradients, search_hessians and force_calls, where a Hessian counts
    as the gradients of its central differences. The rate stage is left out
    because it costs the same for every method.
    """
    rows = json.loads(cost_file.read_text())
    methods = list(dict.fromkeys(row["method"] for row in rows))
    styles = ("o", "s", "^", "D", "v", "P", "X")
    for i, method in enumerate(methods):
        points = sorted((row["temperature_K"], row["force_calls"]) for row in rows
                        if row["method"] == method and row.get("outcome") == "returned_rate")
        if not points:
            continue
        temperature, calls = zip(*points, strict=True)
        ax.plot(temperature, calls, marker=styles[i % len(styles)], markersize=4, linewidth=1.2,
                color=cycle[i % len(cycle)], label=method)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel("Force calls to find the orbit\n(gradients + 2d per Hessian)")
    ax.invert_xaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), fontsize="x-small", frameon=False, ncol=1)


@click.command()
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.argument("harmonic", type=click.Path(exists=True, path_type=Path))
@click.option("--series", multiple=True, type=(str, click.Path(exists=True, path_type=Path)),
              help="Method name and normalized JSON record or JSONL series. Repeat for each calculation.")
@click.option("--compare-series", metavar="NAME", help="Also tabulate differences from this series, interpolated in inverse temperature within its measured range.")
@click.option("--plot", required=True, type=click.Path(path_type=Path))
@click.option("--spectra", type=(click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path)),
              help="Reactant and saddle frequency records for continuous harmonic curves.")
@click.option("--inverse-range", nargs=2, type=float, help="Lower and upper 1000 K/T bounds.")
@click.option("--orbit", type=(click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path),
                               click.Path(exists=True, path_type=Path)),
              help="Equilibrium extxyz, readcon orbit or full bead JSON and normalized level for the inset.")
@click.option("--orbit-label", default="", help="Method used to calculate the displayed orbit.")
@click.option("--cost", type=click.Path(exists=True, path_type=Path),
              help="CSV of method, gradients_min, gradients_max, analytic_hessians_min, analytic_hessians_max.")
@click.option("--search-cost", type=click.Path(exists=True, path_type=Path),
              help="JSON rows of orbit_search_cost.py: force calls to find the orbit per method and temperature.")
def main(reference, harmonic, series, compare_series, plot, spectra, inverse_range, orbit, orbit_label, cost,
         search_cost):
    """Plot absolute log10 rates in inverse seconds and write per-point discrepancies."""
    import matplotlib.pyplot as plt
    from chemparseplot.plot.kinetics import plot_arrhenius
    from chemparseplot.plot.palette import RUHI_COLORS
    from chemparseplot.plot.theme import get_theme, setup_publication_theme
    from chemparseplot.units import Q_

    setup_publication_theme(get_theme("ruhi", font_size=10))
    # Colour and marker both change between series, so neither carries the
    # distinction alone.
    cycle = [RUHI_COLORS[name] for name in ("sky", "magenta", "teal", "coral")] + ["#6a4c93", "#8a6d00"]

    markers = np.genfromtxt(reference, delimiter=",", names=True, ndmin=1)
    xref = markers["inverse_temperature_1000_per_K"]
    yref = markers["log10_rate_per_s"]
    if not np.all(np.diff(xref) > 0):
        raise ValueError("reference inverse temperatures must increase")
    baseline = np.genfromtxt(harmonic, delimiter=",", names=True, ndmin=1)
    side = cost or search_cost
    fig = plt.figure(layout="constrained", figsize=(10.5, 6.6) if side else (7.2, 6.6))
    grid = fig.add_gridspec(2, 2 if side else 1, height_ratios=(3, 1.25), width_ratios=(1.75, 1) if side else None)
    ax = fig.add_subplot(grid[0, 0])
    difference = fig.add_subplot(grid[1, 0], sharex=ax)
    ax.tick_params(labelbottom=False)
    temperatures = baseline["temperature_K"]
    harmonic_rates = {key: baseline[key] for key in ("htst", "qq_htst")}
    if spectra:
        from rgpycrumbs.kinetics import harmonic_log_rates

        bounds = inverse_range or (min(xref.min(), 1000 / temperatures.max()),
                                   max(xref.max(), 1000 / temperatures.min()))
        if not 0 < bounds[0] < bounds[1]:
            raise ValueError("inverse-temperature bounds must be positive and ordered")
        temperatures = 1000 / np.linspace(*bounds, 250)
        minimum, saddle = (json.loads(path.read_text()) for path in spectra)
        rates = harmonic_log_rates(minimum["signed_wavenumbers_cm-1"], saddle["signed_wavenumbers_cm-1"],
                                   saddle["electronic_energy_eV"] - minimum["electronic_energy_eV"], temperatures)
        harmonic_rates = dict(zip(("htst", "qq_htst"), np.asarray(rates) / np.log(10)))
        unstable = [w for w in saddle["signed_wavenumbers_cm-1"] if w < 0]
        if len(unstable) == 1:
            # Harmonic crossover T_c = hbar |omega| / (2 pi k_B), with hc/k_B in cm K.
            crossover = 1.4387768775 * abs(unstable[0]) / (2 * np.pi)
            for axis in (ax, difference):
                axis.axvline(1000 / crossover, color="0.5", linestyle=":", linewidth=0.8)
            ax.annotate(rf"$T_c$ = {crossover:.0f} K", (1000 / crossover, 0.03), xycoords=("data", "axes fraction"),
                        ha="right", va="bottom", rotation=90, fontsize="small", color="0.35")
    if orbit:
        draw_thermal_orbit(ax, *orbit, orbit_label)
    for label, key in (("HTST", "htst"), ("qq-HTST", "qq_htst")):
        plot_arrhenius(ax, Q_(temperatures, "kelvin"), harmonic_rates[key],
                       label=label, linestyle="--" if key == "qq_htst" else "-", color="0.25", linewidth=1.4)
    plot_arrhenius(ax, Q_(1000 / xref, "kelvin"), yref, label="Published HQTST",
                   marker="D", color="black", fillstyle="none", linestyle="none", markersize=8, markeredgewidth=1.1)
    shapes = ("o", "s", "^", "v", "x", "+", "<", ">")
    table = csv.writer(click.get_text_stream("stdout"))
    by_method = {}
    for label, path in series:
        raw = path.read_text()
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        by_method.setdefault(label, []).extend(rows if isinstance(rows, list) else [rows])
    columns = ["method", "temperature_K", "log10_rate_per_s", "published_interpolated", "difference_decades"]
    if compare_series is not None:
        if compare_series not in by_method:
            raise click.BadParameter("must name a supplied --series", param_hint="--compare-series")
        observations = sorted((1000 / row["temperature_K"], row["log10_rate_per_s"])
                              for row in by_method[compare_series] if row["temperature_K"] > 0)
        if len(observations) != len(by_method[compare_series]) or len(observations) < 2:
            raise ValueError("the comparison series requires at least two positive temperatures")
        xcompare, ycompare = np.asarray(observations, dtype=float).T
        if not (np.all(np.isfinite(xcompare) & np.isfinite(ycompare)) and np.all(np.diff(xcompare) > 0)):
            raise ValueError("the comparison series requires distinct temperatures and finite rates")
        columns.extend(("comparison_method", "comparison_interpolated", "difference_from_comparison_decades"))
    table.writerow(columns)
    colors = {}
    for shape, shade, (label, rows) in zip(shapes * len(by_method), cycle * len(by_method), by_method.items()):
        colors[label] = shade
        points = sorted((row["temperature_K"], row["log10_rate_per_s"]) for row in rows)
        temperatures, rates = np.asarray(points, dtype=float).T
        if not np.all(np.isfinite(rates) & np.isfinite(temperatures) & (temperatures > 0)):
            raise ValueError(f"invalid absolute rate or temperature in {label}")
        curve = plot_arrhenius(ax, Q_(temperatures, "kelvin"), rates, marker=shape, linestyle="none",
                               label=label, markersize=4.5, fillstyle="none", color=shade, markeredgewidth=1.1)
        inside = [(1000 / t, 10 ** (rate - np.interp(1000 / t, xref, yref))) for t, rate in points
                  if xref[0] - 1e-9 <= 1000 / t <= xref[-1] + 1e-9]
        if inside:
            difference.plot(*np.asarray(inside).T, marker=shape, linestyle="none", markersize=4.5,
                            fillstyle="none", color=curve.get_color(), markeredgewidth=1.1)
        for t, rate in points:
            x = 1000 / t
            match = np.interp(x, xref, yref) if xref[0] - 1e-9 <= x <= xref[-1] + 1e-9 else None
            values = [label, t, rate, match, rate - match if match is not None else None]
            if compare_series is not None:
                reference_rate = np.interp(x, xcompare, ycompare, left=np.nan, right=np.nan)
                supported = np.isfinite(reference_rate)
                values.extend((compare_series, reference_rate if supported else None,
                               rate - reference_rate if supported else None))
            table.writerow(values)
    ax.set_xlabel("")
    difference.axhline(1, color="black", linewidth=0.8)
    difference.set_ylim(0.9, None)
    difference.set_xlabel(r"$1000\,\mathrm{K}/T$")
    difference.set_ylabel("Calculated rate /\npublished rate")
    kelvin = ax.secondary_xaxis("top", functions=(lambda x: 1000 / np.maximum(x, 1e-12),
                                                  lambda t: 1000 / np.maximum(t, 1e-12)))
    kelvin.set_xticks([1000, 500, 333, 250, 200])
    kelvin.set_xlabel(r"$T$ (K)")
    ax.legend(loc="upper right", fontsize="small", frameon=False)
    if search_cost:
        draw_search_cost(fig.add_subplot(grid[:, 1]), search_cost, cycle)
    elif cost:
        draw_cost(fig.add_subplot(grid[:, 1]), cost, colors)
    fig.savefig(plot)
    plt.close(fig)


if __name__ == "__main__":
    main()
