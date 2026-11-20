"""Compare absolute rate calculations against digitized Arrhenius markers."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "matplotlib", "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@ce577ca547fe1ea3904c44e6ef2a277f18789715"]
# ///

import csv
import json
from pathlib import Path

import click
import numpy as np


@click.command()
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.argument("harmonic", type=click.Path(exists=True, path_type=Path))
@click.option("--series", multiple=True, type=(str, click.Path(exists=True, path_type=Path)),
              help="Method name and normalized JSON record or JSONL series. Repeat for each calculation.")
@click.option("--plot", required=True, type=click.Path(path_type=Path))
def main(reference, harmonic, series, plot):
    """Plot absolute log10 rates in inverse seconds and write per-point discrepancies."""
    import matplotlib.pyplot as plt
    from chemparseplot.plot.kinetics import plot_arrhenius
    from chemparseplot.units import Q_

    markers = np.genfromtxt(reference, delimiter=",", names=True, ndmin=1)
    xref = markers["inverse_temperature_1000_per_K"]
    yref = markers["log10_rate_per_s"]
    if not np.all(np.diff(xref) > 0):
        raise ValueError("reference inverse temperatures must increase")
    baseline = np.genfromtxt(harmonic, delimiter=",", names=True, ndmin=1)
    fig, ax = plt.subplots(layout="constrained")
    for label, key in (("HTST", "htst"), ("qq-HTST", "qq_htst")):
        plot_arrhenius(ax, Q_(baseline["temperature_K"], "kelvin"), baseline[key],
                       label=label, linestyle="--" if key == "qq_htst" else "-")
    plot_arrhenius(ax, Q_(1000 / xref, "kelvin"), yref, label="Published HQTST",
                   marker="D", color="black", fillstyle="none", linestyle="none")
    table = csv.writer(click.get_text_stream("stdout"))
    table.writerow(("method", "temperature_K", "log10_rate_per_s", "published_interpolated", "difference_decades"))
    by_method = {}
    for label, path in series:
        raw = path.read_text()
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        by_method.setdefault(label, []).extend(rows if isinstance(rows, list) else [rows])
    for label, rows in by_method.items():
        points = sorted((row["temperature_K"], row["log10_rate_per_s"]) for row in rows)
        temperatures, rates = np.asarray(points, dtype=float).T
        if not np.all(np.isfinite(rates) & np.isfinite(temperatures) & (temperatures > 0)):
            raise ValueError(f"invalid absolute rate or temperature in {label}")
        plot_arrhenius(ax, Q_(temperatures, "kelvin"), rates, marker="o", label=label)
        for t, rate in points:
            x = 1000 / t
            match = np.interp(x, xref, yref) if xref[0] <= x <= xref[-1] else None
            table.writerow((label, t, rate, match, rate - match if match is not None else None))
    ax.legend()
    fig.savefig(plot)
    plt.close(fig)


if __name__ == "__main__":
    main()
