"""Measure the irreducible discrepancy from an affine correction in inverse temperature."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "scipy"]
# ///

import csv
import hashlib
import json
from pathlib import Path

import click
import numpy as np
from scipy.optimize import linprog


def marker_rates(rows, method, inverse):
    series = [row for row in rows if row["method"] == method]
    temperatures = np.asarray([float(row["temperature_K"]) for row in series])
    rates = np.asarray([float(row["log10_rate_per_s"]) for row in series])
    result = []
    for x in inverse:
        indices = np.flatnonzero(np.isclose(1000 / temperatures, x, rtol=1e-12, atol=0))
        if len(indices) != 1:
            raise ValueError(f"expected one {method!r} observation at 1000/T = {x}")
        result.append(rates[indices[0]])
    return np.asarray(result)


def minimax(inverse, difference):
    """Fit d(x) = a + b*x with the smallest possible maximum absolute residual."""
    design = np.column_stack((np.ones_like(inverse), inverse))
    bounds = np.column_stack((np.vstack((design, -design)), -np.ones(2 * len(inverse))))
    fit = linprog(
        [0, 0, 1],
        A_ub=bounds,
        b_ub=np.r_[difference, -difference],
        bounds=[(None, None), (None, None), (0, None)],
        method="highs",
    )
    if not fit.success:
        raise ValueError(f"minimax fit failed: {fit.message}")
    intercept, slope, _ = fit.x
    residual = difference - design @ fit.x[:2]
    maximum = float(np.max(abs(residual)))
    active = np.flatnonzero(np.isclose(abs(residual), maximum, atol=1e-10, rtol=1e-9))
    return {
        "intercept_decades": float(intercept),
        "slope_decades_per_1000_over_K": float(slope),
        "max_residual_decades": maximum,
        "max_residual_rate_factor": 10**maximum,
        "difference_range_decades": [float(min(difference)), float(max(difference))],
        "active_temperature_K": (1000 / inverse[active]).tolist(),
        "active_residual_decades": residual[active].tolist(),
        "residual_decades": residual.tolist(),
    }


@click.command()
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.argument("rates", type=click.Path(exists=True, path_type=Path))
@click.option("--method", required=True, help="Series in the measured rate CSV.")
@click.option(
    "--coarse-method",
    help="Series with half as many beads for leading 1/P² extrapolation.",
)
def main(reference, rates, method, coarse_method):
    """Compare measured RATES at the inverse-temperature markers in a REFERENCE CSV."""
    with reference.open() as stream:
        markers = list(csv.DictReader(stream))
    with rates.open() as stream:
        series = list(csv.DictReader(stream))
    x = np.asarray([float(row["inverse_temperature_1000_per_K"]) for row in markers])
    y = np.asarray([float(row["log10_rate_per_s"]) for row in markers])
    if (
        len(x) < 3
        or len(set(x)) != len(x)
        or np.any(x <= 0)
        or not np.all(np.isfinite(x))
    ):
        raise ValueError("expected at least three distinct positive inverse temperatures")
    fine = marker_rates(series, method, x)
    out = {
        "method": method,
        "markers": len(x),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "measured_sha256": hashlib.sha256(rates.read_bytes()).hexdigest(),
        "inverse_temperature_1000_per_K": x.tolist(),
        "reference_log10_rate_per_s": y.tolist(),
        "measured_log10_rate_per_s": fine.tolist(),
        "affine_minimax": minimax(x, fine - y),
    }
    if coarse_method:
        coarse = marker_rates(series, coarse_method, x)
        limit = fine + (fine - coarse) / 3
        out.update(
            coarse_method=coarse_method,
            coarse_log10_rate_per_s=coarse.tolist(),
            leading_extrapolated_log10_rate_per_s=limit.tolist(),
            leading_extrapolated_minimax=minimax(x, limit - y),
        )
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
