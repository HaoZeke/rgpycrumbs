"""Interpolate selected Cartesian Hessians along the original mass-weighted path."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "scipy"]
# ///

import hashlib
import json
from pathlib import Path

import click
import numpy as np
from scipy.interpolate import CubicSpline


@click.command()
@click.argument("blocks", type=click.Path(exists=True, path_type=Path))
@click.argument("seed", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option(
    "--keep",
    type=int,
    multiple=True,
    required=True,
    help="Retained image index, starting at zero.",
)
def main(blocks, seed, output, keep):
    """Write Cartesian curvature on every measured knot, using the retained BLOCKS."""
    source = blocks.read_bytes()
    measured = json.loads(source)
    points = np.asarray(measured["positions_angstrom"], dtype=float)
    hessians = np.asarray(measured["hessians_eV_per_angstrom2"], dtype=float)
    masses = np.asarray(json.loads(seed.read_text())["masses"], dtype=float)
    if masses.ndim != 1 or np.any(masses <= 0) or not np.isfinite(masses).all():
        raise click.ClickException("expected positive finite atomic masses")
    if points.ndim != 2 or points.shape[1] != 3 * len(masses):
        raise click.ClickException("expected one Cartesian triple per atom")
    n, d = points.shape
    if hessians.shape != (n, d, d) or not np.isfinite(hessians).all():
        raise click.ClickException("expected one finite square Hessian per image")
    if (
        len(keep) < 4
        or list(keep) != sorted(set(keep))
        or keep[0] != 0
        or keep[-1] != n - 1
    ):
        raise click.ClickException(
            "retain four or more distinct ascending knots including both ends"
        )
    metric = np.sqrt(np.repeat(masses, 3))
    lengths = np.linalg.norm(np.diff(points, axis=0) * metric, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise click.ClickException("expected finite positive mass-weighted chord lengths")
    knots = np.r_[0.0, np.cumsum(lengths)]
    kept = list(keep)
    fitted = CubicSpline(knots[kept], hessians[kept], axis=0, bc_type="not-a-knot")(knots)
    fitted[kept] = hessians[kept]
    missing = np.setdiff1d(np.arange(n), kept)
    if not len(missing):
        raise click.ClickException("hold out at least one image")
    estimate = fitted[missing] - hessians[missing]
    error = np.linalg.norm(estimate, axis=(1, 2))
    norm = np.linalg.norm(hessians[missing], axis=(1, 2))
    errors = [
        dict(
            image=int(i),
            frobenius_eV_per_angstrom2=float(e),
            relative_frobenius=float(e / f) if f else None,
            max_abs_eV_per_angstrom2=float(np.max(np.abs(diff))),
        )
        for i, e, f, diff in zip(missing, error, norm, estimate)
    ]
    report = dict(
        source_sha256=hashlib.sha256(source).hexdigest(),
        retained_images=kept,
        knots_angstrom_sqrt_amu=knots.tolist(),
        holdouts=errors,
        evaluation_counts=dict(new_surface_requests=0),
    )
    output.write_text(
        json.dumps(
            dict(
                method="Not-a-knot cubic along cumulative mass-weighted chords",
                settings=report,
                positions_angstrom=points.tolist(),
                hessians_eV_per_angstrom2=fitted.tolist(),
            ),
            allow_nan=False,
        )
        + "\n"
    )
    click.echo(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
