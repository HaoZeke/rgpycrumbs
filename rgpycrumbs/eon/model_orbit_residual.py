"""Measure Cartesian force balance on an equal-time model orbit."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy"]
# ///

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import click
import numpy as np

from rgpycrumbs.eon.orbit_residual import periodic_force_residual


@click.command()
@click.argument("orbit", type=click.Path(exists=True, path_type=Path))
@click.option("--coordinate", multiple=True, required=True, help="Cartesian CSV column, in order.")
@click.option("--mass", multiple=True, required=True, type=click.FloatRange(min=0, min_open=True),
              help="Mass for each Cartesian column, in model units.")
@click.option("--period", required=True, type=click.FloatRange(min=0, min_open=True), help="Full orbit period in model units.")
@click.option("--surface-command", required=True, help="Command taking Cartesian coordinates and returning energy,gradient components as CSV.")
def main(orbit, coordinate, mass, period, surface_command):
    """Measure the finite-bead and Fourier defects from fresh surface gradients.

    ORBIT supplies one row per uniformly spaced bead around the full period,
    excluding the repeated endpoint. The command is invoked on each row.
    """
    table = np.genfromtxt(orbit, names=True, delimiter=",", ndmin=1)
    if len(mass) != len(coordinate) or any(name not in table.dtype.names for name in coordinate):
        raise click.UsageError("each requested Cartesian column requires one mass")
    positions = np.column_stack([table[name] for name in coordinate])
    if len(positions) < 3 or not np.isfinite(positions).all():
        raise click.UsageError("expected at least three finite bead positions")
    command = shlex.split(surface_command)
    observations = []
    for point in positions:
        raw = subprocess.check_output([*command, *map(repr, point.tolist())], text=True)
        observation = np.array([float(item) for item in raw.strip().split(",")])
        if observation.shape != (len(mass) + 1,) or not np.isfinite(observation).all():
            raise click.ClickException("surface must return finite energy and one gradient per coordinate")
        observations.append(observation)
    observations = np.stack(observations)
    report = {
        "orbit_sha256": hashlib.sha256(orbit.read_bytes()).hexdigest(),
        "coordinate_columns": coordinate, "coordinate_masses": mass, "full_period": period,
        "beads": len(positions), "units": "consistent model units",
        "evaluation_counts": {"scope": "independent orbit validation", "surface_requests": len(positions)},
        "energies": observations[:, 0].tolist(), "gradients": observations[:, 1:].tolist(),
    }
    for name, discrete in [("ring_polymer", True), ("fourier", False)]:
        r = periodic_force_residual(positions, -observations[:, 1:], mass, period, discrete)
        bead, axis = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        report[name] = {"max_component": float(np.max(np.abs(r))), "rms": float(np.mean(r**2)**0.5),
                        "maximum_bead": int(bead), "maximum_coordinate": coordinate[axis]}
    click.echo(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
