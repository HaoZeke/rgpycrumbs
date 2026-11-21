"""Compare the spline and uphill image tangents on a fixed tunneling path."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "scipy"]
# ///

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import click
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import cKDTree


def uphill_tangents(chords, energy):
    """Return improved energy-weighted NEB tangents, with one-sided endpoints."""
    tangents = [chords[0]]
    for i in range(1, len(energy) - 1):
        forward, backward = energy[i + 1] - energy[i], energy[i] - energy[i - 1]
        if forward * backward > 0:
            tangent = chords[i] if forward > 0 else chords[i - 1]
        else:
            hi, lo = sorted((abs(forward), abs(backward)), reverse=True)
            if hi == 0:
                tangent = chords[i] + chords[i - 1]
            else:
                w1, w2 = (hi, lo) if energy[i + 1] > energy[i - 1] else (lo, hi)
                tangent = w1 * chords[i] + w2 * chords[i - 1]
        tangents.append(tangent)
    return np.asarray([*tangents, chords[-1]])


def _normalize(rows):
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= 0):
        raise ValueError("the path contains a vanishing tangent")
    return rows / norms[:, None]


def compare_tangents(
    positions, energies, gradients, masses, target, *, potential=None, hbar_over_kb=None
):
    """Use a force spline's analytic primitive to integrate the turning time.

    Each column represents one Cartesian coordinate. Its mass is in the same
    unit system as the energy, length, time and optional hbar_over_kb.
    """
    q, energy, grad, masses = (
        np.asarray(a, dtype=float) for a in (positions, energies, gradients, masses)
    )
    if (
        q.ndim != 2
        or len(q) < 3
        or energy.shape != (len(q),)
        or grad.shape != q.shape
        or masses.shape != (q.shape[1],)
        or np.any(masses <= 0)
        or not all(np.isfinite(a).all() for a in (q, energy, grad, masses, target))
    ):
        raise ValueError(
            "expected at least three finite image/gradient rows and positive masses"
        )
    weighted = q * np.sqrt(masses)
    chords = np.diff(weighted, axis=0)
    x = np.r_[0, np.cumsum(np.linalg.norm(chords, axis=1))]
    bc = "natural" if len(q) == 3 else "not-a-knot"
    curve = CubicSpline(x, weighted, bc_type=bc)
    metric = lambda xx: np.linalg.norm(curve(xx, 1))
    s = np.r_[
        0, np.cumsum([quad(metric, a, b)[0] for a, b in zip(x[:-1], x[1:], strict=True)])
    ]
    spline = _normalize(curve(x, 1))
    uphill = _normalize(uphill_tangents(chords, energy))
    out = dict(
        target_energy=float(target),
        images=len(q),
        path_arc_length=float(s[-1]),
        angular_difference_degrees=np.degrees(
            np.arccos(np.clip((spline * uphill).sum(axis=1), -1, 1))
        ).tolist(),
    )

    def at_arc(ss):
        i = int(np.clip(np.searchsorted(s, ss) - 1, 0, len(s) - 2))
        xx = brentq(lambda v: quad(metric, x[i], v)[0] + s[i] - ss, x[i], x[i + 1])
        return curve(xx) / np.sqrt(masses)

    if potential:
        at = lambda v: float(potential(curve(v) / np.sqrt(masses)))
        grid = np.unique(
            np.concatenate(
                [np.linspace(a, b, 17) for a, b in zip(x[:-1], x[1:], strict=True)]
            )
        )
        values = np.asarray([at(v) for v in grid])
        peaks = [values.max()]
        for i in (
            np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))
            + 1
        ):
            peaks.append(
                -minimize_scalar(
                    lambda v: -at(v), bounds=(grid[i - 1], grid[i + 1]), method="bounded"
                ).fun
            )
        out["peak_energy_along_path"] = float(max(peaks))
        out["barrier_above_target"] = out["peak_energy_along_path"] - target
    for name, tangents in (
        ("cubic_coordinate_derivative", spline),
        ("higher_energy_chord", uphill),
    ):
        signed = (grad / np.sqrt(masses) * tangents).sum(axis=1)
        if not (signed[0] > 0 and signed[-1] < 0):
            raise ValueError(f"{name} does not point inward at both ends")
        primitive = CubicSpline(s, signed, bc_type=bc).antiderivative()
        work = float(primitive(s[-1]) - primitive(0))
        from_right = work > 0
        start, opposite = (s[-1], 0.0) if from_right else (0.0, s[-1])
        # Polynomial roots include interior returns even on multibarrier paths.
        primitive.c[-1] -= primitive(start)
        sign = -1 if from_right else 1
        roots = primitive.roots(extrapolate=False)
        candidates = [
            float(v)
            for v in roots
            if sign * (v - start) > 0 and sign * primitive(v, 1) < 0
        ]
        if abs(float(primitive(opposite))) <= 64 * np.finfo(float).eps * s[-1] * max(
            abs(signed)
        ):
            candidates.append(float(opposite))
        if not candidates:
            raise ValueError(f"{name} has no return inside the sampled arc")
        turn = min(candidates, key=lambda v: abs(v - start))
        a, b = sorted((start, turn))

        def inverse_speed(u):
            ss = a + (b - a) * np.sin(u) ** 2
            kinetic = float(primitive(ss))
            if kinetic <= 0:
                raise ValueError(
                    "force spline has non-positive kinetic energy inside the orbit"
                )
            return (b - a) * np.sin(2 * u) / np.sqrt(2 * kinetic)

        half = quad(inverse_speed, 0, np.pi / 2)[0]
        pos = at_arc(turn)
        endpoint = 0 if from_right else -1
        result = dict(
            projected_forces=signed.tolist(),
            signed_integral=work,
            launch_image=len(q) - 1 if from_right else 0,
            return_arc=turn,
            distance_from_opposite_arc=float(abs(turn - opposite)),
            half_period=float(half),
            return_positions=pos.tolist(),
            max_cartesian_displacement=float(abs(pos - q[endpoint]).max()),
        )
        if hbar_over_kb is not None:
            result["temperature"] = float(hbar_over_kb / (2 * half))
        if potential:
            v = float(potential(pos))
            result["return_energy_minus_target"] = v - target
            if out["barrier_above_target"] > 0:
                result["energy_residual_fraction_of_barrier"] = (v - target) / out[
                    "barrier_above_target"
                ]
        out[name] = result
    return out


@click.command()
@click.argument("images", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-energy", type=float, help="Tunneling energy in surface units.")
@click.option(
    "--level",
    type=click.Path(exists=True, path_type=Path),
    help="Native energy-level record with hbar and Boltzmann constant.",
)
@click.option(
    "--molecular-seed",
    type=click.Path(exists=True, path_type=Path),
    help="JSON seed with atom masses for Cartesian image rows.",
)
@click.option(
    "--ledger",
    type=click.Path(exists=True, path_type=Path),
    help="Measured Cartesian energy-gradient JSONL.",
)
@click.option(
    "--match-tolerance",
    default=1e-12,
    type=click.FloatRange(min=0),
    show_default=True,
    help="Maximum absolute image-to-ledger coordinate difference.",
)
@click.option(
    "--surface-command",
    help="Optional command accepting Cartesian coordinates as separate arguments and printing energy[,gradient...].",
)
def main(
    images, target_energy, level, molecular_seed, ledger, match_tolerance, surface_command
):
    """Compare path-image tangents and their inferred one-dimensional periods.

    IMAGES is a prepared CSV with image,energy,coordinate,gradient_coordinate
    columns, or a molecular JSON Cartesian array accompanied by --ledger
    and --molecular-seed. Units for prepared CSV are consistent dimensionless
    or physical units with coordinate masses of one.
    """
    sources = [images]
    factor = None
    if level:
        record = json.loads(level.read_text())
        if target_energy is not None and target_energy != record["energy"]:
            raise click.UsageError("target energy disagrees with supplied level")
        target_energy = record["energy"]
        factor = record["controls"]["hbar"] / record["controls"]["boltzmann"]
        sources.append(level)
    if target_energy is None:
        raise click.UsageError("supply --level or --target-energy")
    mismatch = 0.0
    if ledger or molecular_seed:
        if not ledger or not molecular_seed:
            raise click.UsageError("supply both --ledger and --molecular-seed")
        q = np.asarray(json.loads(images.read_text()))
        masses = np.repeat(json.loads(molecular_seed.read_text())["masses"], 3)
        rows = [json.loads(line) for line in ledger.open()]
        points = np.asarray([r["positions_angstrom"] for r in rows]).reshape(
            -1, q.shape[1]
        )
        distances, indices = cKDTree(points).query(q, p=np.inf)
        mismatch = float(max(distances))
        if mismatch > match_tolerance:
            raise click.UsageError(
                f"nearest gradient measurement differs by {mismatch:.9g}"
            )
        energy = [rows[i]["energy_eV"] for i in indices]
        grad = np.asarray([rows[i]["gradient_eV_per_angstrom"] for i in indices]).reshape(
            q.shape
        )
        sources.extend([ledger, molecular_seed])
    else:
        table = np.genfromtxt(images, delimiter=",", names=True)
        cols = [
            k.removeprefix("gradient_")
            for k in table.dtype.names
            if k.startswith("gradient_")
        ]
        if not cols:
            raise click.UsageError("prepared CSV requires gradient_<coordinate> columns")
        q = np.column_stack([table[k] for k in cols])
        grad = np.column_stack([table["gradient_" + k] for k in cols])
        energy = table["energy"]
        masses = np.ones(q.shape[1])
    calls = 0

    def potential(q):
        nonlocal calls
        values = subprocess.check_output(
            [*shlex.split(surface_command), *(format(float(v), ".17g") for v in q)],
            text=True,
        )
        calls += 1
        return float(values.split(",")[0])

    out = compare_tangents(
        q,
        energy,
        grad,
        masses,
        target_energy,
        potential=potential if surface_command else None,
        hbar_over_kb=factor,
    )
    out["input_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    out["maximum_coordinate_lookup_difference"] = mismatch
    out["additional_surface_evaluations"] = calls
    click.echo(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
