"""Compare molecular LI action drives under two rigid image alignment conventions."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy"]
# ///

import hashlib
import json
from pathlib import Path

import click
import numpy as np


def action_drive(rows, masses, target_energy, width, adjacent):
    """Return the interior nudged action drive in centered mass-weighted coordinates."""
    mass = np.array(masses)
    if mass.ndim != 1 or np.any(mass <= 0):
        raise ValueError("atomic masses must be positive")
    root_mass = np.sqrt(mass)[:, None]
    xyz = np.array([r["positions_angstrom"] for r in rows]).reshape(
        len(rows), len(mass), 3
    )
    gradients = np.array([r["gradient_eV_per_angstrom"] for r in rows]).reshape(xyz.shape)
    q = (xyz - np.average(xyz, weights=mass, axis=1)[:, None, :]) * root_mass
    forces = -gradients / root_mass
    gap = np.array([r["energy_eV"] - target_energy for r in rows])
    for i in range(1, len(q)):
        reference = q[i - 1] if adjacent else q[0]
        left, _, right = np.linalg.svd(q[i].T @ reference)
        left[:, -1] *= np.linalg.det(left @ right)
        rotation = left @ right
        q[i], forces[i] = q[i] @ rotation, forces[i] @ rotation
    chords = np.diff(q.reshape(len(q), -1), axis=0)
    length = np.linalg.norm(chords, axis=1)
    if np.any(length <= 0) or width <= 0:
        raise ValueError("chords and momentum width must be positive")
    unit = chords / length[:, None]
    root = np.hypot(gap, width)
    term = gap + root
    negative = gap < 0
    term[negative] = width**2 / (root[negative] - gap[negative])
    sigma = np.sqrt(term)
    weights = 0.5 * (sigma[1:] + sigma[:-1])
    slope = sigma / (2 * root)
    scale = 0.5 * (length[:-1] + length[1:]) * slope[1:-1]
    surface_drive = scale[:, None] * forces.reshape(len(q), -1)[1:-1]
    chord_drive = weights[1:, None] * unit[1:] - weights[:-1, None] * unit[:-1]
    drive = surface_drive + chord_drive
    decomposed = []
    for i, full_drive in enumerate(drive, start=1):
        backward, forward = gap[i] - gap[i - 1], gap[i + 1] - gap[i]
        if backward > 0 and forward > 0:
            tangent = chords[i]
        elif backward < 0 and forward < 0:
            tangent = chords[i - 1]
        else:
            small, large = sorted((abs(backward), abs(forward)))
            left, right = (small, large) if gap[i + 1] >= gap[i - 1] else (large, small)
            tangent = left * chords[i - 1] + right * chords[i]
        norm = np.linalg.norm(tangent)
        if not norm:
            raise ValueError("undefined interior energy-weighted tangent")
        vector = full_drive - np.dot(full_drive, tangent) * tangent / norm**2
        rotations = np.array([np.cross(axis, q[i]).ravel() for axis in np.eye(3)])
        _, singular, basis = np.linalg.svd(rotations, full_matrices=False)
        basis = basis[singular > singular[0] * max(rotations.shape) * np.finfo(float).eps]
        rigid = basis.T @ (basis @ vector)
        surface = surface_drive[i - 1]
        surface_perp = surface - np.dot(surface, tangent) * tangent / norm**2
        surface_rigid = basis.T @ (basis @ surface_perp)
        chord_rigid = rigid - surface_rigid
        decomposed.append(
            {
                "image": i,
                "perpendicular": float(np.linalg.norm(vector)),
                "rotation": float(np.linalg.norm(rigid)),
                "remainder": float(np.linalg.norm(vector - rigid)),
                "surface_rotation_before_nudging": float(np.linalg.norm(basis @ surface)),
                "surface_rotation_after_nudging": float(np.linalg.norm(surface_rigid)),
                "chord_rotation_after_nudging": float(np.linalg.norm(chord_rigid)),
                "surface_chord_rotation_dot": float(surface_rigid @ chord_rigid),
            }
        )
    return {
        "action": float(weights @ length),
        "maximum": max(decomposed, key=lambda row: row["perpendicular"]),
        "interior": decomposed,
    }


@click.command()
@click.argument("ledger", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--mass",
    multiple=True,
    required=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Atomic mass in amu, in atom order.",
)
@click.option("--images", required=True, type=click.IntRange(min=3))
@click.option(
    "--target-energy",
    required=True,
    type=float,
    help="Fixed target electronic energy in eV.",
)
@click.option(
    "--width",
    required=True,
    type=click.FloatRange(min=0, min_open=True),
    help="Absolute momentum width in eV, frozen from the initial band.",
)
def main(ledger, mass, images, target_energy, width):
    """Compare measured image batches from an energy/gradient JSONL ledger."""
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if not rows or len(rows) % images:
        raise click.ClickException("each snapshot must contain one record per image")
    batches = []
    for start in range(0, len(rows), images):
        band = rows[start : start + images]
        ids = [r["request"] for r in band]
        if ids != list(range(ids[0], ids[0] + images)):
            raise click.ClickException(
                "each snapshot requires consecutive solver requests"
            )
        batches.append(
            {
                "requests": [ids[0], ids[-1]],
                "first_image": action_drive(band, mass, target_energy, width, False),
                "adjacent_images": action_drive(band, mass, target_energy, width, True),
            }
        )
    click.echo(
        json.dumps(
            {
                "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
                "target_energy_eV": target_energy,
                "absolute_width_eV": width,
                "atomic_masses_amu": mass,
                "additional_surface_requests": 0,
                "batches": batches,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
