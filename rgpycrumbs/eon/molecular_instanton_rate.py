"""Normalize a molecular instanton flux using a finite-bead harmonic reactant."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "scipy"]
# ///

import json
from pathlib import Path

import click
import numpy as np
from scipy import constants


def log_inertia_det(positions, masses):
    """Inertia of a molecule or full ring, with each replica weighted by 1/P."""
    positions = np.asarray(positions, dtype=float).reshape(-1, len(masses), 3)
    center = np.average(positions.mean(axis=0), axis=0, weights=masses)
    centered = positions - center
    second = np.einsum("pni,n,pnj->ij", centered, masses, centered) / len(positions)
    moments = np.linalg.eigvalsh(np.trace(second) * np.eye(3) - second)
    if not np.all(np.isfinite(moments) & (moments > 0)):
        raise ValueError("expected a nonlinear molecule with positive moments of inertia")
    return np.log(moments).sum()


def molecular_rate(level, minimum, atoms, ring):
    """Return a single-channel rate in s⁻¹ for an unconstrained nonlinear molecule."""
    rate = level["rate"]
    p = rate["beads"]
    temperature = rate["temperature"]
    masses = atoms.get_masses()
    freq = np.asarray(minimum["signed_wavenumbers_cm-1"], dtype=float)
    if (p < 3 or ring.size != p * 3 * len(masses) or rate["modes"] != p * 3 * len(masses) - 7
            or not np.isfinite(ring).all() or not np.all(np.isfinite(masses) & (masses > 0))):
        raise ValueError("expected a full molecular ring with six rigid modes and one path mode removed")
    if freq.shape != (3 * len(masses) - 6,) or not np.all(np.isfinite(freq) & (freq > 0)):
        raise ValueError("expected positive internal reactant wavenumbers")
    if not np.isclose(level["reference_energy"], minimum["electronic_energy_eV"], atol=1e-7, rtol=0):
        raise ValueError("reactant and flux must use the same electronic energy reference")
    controls = level["controls"]
    time_s = 1e-10 * np.sqrt(constants.atomic_mass / constants.electron_volt)
    beta = 1 / (controls["boltzmann"] * temperature)
    if not np.isfinite(beta) or beta <= 0 or controls["hbar"] <= 0:
        raise ValueError("expected positive thermal energy and Planck constant")
    # Each internal oscillator contributes [2 sinh(P asinh(beta hbar omega / 2P))]^-1.
    omega = 2 * np.pi * constants.c * 100 * freq * time_s
    arg = p * np.arcsinh(beta * controls["hbar"] * omega / (2 * p))
    log_q_vib = -np.sum(arg + np.log(-np.expm1(-2 * arg)))
    # For each rigid direction the non-centroid spring frequencies yield
    # product_{k=1}^{P-1} [2 sin(pi k/P)]^-1 = 1/P.
    log_q_rigid_springs = -6 * np.log(p)
    log_rotation = 0.5 * (log_inertia_det(ring, masses) - log_inertia_det(atoms.positions, masses))
    log_flux = rate["log_k_times_Zr"] + beta * minimum["electronic_energy_eV"]
    log_rate = log_flux - log_q_vib - log_q_rigid_springs + log_rotation - np.log(time_s)
    if not np.isfinite(log_rate):
        raise ValueError("nonfinite normalized rate")
    return dict(temperature_K=temperature, beads=p, log10_rate_per_s=log_rate / np.log(10),
                log_flux_reactant_electronic_zero=log_flux, log_Q_vib=log_q_vib,
                log_Q_rigid_springs=log_q_rigid_springs, log_rotation_ratio=log_rotation,
                time_unit_s=time_s, path_multiplicity=1)


@click.command()
@click.argument("level_file", type=click.Path(exists=True, path_type=Path))
@click.argument("reactant_spectrum", type=click.Path(exists=True, path_type=Path))
@click.argument("reactant_extxyz", type=click.Path(exists=True, path_type=Path))
@click.argument("beads_json", type=click.Path(exists=True, path_type=Path))
@click.option("--index", type=click.IntRange(min=0), default=0, show_default=True)
def main(level_file, reactant_spectrum, reactant_extxyz, beads_json, index):
    """Normalize one LI-NEB JSONL level using its full ordered Cartesian ring."""
    from ase.io import read

    level = json.loads(level_file.read_text().splitlines()[index])
    minimum = json.loads(reactant_spectrum.read_text())
    atoms = read(reactant_extxyz, format="extxyz")
    ring = np.array(json.loads(beads_json.read_text()), dtype=float)
    click.echo(json.dumps(molecular_rate(level, minimum, atoms, ring), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
