"""Normalize an instanton flux for hopping among equivalent fixed-lattice sites."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "scipy"]
# ///

import json
from pathlib import Path

import click
import numpy as np
from scipy import constants

from rgpycrumbs.eon.harmonic_diffusivity import log10_diffusivity_cm2_per_s
from rgpycrumbs.kinetics import log_finite_bead_partition


def lattice_diffusivity(level, minimum, seed, hop_length, path_multiplicity, dimension):
    """Normalize the flux in the mobile Cartesian subspace of a fixed lattice."""
    rate = level["rate"]
    mobile = minimum["mobile_atoms"]
    masses = np.asarray(seed["masses"], dtype=float)
    if (not mobile or len(set(mobile)) != len(mobile) or min(mobile) < 0
            or seed["active_atoms"] != mobile or masses.shape != (len(mobile),)
            or not np.all(np.isfinite(masses) & (masses > 0))):
        raise ValueError("seed masses and ordered mobile atom indices must match the spectrum")
    frequencies = np.asarray(minimum["signed_wavenumbers_cm-1"], dtype=float)
    if frequencies.shape != (3 * len(mobile),) or not np.all(np.isfinite(frequencies) & (frequencies > 0)):
        raise ValueError("expected three positive reactant frequencies per mobile atom")
    reference = minimum["electronic_energy_eV"]
    if not np.allclose([seed["reference_energy"], level["reference_energy"]], reference, rtol=0, atol=1e-7):
        raise ValueError("seed, flux and spectrum must share the reactant electronic reference")
    beads = rate["beads"]
    if beads < 3 or rate["negative_modes"] != 1 or rate["modes"] != beads * len(frequencies) - 1:
        raise ValueError("expected one path mode removed and one retained negative mode")
    temperature = rate["temperature"]
    thermal = level["controls"]["boltzmann"] * temperature
    hbar = level["controls"]["hbar"]
    if not np.isfinite([thermal, temperature, hbar]).all() or min(thermal, temperature, hbar) <= 0:
        raise ValueError("expected positive thermal and action scales")
    time_s = 1e-10 * np.sqrt(constants.atomic_mass / constants.electron_volt)
    omega = 2 * np.pi * constants.c * 100 * frequencies * time_s
    log_q = log_finite_bead_partition(hbar * omega / thermal, beads)
    log_flux = rate["log_k_times_Zr"] + reference / thermal
    log_rate = log_flux - log_q - np.log(time_s)
    log_d = log10_diffusivity_cm2_per_s(log_rate, hop_length, path_multiplicity, dimension)
    return dict(temperature_K=temperature, beads=beads, mobile_atoms=mobile,
                masses_amu=masses.tolist(), retained_modes=rate["modes"], negative_modes=1,
                log_Q_vib=log_q, log_flux_reactant_electronic_zero=log_flux,
                time_unit_s=time_s, log10_hop_rate_per_s=log_rate / np.log(10),
                hop_length_angstrom=hop_length, path_multiplicity=path_multiplicity,
                lattice_dimension=dimension, log10_diffusivity_cm2_per_s=float(log_d))


@click.command()
@click.argument("levels", type=click.Path(exists=True, path_type=Path))
@click.argument("reactant", type=click.Path(exists=True, path_type=Path))
@click.argument("seed", type=click.Path(exists=True, path_type=Path))
@click.option("--index", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--hop-length", type=float, required=True, help="Hop length in angstrom.")
@click.option("--path-multiplicity", type=click.IntRange(min=1), required=True)
@click.option("--lattice-dimension", type=click.IntRange(min=1), required=True)
def main(levels, reactant, seed, index, hop_length, path_multiplicity, lattice_dimension):
    """Convert a fixed-lattice LI level, reactant spectrum and native seed to diffusivity."""
    try:
        result = lattice_diffusivity(json.loads(levels.read_text().splitlines()[index]),
                                     json.loads(reactant.read_text()), json.loads(seed.read_text()),
                                     hop_length, path_multiplicity, lattice_dimension)
        click.echo(json.dumps(result, indent=2, allow_nan=False))
    except (KeyError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
