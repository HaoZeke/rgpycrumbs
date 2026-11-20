"""Map harmonic hop rates onto log10 diffusivity and compare with a reference series.

Prefer the dispatcher::

    rgpycrumbs eon harmonic-diffusivity reactant.json saddle.json reference.csv \\
        --hop-length 2.5 --path-multiplicity 12 --lattice-dimension 3
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click", "numpy", "scipy",
# ]
# ///

import json
from pathlib import Path

import click
import numpy as np

from rgpycrumbs.kinetics import harmonic_log_rates

# 1 Å = 10⁻⁸ cm; log10 of the squared conversion is exact in decimal.
_LOG10_ANGSTROM2_TO_CM2 = -16.0


def log10_diffusivity_cm2_per_s(
    log_rate_per_s,
    hop_length_angstrom: float,
    path_multiplicity: float,
    lattice_dimension: float,
):
    """Convert natural-log hop rates (s⁻¹) to log10 diffusivity (cm²/s).

    Uses ``D = n λ² Γ / (2 d)`` with hop length ``λ`` in angstrom, path
    multiplicity ``n``, and lattice dimension ``d``. Arithmetic stays in
    log10 space.
    """
    hop = float(hop_length_angstrom)
    paths = float(path_multiplicity)
    dimension = float(lattice_dimension)
    if not np.isfinite(hop) or hop <= 0:
        raise ValueError("hop length must be positive and finite")
    if not np.isfinite(paths) or paths <= 0:
        raise ValueError("path multiplicity must be positive and finite")
    if not np.isfinite(dimension) or dimension <= 0:
        raise ValueError("lattice dimension must be positive and finite")
    log10_rate = np.asarray(log_rate_per_s, dtype=float) / np.log(10.0)
    log10_geometric = (
        np.log10(paths)
        + 2.0 * np.log10(hop)
        + _LOG10_ANGSTROM2_TO_CM2
        - np.log10(2.0 * dimension)
    )
    return log10_geometric + log10_rate


@click.command()
@click.argument("reactant", type=click.Path(exists=True, path_type=Path))
@click.argument("saddle", type=click.Path(exists=True, path_type=Path))
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.option("--hop-length", type=float, required=True, help="Positive hop length in angstrom.")
@click.option(
    "--path-multiplicity",
    type=float,
    required=True,
    help="Positive number of equivalent hop paths.",
)
@click.option(
    "--lattice-dimension",
    type=float,
    required=True,
    help="Positive lattice dimensionality in the diffusivity prefactor.",
)
def main(reactant, saddle, reference, hop_length, path_multiplicity, lattice_dimension):
    """Compare HTST and qq-HTST log10 diffusivities with a reference CSV.

    Vibration records match ``harmonic-rates`` (electronic_energy_eV and
    signed_wavenumbers_cm-1). The CSV columns are
    inverse_temperature_1000_per_K and log10_diffusivity_cm2_per_s.
    """
    minimum, transition = (json.loads(p.read_text()) for p in (reactant, saddle))
    points = np.genfromtxt(reference, delimiter=",", names=True, ndmin=1)
    inverse_temperature = points["inverse_temperature_1000_per_K"]
    temperatures = 1000.0 / np.asarray(inverse_temperature, dtype=float)
    rates = harmonic_log_rates(
        minimum["signed_wavenumbers_cm-1"],
        transition["signed_wavenumbers_cm-1"],
        transition["electronic_energy_eV"] - minimum["electronic_energy_eV"],
        temperatures,
    )
    classical = log10_diffusivity_cm2_per_s(
        rates.classical, hop_length, path_multiplicity, lattice_dimension
    )
    quantum = log10_diffusivity_cm2_per_s(
        rates.quasi_quantum, hop_length, path_multiplicity, lattice_dimension
    )
    reference_values = points["log10_diffusivity_cm2_per_s"]
    table = np.column_stack(
        (
            inverse_temperature,
            reference_values,
            classical,
            classical - reference_values,
            quantum,
            quantum - reference_values,
        )
    )
    np.savetxt(
        click.get_text_stream("stdout"),
        table,
        delimiter=",",
        fmt="%.9g",
        comments="",
        header=(
            "inverse_temperature_1000_per_K,log10_diffusivity_cm2_per_s,"
            "htst_log10_diffusivity_cm2_per_s,htst_difference,"
            "qq_htst_log10_diffusivity_cm2_per_s,qq_htst_difference"
        ),
    )


if __name__ == "__main__":
    main()
