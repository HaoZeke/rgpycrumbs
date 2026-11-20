import numpy as np
from ase import Atoms
from scipy import constants

from rgpycrumbs.eon.molecular_instanton_rate import molecular_rate


def test_absolute_rate_uses_explicit_ring_partition_and_cancels_energy_zero():
    atoms = Atoms("H2O", positions=[[0.75, 0.5, 0], [-0.75, 0.5, 0], [0, 0, 0]])
    p, t, hbar, kb = 12, 275.0, 0.06466, constants.value("Boltzmann constant in eV/K")
    modes = [900.0, 1800.0, 3000.0]
    minimum = {"signed_wavenumbers_cm-1": modes, "electronic_energy_eV": 0.0}
    ring = np.tile(atoms.positions, (p, 1, 1))
    level = {"controls": {"boltzmann": kb, "hbar": hbar}, "reference_energy": 0.0,
             "rate": {"temperature": t, "beads": p, "modes": 9 * p - 7, "log_k_times_Zr": -52.0}}
    time_unit = 1e-10 * np.sqrt(constants.atomic_mass / constants.electron_volt)
    quantum = 2 * np.pi * constants.c * 100 * np.array(modes) * time_unit * hbar / (p * kb * t)
    spring_terms = 4 * np.sin(np.pi * np.arange(p) / p) ** 2
    log_q = -0.5 * np.log(spring_terms[:, None] + quantum**2).sum() - 3 * np.log(spring_terms[1:]).sum()
    expected = (-52.0 - log_q - np.log(time_unit)) / np.log(10)
    actual = molecular_rate(level, minimum, atoms, ring)["log10_rate_per_s"]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    shift = -124.5
    level["reference_energy"] = minimum["electronic_energy_eV"] = shift
    level["rate"]["log_k_times_Zr"] -= shift / (kb * t)
    ring = 2 * ring + np.array([2, 3, 4])
    actual = molecular_rate(level, minimum, atoms, ring)["log10_rate_per_s"]
    np.testing.assert_allclose(actual, expected + np.log10(8), atol=1e-11)
