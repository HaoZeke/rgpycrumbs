import numpy as np
import pytest

from rgpycrumbs.eon.orbit_residual import euclidean_force_residual


@pytest.mark.interpolation
def test_two_thermal_modes_balance_independent_of_constant_translation():
    from scipy.constants import Boltzmann, atomic_mass, electron_volt, hbar

    temperature = 137.0
    masses = np.array([1.0, 16.0])
    phase = 2 * np.pi * np.arange(15)[:, None] / 15
    coordinates = np.cos(phase * np.array([1, 3]))
    omega = 2 * np.pi * Boltzmann * temperature / hbar
    force = coordinates * masses * (omega * np.array([1, 3])) ** 2 * atomic_mass * 1e-20 / electron_volt
    np.testing.assert_allclose(
        euclidean_force_residual(coordinates + 5, force, masses, temperature), 0, atol=1e-10
    )
    shifted = force.copy()
    shifted[:, 1] += 0.01
    residual = euclidean_force_residual(coordinates, shifted, masses, temperature)
    np.testing.assert_allclose(residual[:, 1], 0.01, atol=1e-10)
