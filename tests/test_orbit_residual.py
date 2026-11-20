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
    force = (
        coordinates
        * masses
        * (omega * np.array([1, 3])) ** 2
        * atomic_mass
        * 1e-20
        / electron_volt
    )
    np.testing.assert_allclose(
        euclidean_force_residual(coordinates + 5, force, masses, temperature),
        0,
        atol=1e-10,
    )
    shifted = force.copy()
    shifted[:, 1] += 0.01
    residual = euclidean_force_residual(coordinates, shifted, masses, temperature)
    np.testing.assert_allclose(residual[:, 1], 0.01, atol=1e-10)


@pytest.mark.interpolation
def test_finite_bead_spring_balances_the_discrete_force():
    from scipy.constants import Boltzmann, atomic_mass, electron_volt, hbar

    temperature = 230.0
    x = np.array([0.3, 0.8, 1.4, 1.4, 0.8, 0.3])[:, None]
    masses = np.array([2.0])
    dt = hbar / (len(x) * Boltzmann * temperature)
    spring = (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) - 2 * x)
    force = -spring * masses * atomic_mass * 1e-20 / electron_volt / dt**2
    np.testing.assert_allclose(
        euclidean_force_residual(x, force, masses, temperature, ring_polymer=True),
        0, atol=1e-10,
    )
