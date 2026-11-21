import numpy as np
import pytest

from rgpycrumbs.eon.orbit_residual import euclidean_force_residual, orbit_extent


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
    spring = np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) - 2 * x
    force = -spring * masses * atomic_mass * 1e-20 / electron_volt / dt**2
    np.testing.assert_allclose(
        euclidean_force_residual(x, force, masses, temperature, ring_polymer=True),
        0,
        atol=1e-10,
    )


@pytest.mark.interpolation
def test_orbit_extent_distinguishes_a_collapsed_ring_at_zero_force():
    stationary = np.zeros((6, 2, 3))
    assert orbit_extent(stationary, [1.0, 4.0]) == {
        "distinct_geometries": 1,
        "max_atom_excursion_angstrom": 0.0,
        "mass_weighted_diameter_angstrom_sqrt_u": 0.0,
    }
    stationary[1, 0, 0] = 3.0
    stationary[1, 1, 2] = 2.0
    extent = orbit_extent(stationary + 19.0, [1.0, 4.0])
    assert extent["distinct_geometries"] == 2
    assert extent["max_atom_excursion_angstrom"] == pytest.approx(3.0)
    assert extent["mass_weighted_diameter_angstrom_sqrt_u"] == pytest.approx(5.0)


@pytest.mark.interpolation
def test_measured_orbit_maps_gradient_ids_and_uses_the_recorded_period():
    from types import SimpleNamespace
    from scipy.constants import atomic_mass, electron_volt
    from rgpycrumbs.eon.orbit_residual import observed_eon_forces

    atoms = [SimpleNamespace(atom_id=i, fixed=[False]*3) for i in (7, 2)]
    lineb = {"beads": 4, "temperature": 32., "half_period": 1.5}
    gradients = np.array([[1., 2., 3.], [-4., -5., -6.]])
    observed = {"mobile_atom_ids": [2, 7], "gradients_eV_per_angstrom": gradients.tolist()}
    frame = SimpleNamespace(atoms=atoms, metadata={"lineb": lineb, "lineb_orbit": observed})
    forces, period, source = observed_eon_forces([frame] * 4)
    np.testing.assert_array_equal(forces, np.broadcast_to(-gradients[::-1], (4, 2, 3)))
    assert period == pytest.approx(3 * 1e-10 * np.sqrt(atomic_mass / electron_volt), rel=1e-14, abs=0)
    assert source == "twice the recorded eOn half-period"


@pytest.mark.interpolation
def test_force_covector_projection_respects_masses_and_turning_points():
    from rgpycrumbs.eon.orbit_residual import decompose_fourier_residual

    angle = 2 * np.pi * np.arange(12) / 12
    positions = np.column_stack((np.cos(angle), np.cos(angle)))
    residual = np.broadcast_to([1., 0.], positions.shape)
    result = decompose_fourier_residual(positions, residual, np.array([1., 4.]))
    assert result["zero_velocity_beads"] == [0, 6]
    assert result["nonzero_velocity_beads"] == 10
    assert result["normal"]["max_eV_per_angstrom"] == pytest.approx(0.8)
    assert result["tangential"]["max_eV_per_angstrom"] == pytest.approx(0.8)
    assert result["normal_fraction_squared_inverse_mass_norm"] == pytest.approx(0.8)
