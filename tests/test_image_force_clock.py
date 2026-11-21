import numpy as np
import pytest


@pytest.mark.interpolation
def test_projected_force_integral_recovers_a_mass_weighted_harmonic_orbit():
    from rgpycrumbs.eon.image_force_clock import compare_tangents

    q = np.array([[-1.0], [-0.3], [0.1], [0.6], [1.0]])
    energy = 1 - q[:, 0] ** 2
    result = compare_tangents(q, energy, -2 * q, [2.0], 0.0)
    for name in ("cubic_coordinate_derivative", "higher_energy_chord"):
        orbit = result[name]
        assert orbit["half_period"] == pytest.approx(np.pi, abs=1e-9)
        assert orbit["distance_from_opposite_arc"] < 1e-12
        assert orbit["max_cartesian_displacement"] < 1e-12
    assert result["path_arc_length"] == pytest.approx(2 * np.sqrt(2))
