import numpy as np
import pytest

from rgpycrumbs.instanton.orbit import reconstruct_orbit

pytestmark = pytest.mark.pure


def test_reconstruction_finds_crossing_between_fluctuation_nodes():
    k, beta = 26, 4.0
    theta = np.linspace(0, np.pi, 2 * k + 1)
    x = (np.cos(24 * theta) - np.cos(8 * theta))[:, None]
    energies = x[:, 0] ** 2
    assert np.max(energies[::13]) == pytest.approx(0, abs=1e-25)
    spectrum = np.arange(k) * (2 * np.pi / beta)
    gradients = (
        -(spectrum[24] ** 2) * np.cos(24 * theta) + spectrum[8] ** 2 * np.cos(8 * theta)
    )[:, None]
    audit = reconstruct_orbit(
        x, energies, gradients, [1], beta=beta, cosines=k, reference=0
    )
    assert audit.coefficients[24, 0] == pytest.approx(1)
    assert audit.residual < 1e-10
    assert audit.action == pytest.approx(beta + beta / 4 * sum(spectrum[[8, 24]] ** 2))
    assert audit.max_potential_rise > 2


def test_rejects_off_basis_geometry():
    theta = np.linspace(0, np.pi, 9)
    with pytest.raises(ValueError, match="cosine basis"):
        reconstruct_orbit(
            np.cos(4 * theta)[:, None],
            np.zeros(9),
            np.zeros((9, 1)),
            [1],
            beta=1,
            cosines=4,
            reference=0,
        )
