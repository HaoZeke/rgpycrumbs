import math

import numpy as np
import pytest

from rgpycrumbs.eon.harmonic_diffusivity import log10_diffusivity_cm2_per_s
from rgpycrumbs.kinetics import harmonic_log_rates


def test_log10_diffusivity_matches_linear_definition():
    rate = math.log(1.5e12)
    hop_angstrom = 2.0
    paths = 4.0
    dimension = 3.0
    expected = math.log10(paths * (hop_angstrom * 1e-8) ** 2 * math.exp(rate) / (2.0 * dimension))
    assert log10_diffusivity_cm2_per_s(rate, hop_angstrom, paths, dimension) == pytest.approx(
        expected, rel=1e-12
    )


def test_rejects_nonpositive_geometry():
    with pytest.raises(ValueError, match="hop length"):
        log10_diffusivity_cm2_per_s(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="path multiplicity"):
        log10_diffusivity_cm2_per_s(0.0, 1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="lattice dimension"):
        log10_diffusivity_cm2_per_s(0.0, 1.0, 1.0, 0.0)


def test_array_rates_preserve_temperature_shape():
    temperatures = np.array([200.0, 400.0, 800.0])
    rates = harmonic_log_rates([500, 1200], [-600, 900], 0.3, temperatures)
    classical = log10_diffusivity_cm2_per_s(rates.classical, 1.5, 8.0, 2.0)
    assert classical.shape == temperatures.shape
    singles = [
        log10_diffusivity_cm2_per_s(
            harmonic_log_rates([500, 1200], [-600, 900], 0.3, t).classical, 1.5, 8.0, 2.0
        )
        for t in temperatures
    ]
    np.testing.assert_allclose(classical, singles)
