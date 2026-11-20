import math

import pytest

from rgpycrumbs.kinetics import harmonic_log_rates


def test_one_mode_and_high_temperature_limit():
    result = harmonic_log_rates([100], [-200], 0.0, 1e7)
    assert result.classical == pytest.approx(math.log(29979245800 * 100))
    assert result.quasi_quantum == pytest.approx(result.classical, abs=1e-9)


def test_saddle_index_and_cold_rate():
    with pytest.raises(ValueError, match="one unstable mode"):
        harmonic_log_rates([100, 200], [-100, -200], 0.5, 50)
    assert math.isfinite(harmonic_log_rates([3200], [-1000], 1.5, 1).quasi_quantum)


def test_temperature_array_matches_scalar_calls():
    import numpy as np

    temperatures = np.array([50.0, 300.0, 1500.0])
    rates = harmonic_log_rates([500, 1200], [-600, 900], 0.3, temperatures)
    singles = [harmonic_log_rates([500, 1200], [-600, 900], 0.3, t) for t in temperatures]
    np.testing.assert_allclose(rates, np.array(singles).T)
