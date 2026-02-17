import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("pure")

pytestmark = pytest.mark.pure


def test_spline_interp_basic():
    """Test that spline interpolation returns expected shapes."""
    from rgpycrumbs.interpolation import spline_interp

    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)

    x_fine, y_fine = spline_interp(x, y, num=50)
    assert len(x_fine) == 50
    assert len(y_fine) == 50
    assert x_fine[0] == pytest.approx(x[0])
    assert x_fine[-1] == pytest.approx(x[-1])


def test_spline_interp_roundtrip():
    """Test that interpolation of a polynomial recovers exact values."""
    from rgpycrumbs.interpolation import spline_interp

    x = np.linspace(0, 5, 30)
    y = 2 * x**2 - 3 * x + 1  # quadratic

    x_fine, y_fine = spline_interp(x, y, num=100)
    y_expected = 2 * x_fine**2 - 3 * x_fine + 1

    np.testing.assert_allclose(y_fine, y_expected, atol=1e-2)
