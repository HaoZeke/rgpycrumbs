import pytest

from rgpycrumbs.eon.lineb_replay import differences


def test_gradient_difference_reports_component_and_rejects_broadcast():
    a = [[0, 0, 1], [0, 0, -1]]
    b = [[0, 0, 1], [0, 0.25, -1]]
    assert differences(a, b) == {"shape": [2, 3], "max_abs": 0.25, "index": [1, 1]}
    with pytest.raises(ValueError, match="shape"):
        differences(a, b[:1])
