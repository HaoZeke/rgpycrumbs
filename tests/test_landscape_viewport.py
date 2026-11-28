import numpy as np

from rgpycrumbs.eon.plt_neb import square_landscape_viewport


def test_viewport_is_square_and_covers_the_data():
    s = np.array([0.0, 0.3, 0.6, 1.2])
    d = np.array([0.0, 2.9, 1.7, 0.0])
    (x0, x1), (y0, y1) = square_landscape_viewport(s, d)
    assert np.isclose(x1 - x0, y1 - y0)
    assert y0 == -y1
    assert x0 < s.min() and x1 > s.max()
    assert y1 >= 2.9 * 1.12


def test_markers_widen_the_window():
    s = np.array([0.0, 1.0])
    d = np.array([0.0, 0.5])
    (_, _), (_, y1) = square_landscape_viewport(s, d, marker_d=[-4.0])
    assert np.isclose(y1, 4.0 * 1.12)


def test_wide_paths_keep_their_s_range():
    s = np.array([0.0, 10.0])
    d = np.array([0.0, 0.1])
    (x0, x1), (y0, y1) = square_landscape_viewport(s, d)
    assert x0 < 0.0 and x1 > 10.0
    assert np.isclose(x1 - x0, y1 - y0)
