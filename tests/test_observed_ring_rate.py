import numpy as np
import pytest


def test_projected_logdet_matches_dense_signed_spectrum():
    from scipy.sparse import csc_matrix
    from rgpycrumbs.orca.observed_ring_rate import projected_logdet

    basis = np.array([[1., 1.], [1., -1.], [0., 0.], [0., 0.]])
    matrix = csc_matrix(np.diag([10., 12., -2., 3.]))
    assert projected_logdet(matrix, basis) == pytest.approx(np.log(6.))
    with pytest.raises(ValueError, match="independent"):
        projected_logdet(matrix, np.column_stack([basis[:, 0], basis[:, 0]]))
