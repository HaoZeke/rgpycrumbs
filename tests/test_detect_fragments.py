from tests.conftest import skip_if_not_env

skip_if_not_env("fragments")

import pytest  # noqa: E402
from ase.atoms import Atoms  # noqa: E402

from rgpycrumbs.geom.detect_fragments import (  # noqa: E402
    build_graph_and_find_components,
    find_fragments_bond_order,
    find_fragments_geometric,
    merge_fragments_by_distance,
)

pytestmark = pytest.mark.fragments

@pytest.fixture
def nitrogen_molecule():
    """Creates a nitrogen molecule with a standard triple bond length."""
    return Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.1]])


@pytest.fixture
def water_dimer():
    """Creates two water molecules with a 3.0 Angstrom separation."""
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0.7, 0.7], [0, -0.7, 0.7]])
    dimer = h2o.copy()
    h2o_2 = h2o.copy()
    h2o_2.translate([3.0, 0, 0])
    dimer.extend(h2o_2)
    return dimer


def test_geometric_connectivity(water_dimer):
    """Verifies that geometric detection yields two fragments."""
    n, labels = find_fragments_geometric(water_dimer, bond_multiplier=1.2)
    assert n == 2
    assert labels[0] != labels[3]


def test_bond_order_calculation(nitrogen_molecule):
    """Performs a live GFN2-xTB calculation to verify WBO detection."""
    # A triple bond should have a WBO near 3.0
    n, _, indices, matrix = find_fragments_bond_order(
        nitrogen_molecule, threshold=0.5, charge=0, multiplicity=1
    )
    assert n == 1
    assert matrix[0, 1] > 2.5
    assert len(indices) == 1


def test_merge_by_centroid(water_dimer):
    """Tests the merging of fragments based on spatial basins."""
    n_init, labels_init = find_fragments_geometric(water_dimer, 1.2)
    n_final, _ = merge_fragments_by_distance(
        water_dimer, n_init, labels_init, min_dist=5.0
    )
    assert n_final == 1


def test_sparse_graph_resolution():
    """Validates the connected components logic for a chain."""
    rows, cols = [0, 1], [1, 2]
    n, labels = build_graph_and_find_components(4, rows, cols)
    assert n == 2
    assert labels[0] == labels[2]
    assert labels[3] != labels[0]
