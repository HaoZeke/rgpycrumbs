import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("align")

import numpy as np  # noqa: E402
from ase.atoms import Atoms  # noqa: E402

from rgpycrumbs.geom.analysis import analyze_structure  # noqa: E402

pytestmark = pytest.mark.align


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def water_molecule():
    """Creates a single water molecule with correct O-H bond lengths (~0.96 A)."""
    return Atoms("OH2", positions=[[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])


@pytest.fixture
def water_dimer():
    """Creates two water molecules separated by 5 Angstroms."""
    h2o = Atoms("OH2", positions=[[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])
    dimer = h2o.copy()
    h2o_2 = h2o.copy()
    h2o_2.translate([5.0, 0, 0])
    dimer.extend(h2o_2)
    return dimer


@pytest.fixture
def single_atom():
    """A single hydrogen atom."""
    return Atoms("H", positions=[[0, 0, 0]])


@pytest.fixture
def h2_molecule():
    """A simple H2 molecule with a typical bond length."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def three_fragments():
    """Three water molecules well-separated."""
    h2o = Atoms("OH2", positions=[[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])
    system = h2o.copy()
    h2o_2 = h2o.copy()
    h2o_2.translate([5.0, 0, 0])
    system.extend(h2o_2)
    h2o_3 = h2o.copy()
    h2o_3.translate([0, 5.0, 0])
    system.extend(h2o_3)
    return system


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_analyze_single_molecule(water_molecule):
    """A single molecule should have one fragment."""
    dist_mat, bond_mat, frags, _, _ = analyze_structure(water_molecule)
    assert dist_mat.shape == (3, 3)
    assert bond_mat.shape == (3, 3)
    assert len(frags) == 1
    assert sorted(frags[0]) == [0, 1, 2]


def test_analyze_dimer_two_fragments(water_dimer):
    """Two well-separated molecules should be identified as two fragments."""
    _, _, frags, centroid_dists, corrected_dists = analyze_structure(water_dimer)
    assert len(frags) == 2
    assert len(frags[0]) == 3
    assert len(frags[1]) == 3
    # Centroid distance should be approximately 5.0
    assert centroid_dists[0, 1] > 4.0


def test_analyze_three_fragments(three_fragments):
    """Three well-separated molecules should yield three fragments."""
    _, _, frags, centroid_dists, corrected_dists = analyze_structure(three_fragments)
    assert len(frags) == 3
    for frag in frags:
        assert len(frag) == 3  # each water has 3 atoms


# ---------------------------------------------------------------------------
# Matrix property tests
# ---------------------------------------------------------------------------


def test_distance_matrix_symmetric(water_molecule):
    """Distance matrix should be symmetric."""
    dist_mat, _, _, _, _ = analyze_structure(water_molecule)
    np.testing.assert_allclose(dist_mat, dist_mat.T, atol=1e-10)


def test_bond_matrix_symmetric(water_molecule):
    """Bond matrix should be symmetric."""
    _, bond_mat, _, _, _ = analyze_structure(water_molecule)
    np.testing.assert_array_equal(bond_mat, bond_mat.T)


def test_distance_matrix_zero_diagonal(water_molecule):
    """Diagonal of distance matrix should be zero (self-distance)."""
    dist_mat, _, _, _, _ = analyze_structure(water_molecule)
    np.testing.assert_allclose(np.diag(dist_mat), 0.0, atol=1e-10)


def test_bond_matrix_zero_diagonal(water_molecule):
    """Diagonal of bond matrix should be zero (no self-bonds)."""
    _, bond_mat, _, _, _ = analyze_structure(water_molecule)
    np.testing.assert_array_equal(np.diag(bond_mat), 0)


def test_distance_matrix_nonnegative(water_dimer):
    """All distances should be non-negative."""
    dist_mat, _, _, _, _ = analyze_structure(water_dimer)
    assert np.all(dist_mat >= 0)


def test_bond_matrix_binary(water_dimer):
    """Bond matrix entries should be 0 or 1."""
    _, bond_mat, _, _, _ = analyze_structure(water_dimer)
    unique_vals = np.unique(bond_mat)
    assert set(unique_vals).issubset({0, 1})


# ---------------------------------------------------------------------------
# Single atom edge case
# ---------------------------------------------------------------------------


def test_single_atom(single_atom):
    """A single atom should yield one fragment with no bonds."""
    dist_mat, bond_mat, frags, centroid_dists, corrected_dists = analyze_structure(
        single_atom
    )
    assert dist_mat.shape == (1, 1)
    assert bond_mat.shape == (1, 1)
    assert len(frags) == 1
    assert frags[0] == [0]
    assert bond_mat[0, 0] == 0


# ---------------------------------------------------------------------------
# Fragment index coverage
# ---------------------------------------------------------------------------


def test_all_atoms_in_fragments(water_dimer):
    """Every atom index should appear exactly once across all fragments."""
    _, _, frags, _, _ = analyze_structure(water_dimer)
    all_indices = sorted(idx for frag in frags for idx in frag)
    assert all_indices == list(range(len(water_dimer)))


# ---------------------------------------------------------------------------
# Covalent scale parameter
# ---------------------------------------------------------------------------


def test_covalent_scale_affects_bonding(h2_molecule):
    """Increasing covalent scale should not decrease the number of bonds."""
    _, bond_tight, _, _, _ = analyze_structure(h2_molecule, covalent_scale=0.5)
    _, bond_loose, _, _, _ = analyze_structure(h2_molecule, covalent_scale=2.0)
    assert np.sum(bond_loose) >= np.sum(bond_tight)


# ---------------------------------------------------------------------------
# Corrected distances
# ---------------------------------------------------------------------------


def test_corrected_distances_structure(water_dimer):
    """Corrected distances should return a list of 5-tuples for multi-fragment systems."""
    _, _, frags, _, corrected_dists = analyze_structure(water_dimer)
    assert len(frags) == 2
    assert len(corrected_dists) == 1  # one pair: fragment 0 vs fragment 1
    # Each entry: (min_dist, symbol_i, symbol_j, covrad_sum, corrected)
    entry = corrected_dists[0]
    assert len(entry) == 5
    assert isinstance(entry[0], float)  # min_dist
    assert isinstance(entry[1], str)  # atom symbol
    assert isinstance(entry[2], str)  # atom symbol
    assert isinstance(entry[3], float)  # covrad_sum
    assert isinstance(entry[4], float)  # corrected distance


def test_centroid_distances_symmetric(three_fragments):
    """Centroid distance matrix should be symmetric."""
    _, _, _, centroid_dists, _ = analyze_structure(three_fragments)
    np.testing.assert_allclose(centroid_dists, centroid_dists.T, atol=1e-10)
