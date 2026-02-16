import numpy as np
import pytest

from tests.conftest import skip_if_not_env


class TestIRADataStructures:
    """Tests for IRA data structures that don't require ira_mod."""

    pytestmark = pytest.mark.pure

    def test_iracomp_dataclass(self):
        """IRAComp should store rotation, translation, permutation, and Hausdorff distance."""
        skip_if_not_env("ira")
        from rgpycrumbs.geom.ira import IRAComp

        comp = IRAComp(
            rot=np.eye(3),
            trans=np.zeros(3),
            perm=np.array([0, 1, 2]),
            hd=0.01,
        )
        assert comp.rot.shape == (3, 3)
        assert comp.trans.shape == (3,)
        assert comp.perm.shape == (3,)
        assert comp.hd == 0.01

    def test_incomparable_structures_error(self):
        """IncomparableStructuresError should be a ValueError subclass."""
        skip_if_not_env("ira")
        from rgpycrumbs.geom.ira import IncomparableStructuresError

        assert issubclass(IncomparableStructuresError, ValueError)
        with pytest.raises(IncomparableStructuresError):
            raise IncomparableStructuresError("test message")


class TestIRAMatching:
    """Tests for IRA matching functions that require ira_mod and ase."""

    pytestmark = pytest.mark.ira

    @pytest.fixture
    def h2_molecule(self):
        """A simple H2 molecule."""
        skip_if_not_env("ira")
        from ase.atoms import Atoms

        return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    @pytest.fixture
    def h2_translated(self, h2_molecule):
        """H2 molecule translated by [1, 0, 0]."""
        shifted = h2_molecule.copy()
        shifted.translate([1.0, 0.0, 0.0])
        return shifted

    @pytest.fixture
    def h2_permuted(self):
        """H2 molecule with swapped atom indices."""
        from ase.atoms import Atoms

        return Atoms("H2", positions=[[0, 0, 0.74], [0, 0, 0]])

    def test_perform_ira_match_different_lengths(self):
        """Should raise IncomparableStructuresError for different-length structures."""
        skip_if_not_env("ira")
        from ase.atoms import Atoms

        from rgpycrumbs.geom.ira import IncomparableStructuresError, _perform_ira_match

        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        h3 = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.74], [0, 0.74, 0]])
        with pytest.raises(IncomparableStructuresError, match="same number of atoms"):
            _perform_ira_match(h2, h3)

    def test_perform_ira_match_different_species(self):
        """Should raise IncomparableStructuresError for different atom types."""
        skip_if_not_env("ira")
        from ase.atoms import Atoms

        from rgpycrumbs.geom.ira import IncomparableStructuresError, _perform_ira_match

        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        he2 = Atoms("He2", positions=[[0, 0, 0], [0, 0, 0.74]])
        with pytest.raises(IncomparableStructuresError, match="same atom types"):
            _perform_ira_match(h2, he2)

    def test_is_ira_pair_identical(self, h2_molecule):
        """Identical structures should be an IRA pair."""
        skip_if_not_env("ira")
        from rgpycrumbs.geom.ira import is_ira_pair

        assert is_ira_pair(h2_molecule, h2_molecule.copy())

    def test_is_ira_pair_incompatible_returns_false(self):
        """Incompatible structures should return False, not raise."""
        skip_if_not_env("ira")
        from ase.atoms import Atoms

        from rgpycrumbs.geom.ira import is_ira_pair

        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        he2 = Atoms("He2", positions=[[0, 0, 0], [0, 0, 0.74]])
        assert is_ira_pair(h2, he2) is False

    def test_do_ira_returns_iracomp(self, h2_molecule):
        """do_ira should return an IRAComp dataclass."""
        skip_if_not_env("ira")
        from rgpycrumbs.geom.ira import IRAComp, do_ira

        result = do_ira(h2_molecule, h2_molecule.copy())
        assert isinstance(result, IRAComp)
        assert result.rot.shape == (3, 3)
        assert result.hd >= 0

    def test_calculate_rmsd_identical_zero(self, h2_molecule):
        """RMSD of identical structures should be near zero."""
        skip_if_not_env("ira")
        from rgpycrumbs.geom.ira import calculate_rmsd

        rmsd = calculate_rmsd(h2_molecule, h2_molecule.copy())
        assert rmsd < 1e-6

    def test_calculate_rmsd_permuted_near_zero(self, h2_molecule, h2_permuted):
        """RMSD should be near zero for permuted identical structures."""
        skip_if_not_env("ira")
        from rgpycrumbs.geom.ira import calculate_rmsd

        rmsd = calculate_rmsd(h2_molecule, h2_permuted)
        assert rmsd < 1e-3
