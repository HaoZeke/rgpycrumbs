from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("align")

from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402

from rgpycrumbs.geom.api.alignment import (  # noqa: E402
    AlignmentMethod,
    IRAConfig,
    align_structure_robust,
    ira_mod,
)

pytestmark = pytest.mark.align

requires_ira = pytest.mark.skipif(
    ira_mod is None, reason="IRA module not found in the current environment"
)


@pytest.fixture
def water_molecule():
    """Returns a standard water molecule for reference."""
    return molecule("H2O")


@pytest.fixture
def rotated_water(water_molecule):
    """Returns a water molecule rotated by 90 degrees."""
    mobile = water_molecule.copy()
    mobile.rotate(90, "z")
    return mobile


@pytest.fixture
def permuted_water(water_molecule):
    """Returns a water molecule with swapped hydrogen indices."""
    mobile = water_molecule.copy()
    # Swap the two hydrogen atoms (indices 1 and 2)
    indices = [0, 2, 1]
    permuted = Atoms(
        symbols=[mobile.get_chemical_symbols()[i] for i in indices],
        positions=mobile.positions[indices],
    )
    return permuted


class TestStructuralAlignment:
    """Focuses on the physical consistency of atomic properties during permutation."""

    def test_identity_alignment(self, water_molecule):
        """Checks if alignment of a structure with itself returns correct status."""
        ref = water_molecule.copy()
        mobile = water_molecule.copy()
        config = IRAConfig(enabled=False)

        result = align_structure_robust(ref, mobile, config)

        assert result.method == AlignmentMethod.ASE_PROCRUSTES
        assert np.allclose(mobile.positions, ref.positions)

    def test_ase_fallback_rotation(self, water_molecule, rotated_water):
        """Verifies that ASE Procrustes handles rotation when IRA is disabled."""
        config = IRAConfig(enabled=False)

        result = align_structure_robust(water_molecule, rotated_water, config)

        assert result.method == AlignmentMethod.ASE_PROCRUSTES
        assert np.allclose(rotated_water.positions, water_molecule.positions, atol=1e-5)

    @requires_ira
    def test_ira_permutation_success(self, water_molecule, permuted_water):
        """
        IRA match to verify permutation handling.
        """
        result = align_structure_robust(
            water_molecule, permuted_water, IRAConfig(enabled=True)
        )
        assert result.method == AlignmentMethod.IRA_PERMUTATION
        # The permuted water should now match the reference positions AND atomic order
        assert np.allclose(permuted_water.positions, water_molecule.positions)
        assert list(permuted_water.get_chemical_symbols()) == list(
            water_molecule.get_chemical_symbols()
        )

    @patch("rgpycrumbs.geom.api.alignment.ira_mod")
    def test_ira_failure_fallback(self, mock_ira_mod, water_molecule, rotated_water):
        """Ensures the code falls back to ASE if the IRA library raises an exception."""
        mock_ira_instance = MagicMock()
        mock_ira_mod.IRA.return_value = mock_ira_instance
        mock_ira_instance.match.side_effect = Exception("IRA Internal Error")

        config = IRAConfig(enabled=True)

        # Should catch exception and use ASE
        result = align_structure_robust(water_molecule, rotated_water, config)

        assert result.method == AlignmentMethod.ASE_PROCRUSTES
        assert np.allclose(rotated_water.positions, water_molecule.positions, atol=1e-5)

    @requires_ira
    def test_ase_fails_on_permutation_but_ira_succeeds(self, water_molecule):
        # Create a permuted water molecule
        indices = [0, 2, 1]
        permuted_water = Atoms(
            symbols=[water_molecule.get_chemical_symbols()[i] for i in indices],
            positions=water_molecule.positions[indices],
        )

        # Break the C2v symmetry by slightly nudging one hydrogen atom.
        # This prevents a pure 180-degree rotation from achieving zero RMSD.
        permuted_water.positions[1] += [0.05, 0.05, 0.0]

        # 3. Force ASE to handle the permuted/distorted water (IRA disabled)
        config_ase = IRAConfig(enabled=False)
        result_ase = align_structure_robust(
            water_molecule, permuted_water.copy(), config_ase
        )

        # ASE will minimize the error via rotation but cannot fix the local distortion
        # and the index mismatch simultaneously.
        ase_dist = np.linalg.norm(result_ase.atoms.positions - water_molecule.positions)

        # Let IRA handle the permuted water
        config_ira = IRAConfig(enabled=True)
        result_ira = align_structure_robust(
            water_molecule, permuted_water.copy(), config_ira
        )

        # IRA will first fix the permutation, then the subsequent alignment
        # will only see the small 0.05 displacement.
        ira_dist = np.linalg.norm(result_ira.atoms.positions - water_molecule.positions)

        # Assertions
        assert result_ase.method == AlignmentMethod.ASE_PROCRUSTES
        assert result_ira.method == AlignmentMethod.IRA_PERMUTATION

        # IRA should still produce a lower error because it correctly identifies
        # which hydrogen corresponds to the reference positions.
        assert ira_dist < ase_dist

    @requires_ira
    def test_mass_and_identity_sync(self, water_molecule):
        """
        Ensures that masses and identities follow the permutation.
        If we align a deuterated water where the D and H are swapped,
        the resulting structure must have the D at the correct reference index.
        """
        # 1. Setup reference: O at 0, H at 1, H at 2
        ref = water_molecule.copy()

        # 2. Setup mobile: Swap O and one H, and make that H a Deuterium
        # Original: [O, H, H] -> Permuted: [H(D), H, O]
        # Indices: [1, 2, 0]
        indices = [1, 2, 0]
        mobile = Atoms(symbols=["H", "H", "O"], positions=ref.positions[indices])
        # Set mass of the first atom (index 0) to Deuterium (~2.014)
        mobile.set_masses([2.014, 1.008, 15.999])

        # Slightly rotate to make it a real matching problem
        mobile.rotate(15, "x")

        config = IRAConfig(enabled=True)
        result = align_structure_robust(ref, mobile, config)

        assert result.method == AlignmentMethod.IRA_PERMUTATION

        # Check if the positions match the reference
        assert np.allclose(mobile.positions, ref.positions, atol=1e-5)

        # Check if atomic species were reordered to match reference [O, H, H]
        assert list(mobile.get_chemical_symbols()) == ["O", "H", "H"]

        # Check if the Deuterium mass followed the permutation.
        # The Deuterium was at index 0 of the mobile structure.
        # In the reference, O is at 0, H at 1, H at 2.
        # The mobile atom that was at index 0 (Deuterium) should now be at index 1 or 2.
        masses = mobile.get_masses()
        assert np.isclose(masses[0], 15.999, atol=1e-3)
        # One of the hydrogens must be the Deuterium
        assert any(np.isclose(m, 2.014, atol=1e-3) for m in masses[1:])

    @requires_ira
    def test_in_place_modification_verification(self, water_molecule):
        """
        Verifies that the original object is modified in-place.
        """
        ref = water_molecule.copy()
        mobile = water_molecule.copy()

        # Manually permute mobile
        mobile = mobile[[0, 2, 1]]
        mobile.rotate(45, "z")

        # Store the object ID to ensure it doesn't change
        original_id = id(mobile)

        align_structure_robust(ref, mobile, IRAConfig(enabled=True))

        # The ID should remain the same (in-place modification)
        assert id(mobile) == original_id
        # The content should be aligned
        assert np.allclose(mobile.positions, ref.positions, atol=1e-5)

    @requires_ira
    def test_subset_matching_nat1_less_than_nat2(self):
        """
        Verifies behavior when ref structure is a subset of the mobile structure.
        IRA supports nat1 <= nat2.
        """
        # Ref: Water (3 atoms)
        ref = molecule("H2O")

        # Mobile: Methane (5 atoms) - IRA should attempt to find the H2O shape within it
        # Note: This is a stress test for the permutation logic.
        mobile = molecule("CH4")

        # Move them apart
        mobile.translate([5.0, 5.0, 5.0])

        config = IRAConfig(enabled=True)
        # Should not crash and should return a result or handle the mismatch gracefully
        try:
            result = align_structure_robust(ref, mobile, config)
            # If IRA succeeds, the first 3 atoms of mobile should now match H2O
            if result.method == AlignmentMethod.IRA_PERMUTATION:
                assert len(mobile) == 5  # Total count remains the same
                assert list(mobile.get_chemical_symbols()[:3]) == ["O", "H", "H"]
        except ValueError:
            # If libira raises an error due to no congruent match found, that is also valid.
            pass
