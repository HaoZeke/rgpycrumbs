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
        positions=mobile.get_positions()[indices],
    )
    return permuted


class TestStructuralAlignment:
    def test_identity_alignment(self, water_molecule):
        """Checks if alignment of a structure with itself returns correct status."""
        ref = water_molecule.copy()
        mobile = water_molecule.copy()
        config = IRAConfig(enabled=False)

        result = align_structure_robust(ref, mobile, config)

        assert result.method == AlignmentMethod.ASE_PROCRUSTES
        assert np.allclose(mobile.get_positions(), ref.get_positions())

    def test_ase_fallback_rotation(self, water_molecule, rotated_water):
        """Verifies that ASE Procrustes handles rotation when IRA is disabled."""
        config = IRAConfig(enabled=False)

        result = align_structure_robust(water_molecule, rotated_water, config)

        assert result.method == AlignmentMethod.ASE_PROCRUSTES
        assert np.allclose(
            rotated_water.get_positions(), water_molecule.get_positions(), atol=1e-5
        )

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
        assert np.allclose(
            permuted_water.get_positions(), water_molecule.get_positions()
        )
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
        assert np.allclose(
            rotated_water.get_positions(), water_molecule.get_positions(), atol=1e-5
        )

    @requires_ira
    def test_ase_fails_on_permutation_but_ira_succeeds(self, water_molecule):
        # Create a permuted water molecule
        indices = [0, 2, 1]
        permuted_water = Atoms(
            symbols=[water_molecule.get_chemical_symbols()[i] for i in indices],
            positions=water_molecule.get_positions()[indices],
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
