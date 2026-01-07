import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation

# Attempt to import IRA once at the module level
try:
    from rgpycrumbs._aux import _import_from_parent_env

    ira_mod = _import_from_parent_env("ira_mod")
except ImportError:
    ira_mod = None


@dataclass(frozen=True)
class IRAMatchInputs:
    """Encapsulates the raw arrays required for an IRA graph match."""

    ref_count: int
    ref_numbers: np.ndarray
    ref_positions: np.ndarray
    mobile_count: int
    mobile_numbers: np.ndarray
    mobile_positions: np.ndarray
    kmax: float


@dataclass(frozen=True)
class IRAMatchResults:
    """Encapsulates the transformation outputs from the IRA algorithm."""

    rotation: np.ndarray  # $R$ matrix
    translation: np.ndarray  # $t$ vector
    permutation: np.ndarray  # $P$ index mapping
    hausdorff_dist: float  # $hd$ metric


@dataclass(frozen=True)
class IRAConfig:
    """Configuration parameters for the Iterative Rotations and Alignment."""

    enabled: bool = False
    kmax: float = 1.8


class AlignmentMethod(Enum):
    """Tracks which algorithm was successfully applied."""

    ASE_PROCRUSTES = auto()  # Standard rigid rotation/translation
    IRA_PERMUTATION = auto()  # Isomorphic mapping + alignment
    NONE = auto()  # No operation performed


@dataclass
class AlignmentResult:
    """
    Container for alignment outputs.

    :param atoms: The aligned structure (modified in-place, but returned for chaining).
    :param method: The specific algorithm that was used.
    """

    atoms: Atoms
    method: AlignmentMethod

    @property
    def used_ira(self) -> bool:
        """Helper to maintain backward compatibility with boolean checks."""
        return self.method == AlignmentMethod.IRA_PERMUTATION


def _apply_ira_alignment(
    ref_atoms: Atoms, mobile_atoms: Atoms, config: IRAConfig
) -> bool:
    """
    Performs the low-level IRA isomorphism and affine transformation.

    This function modifies mobile_atoms in-place. It returns True if the
    alignment succeeds, or False if an error occurs or IRA remains unavailable.
    """
    if not (config.enabled and ira_mod):
        return False

    if len(ref_atoms) > len(mobile_atoms):
        return False

    ira_instance = ira_mod.IRA()

    inputs = IRAMatchInputs(
        ref_count=len(ref_atoms),
        ref_numbers=ref_atoms.get_atomic_numbers(),
        ref_positions=ref_atoms.get_positions(),
        mobile_count=len(mobile_atoms),
        mobile_numbers=mobile_atoms.get_atomic_numbers(),
        mobile_positions=mobile_atoms.get_positions(),
        kmax=config.kmax,
    )

    raw_output = ira_instance.match(
        inputs.ref_count,
        inputs.ref_numbers,
        inputs.ref_positions,
        inputs.mobile_count,
        inputs.mobile_numbers,
        inputs.mobile_positions,
        inputs.kmax,
    )
    res = IRAMatchResults(*raw_output)

    # Apply transformation: $x' = xR^T + t$
    transformed_pos = (mobile_atoms.get_positions() @ res.rotation.T) + res.translation
    # Set positions and identities based on the permutation vector $P$
    p = res.permutation
    mobile_atoms.positions = transformed_pos[p]
    mobile_atoms.set_atomic_numbers(mobile_atoms.get_atomic_numbers()[p])
    mobile_atoms.set_masses(mobile_atoms.get_masses()[p])
    if mobile_atoms.get_velocities() is not None:
        v_rotated = mobile_atoms.get_velocities() @ res.rotation.T
        mobile_atoms.set_velocities(v_rotated[p])
    return True


def align_structure_robust(
    ref_atoms: Atoms,
    mobile_atoms: Atoms,
    ira_config: IRAConfig,
) -> AlignmentResult:
    """
    Aligns a mobile structure to a reference using IRA with an ASE fallback.

    This method minimizes the RMSD between the reference and mobile structures.
    It first attempts to solve the isomorphism problem (finding $P, R, t$)
    using IRA. If IRA fails or remains unavailable, it defaults to standard
    Procrustes superimposition (finding $R, t$) via ASE.

    :param ref_atoms: The fixed reference configuration.
    :param mobile_atoms: The configuration to align (modified in-place).
    :param ira_config: Configuration object with state of IRA and the adjacency
                       cutoff distance for IRA graph matching.
    :return: An AlignmentResult.
    """
    try:
        if _apply_ira_alignment(ref_atoms, mobile_atoms, ira_config):
            return AlignmentResult(
                atoms=mobile_atoms, method=AlignmentMethod.IRA_PERMUTATION
            )

    except Exception as e:
        logging.debug(f"IRA alignment failed: {e}. Proceeding to fallback.")
    # Fallback to standard rigid superimposition
    minimize_rotation_and_translation(ref_atoms, mobile_atoms)
    return AlignmentResult(atoms=mobile_atoms, method=AlignmentMethod.ASE_PROCRUSTES)
