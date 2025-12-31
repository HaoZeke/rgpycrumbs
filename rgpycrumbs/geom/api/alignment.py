import logging

from ase import Atoms
from ase.build import minimize_rotation_and_translation

# Attempt to import IRA once at the module level
try:
    from rgpycrumbs._aux import _import_from_parent_env

    ira_mod = _import_from_parent_env("ira_mod")
except ImportError:
    ira_mod = None


def align_structure_robust(
    ref_atoms: Atoms,
    mobile_atoms: Atoms,
    use_ira: bool = False,
    ira_kmax: float = 1.8,
) -> tuple[Atoms, bool]:
    """
    Aligns a mobile structure to a reference using IRA with an ASE fallback.

    This method minimizes the RMSD between the reference and mobile structures.
    It first attempts to solve the isomorphism problem (finding $P, R, t$)
    using IRA. If IRA fails or remains unavailable, it defaults to standard
    Procrustes superimposition (finding $R, t$) via ASE.

    :param ref_atoms: The fixed reference configuration.
    :param mobile_atoms: The configuration to align (modified in-place).
    :param use_ira: Boolean flag to enable permutation invariance.
    :param ira_kmax: The adjacency cutoff distance for IRA graph matching.
    :return: A tuple containing the modified mobile_atoms and a boolean
             indicating if IRA successfully handled the alignment.
    """
    # Create a copy if specific preservation is required,
    # but ASE standardizes on in-place modification.

    aligned_successfully_with_ira = False

    if use_ira and ira_mod:
        try:
            ira_instance = ira_mod.IRA()

            # IRA returns rotation ($R$), translation ($t$), and permutation
            # ($P$), along with the Hausdorff distance (hd) (unused here)
            r_mat, t_vec, p_vec, _ = ira_instance.match(
                len(ref_atoms),
                ref_atoms.get_atomic_numbers(),
                ref_atoms.get_positions(),
                len(mobile_atoms),
                mobile_atoms.get_atomic_numbers(),
                mobile_atoms.get_positions(),
                ira_kmax,
            )

            # Apply the affine transformation: $x' = xR^T + t$
            new_pos = (mobile_atoms.get_positions() @ r_mat.T) + t_vec

            # Apply the permutation vector $P$ to reorder indices
            mobile_atoms.set_positions(new_pos[p_vec])
            mobile_atoms.set_atomic_numbers(mobile_atoms.get_atomic_numbers()[p_vec])

            aligned_successfully_with_ira = True

        except Exception as e:
            logging.debug(f"IRA alignment failed: {e}. Proceeding to fallback.")

    if not aligned_successfully_with_ira:
        # Fallback: Rigid body alignment via SVD (Procrustes) without reordering
        minimize_rotation_and_translation(ref_atoms, mobile_atoms)

    return mobile_atoms, aligned_successfully_with_ira
