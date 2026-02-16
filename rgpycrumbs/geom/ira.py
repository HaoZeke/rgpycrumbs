# This is a separate module because building ira_mod [1] needs a Makefile and a
# PYTHONPATH export.. after that's cleared up this will probably be a dependency
# [1]: https://github.com/mammasmias/IterativeRotationsAssignments
import dataclasses

import ira_mod
import numpy as np
from collections import Counter


@dataclasses.dataclass
class IRAComp:
    rot: np.array
    trans: np.array
    perm: np.array
    hd: float


class IncomparableStructuresError(ValueError):
    """Custom exception raised for incompatible atomistic structures."""

    pass


def _perform_ira_match(atm1, atm2, k_factor=2.8):
    """Performs IRA matching. Internal function."""
    if len(atm1) != len(atm2):
        errmsg = "Atomistic structures must have the same number of atoms."
        raise IncomparableStructuresError(errmsg)
    if not Counter(atm1.symbols) == Counter(atm2.symbols):
        errmsg = "Atomistic structures must have the same atom types."
        raise IncomparableStructuresError(errmsg)

    ira = ira_mod.IRA()
    return ira.match(
        len(atm1),
        atm1.get_atomic_numbers(),
        atm1.get_positions(),
        len(atm2),
        atm2.get_atomic_numbers(),
        atm2.get_positions(),
        k_factor,
    )


def is_ira_pair(atm1, atm2, hd_tol=1, k_factor=2.8):
    """Checks if two atomistic structures are an IRA pair."""
    try:
        _, _, _, hd = _perform_ira_match(atm1, atm2, k_factor)
        return hd < hd_tol
    except IncomparableStructuresError:
        return False


def do_ira(atm1, atm2, k_factor=2.8):
    """Performs IRA matching on two atomistic structures."""
    rotation, translation, perm, hd = _perform_ira_match(atm1, atm2, k_factor)
    return IRAComp(rot=rotation, trans=translation, perm=perm, hd=hd)


def calculate_rmsd(atm1, atm2, k_factor=2.8):
    """Calculates RMSD using do_ira."""
    ira_comp = do_ira(atm1, atm2, k_factor)
    atm2_coords_permuted = atm2.get_positions()[ira_comp.perm]
    atm2_coords_transformed = (
        np.dot(atm2_coords_permuted, ira_comp.rot) + ira_comp.trans
    )
    n_atoms = len(atm1.get_positions())
    squared_distances = np.sum(
        (atm1.get_positions() - atm2_coords_transformed) ** 2, axis=1
    )
    rmsd = np.sqrt(np.sum(squared_distances) / n_atoms)
    return rmsd
