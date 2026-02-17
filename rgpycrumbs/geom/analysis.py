import ase
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from scipy.spatial.distance import cdist


def analyze_structure(
    atoms: ase.Atoms, covalent_scale: float = 1.2
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Analyzes an ASE Atoms object to calculate the distance matrix, bond matrix,
    and identify molecular fragments.

    Args:
        atoms: The ASE Atoms object to analyze.
        covalent_scale:  A scaling factor applied to the covalent radii to
            determine the bonding threshold.  A bond is formed if the distance
            between two atoms is less than or equal to the sum of their
            covalent radii, multiplied by this factor.

    Returns:
        A tuple containing:
        - distance_matrix: A NumPy array representing the pairwise distances
          between all atoms in the system.
        - bond_matrix: A NumPy array representing the adjacency matrix of the
          molecular graph.  bond_matrix[i, j] = 1 if atoms i and j are
          bonded, and 0 otherwise.
        - fragments: A list of lists, where each inner list contains the indices
          of the atoms belonging to a connected fragment.
    """

    num_atoms = len(atoms)

    # 1. Calculate the distance matrix.
    distance_matrix = atoms.get_all_distances(
        mic=True
    )  # Use minimum image convention (MIC)

    # 2. Calculate the bond matrix.
    bond_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    radii = [covalent_radii[atom.number] for atom in atoms]  # Atomic numbers

    # More efficient with neighborlist
    neighbor_list = NeighborList(
        cutoffs=[covalent_scale * r for r in radii],  # Scaled cutoffs
        skin=0.0,  # No skin, we use the cutoffs directly
        bothways=True,  # Get both i->j and j->i
        self_interaction=False,  # No i->i connections
        primitive=ase.neighborlist.NewPrimitiveNeighborList,
    )
    neighbor_list.update(atoms)

    for i in range(num_atoms):
        indices, offsets = neighbor_list.get_neighbors(i)
        for j, _ in zip(indices, offsets, strict=False):
            bond_matrix[i, j] = 1

    # 3. Identify fragments.
    visited = [False] * num_atoms
    fragments = []

    def dfs(atom_index: int, current_fragment: list[int]):
        """Depth-First Search to find connected components."""
        visited[atom_index] = True
        current_fragment.append(atom_index)
        for neighbor_index in range(num_atoms):
            if (
                bond_matrix[atom_index, neighbor_index] == 1
                and not visited[neighbor_index]
            ):
                dfs(neighbor_index, current_fragment)

    for i in range(num_atoms):
        if not visited[i]:
            new_fragment = []
            dfs(i, new_fragment)
            fragments.append(new_fragment)

    # 4. Calculate Centroid Distances and *Corrected* Distances.
    num_fragments = len(fragments)
    centroid_distances = np.zeros((num_fragments, num_fragments))
    corrected_distances = []  # Initialize with infinity
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_symbols = atoms.get_chemical_symbols()

    if num_fragments > 1:
        for i in range(num_fragments):
            frag_i_positions = positions[fragments[i]]
            centroid_i = np.mean(frag_i_positions, axis=0)

            for j in range(i + 1, num_fragments):  # Iterate only over i < j
                frag_j_positions = positions[fragments[j]]
                centroid_j = np.mean(frag_j_positions, axis=0)
                centroid_dist = np.linalg.norm(centroid_i - centroid_j)
                centroid_distances[i, j] = centroid_distances[j, i] = centroid_dist

                # Find closest pair of atoms and calculate corrected distance
                all_distances = cdist(frag_i_positions, frag_j_positions)
                min_dist = np.min(all_distances)
                min_index_i, min_index_j = np.unravel_index(
                    np.argmin(all_distances), all_distances.shape
                )

                atom_i_number = atomic_numbers[fragments[i][min_index_i]]
                atom_j_number = atomic_numbers[fragments[j][min_index_j]]
                atom_i_symbol = atomic_symbols[fragments[i][min_index_i]]
                atom_j_symbol = atomic_symbols[fragments[j][min_index_j]]
                covrad_sum = float(
                    covalent_radii[atom_i_number] + covalent_radii[atom_j_number]
                )

                corrected_distances.append(
                    (
                        float(min_dist),
                        atom_i_symbol,
                        atom_j_symbol,
                        covrad_sum,
                        float(min_dist - covrad_sum * covalent_scale),
                    )
                )

    return (
        distance_matrix,
        bond_matrix,
        fragments,
        centroid_distances,
        corrected_distances,
    )
