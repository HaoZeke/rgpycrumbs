"""
Core fragment detection algorithms for molecular systems.

Provides geometric (scaled covalent radii) and bond-order (GFN2-xTB) based
fragment detection, plus distance-based fragment merging.

.. versionadded:: 0.0.6
"""

import logging
from enum import StrEnum

import numpy as np
from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from ase.units import Bohr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from rgpycrumbs._aux import _import_from_parent_env

tbliteinterface = _import_from_parent_env("tblite.interface")


class DetectionMethod(StrEnum):
    """Available detection methodologies.

    .. versionadded:: 0.0.6
    """

    GEOMETRIC = "geometric"
    BOND_ORDER = "bond-order"


DEFAULT_BOND_MULTIPLIER = 1.2
DEFAULT_BOND_ORDER_THRESHOLD = 0.8


def find_fragments_geometric(
    atoms: Atoms, bond_multiplier: float, radius_type: str = "natural"
) -> tuple[int, np.ndarray]:
    """Detect molecular fragments using scaled covalent radii.

    .. versionadded:: 0.0.6
    """
    num_atoms = len(atoms)
    if num_atoms == 0:
        return 0, np.array([])

    # Selection of radii generation strategy
    if radius_type == "covalent":
        # Direct usage of ASE standard covalent radii
        # We apply the multiplier directly to these radii
        cutoffs = covalent_radii[atoms.get_atomic_numbers()] * bond_multiplier
    else:
        # Default to ASE 'natural' cutoffs (Cordero parameters)
        # natural_cutoffs handles the multiplier internally
        cutoffs = natural_cutoffs(atoms, mult=bond_multiplier)

    nl = build_neighbor_list(atoms, cutoffs=cutoffs, self_interaction=False)

    row_indices, col_indices = [], []
    for i in range(num_atoms):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            if i < j:
                row_indices.append(i)
                col_indices.append(j)

    return build_graph_and_find_components(num_atoms, row_indices, col_indices)


def find_fragments_bond_order(
    atoms: Atoms,
    threshold: float,
    charge: int,
    multiplicity: int,
    method: str = "GFN2-xTB",
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze connectivity via the Wiberg Bond Order (WBO) matrix.
    Calculate electronic structure using the specified xTB level.

    .. versionadded:: 0.0.6
    """
    num_atoms = len(atoms)
    if num_atoms == 0:
        return 0, np.array([]), np.array([]), np.array([])

    logging.info(f"Running {method} for {atoms.get_chemical_formula(mode='hill')}...")

    # Initialize the calculator with the chosen xTB method
    calc = tbliteinterface.Calculator(
        method=method,
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions() / Bohr,
        charge=float(charge),
        uhf=int(multiplicity - 1),
    )

    results = calc.singlepoint()
    bond_order_matrix = results.get("bond-orders")

    if bond_order_matrix is None:
        rerr = f"The method {method} did not return bond orders."
        raise ValueError(rerr)

    # WBO matrix analysis
    # k=1 excludes the diagonal (self-interactions/valency)
    indices = np.argwhere(np.triu(bond_order_matrix, k=1) > threshold)
    row_indices, col_indices = indices[:, 0], indices[:, 1]

    n_components, labels = build_graph_and_find_components(
        num_atoms, row_indices.tolist(), col_indices.tolist()
    )
    return n_components, labels, indices, bond_order_matrix


def build_graph_and_find_components(
    num_atoms: int,
    row_indices: np.ndarray | list[int],
    col_indices: np.ndarray | list[int],
) -> tuple[int, np.ndarray]:
    """
    Identify connected components using direct CSR sparse matrix construction.

    .. versionadded:: 0.0.6

    This function avoids Python list overhead by passing interaction indices
    directly to the SciPy sparse engine.
    """
    # Convert inputs to numpy arrays to ensure efficient slicing and memory access
    rows = np.asarray(row_indices)
    cols = np.asarray(col_indices)

    if rows.size == 0:
        return num_atoms, np.arange(num_atoms)

    # Define bond weights as a simple integer array
    # Using int8 saves memory for large systems
    data = np.ones(rows.size, dtype=np.int8)

    # Construct the Compressed Sparse Row matrix
    # SciPy handles the undirected nature when directed=False
    adj = csr_matrix((data, (rows, cols)), shape=(num_atoms, num_atoms))

    # Calculate connected components using the Laplacian-based graph traversal
    return connected_components(csgraph=adj, directed=False, return_labels=True)


def merge_fragments_by_distance(
    atoms: Atoms, n_components: int, labels: np.ndarray, min_dist: float
) -> tuple[int, np.ndarray]:
    """Merges fragments with geometric centers closer than the specified distance.

    .. versionadded:: 0.0.6
    """
    if n_components <= 1:
        return n_components, labels

    centers = np.array(
        [atoms.positions[labels == i].mean(axis=0) for i in range(n_components)]
    )
    row_indices, col_indices = [], []
    for i in range(n_components):
        for j in range(i + 1, n_components):
            if np.linalg.norm(centers[i] - centers[j]) < min_dist:
                row_indices.append(i)
                col_indices.append(j)

    if not row_indices:
        return n_components, labels

    fragment_adj = csr_matrix(
        (
            np.ones(len(row_indices) * 2),
            (
                np.concatenate([row_indices, col_indices]),
                np.concatenate([col_indices, row_indices]),
            ),
        ),
        shape=(n_components, n_components),
    )
    new_n, merge_labels = connected_components(
        fragment_adj, directed=False, return_labels=True
    )

    final_labels = -np.ones_like(labels)
    for i in range(n_components):
        final_labels[np.where(labels == i)[0]] = merge_labels[i]

    return new_n, final_labels
