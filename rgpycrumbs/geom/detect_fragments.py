#!/usr/bin/env python3
"""
Detects molecular fragments in coordinate files using two distinct methodologies:
1. Geometric: Utilizes scaled covalent radii.
2. Bond Order: Employs GFN2-xTB semi-empirical calculations.

The tool supports fragment merging based on centroid proximity and batch
processing for high-throughput computational chemistry workflows.

Usage for a single file:
uv run python detect_fragments.py geometric your_file.xyz --multiplier 1.1
uv run python detect_fragments.py bond-order your_file.xyz --threshold 0.7 --min-dist 4.0

Usage for a directory (batch mode):
uv run python detect_fragments.py batch ./your_folder/ --method geometric --min-dist 3.5
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase~=3.23",
#     "click~=8.1",
#     "numpy~=1.26",
#     "rich~=13.7",
#     "scipy~=1.14",
#     "pyvista~=0.43",
#     "matplotlib~=3.9",
#     "cmcrameri~=1.8",
# ]
# ///

import csv
import logging
from enum import StrEnum
from pathlib import Path

import click
import cmcrameri.cm as cmcrameri_cm
import matplotlib as mpl
import numpy as np
import pyvista as pv
from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.io import read
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from ase.units import Bohr
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from rgpycrumbs._aux import _import_from_parent_env

mpl.colormaps.register(cmcrameri_cm.batlow, force=True)
cmap_name = "batlow"


tbliteinterface = _import_from_parent_env("tblite.interface")

# --- Setup ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)


class DetectionMethod(StrEnum):
    """Available detection methodologies."""

    GEOMETRIC = "geometric"
    BOND_ORDER = "bond-order"


DEFAULT_BOND_MULTIPLIER = 1.2
DEFAULT_BOND_ORDER_THRESHOLD = 0.8

# Plot Settings
SCALAR_BAR_ARGS = {
    "title": "Wiberg Bond Order",
    "vertical": True,
    "position_x": 0.85,  # Slightly away from the right edge
    "position_y": 0.05,  # Start near the bottom
    "height": 0.9,  # Stretch to cover 90% of the window height
    "width": 0.05,  # Adjust thickness as needed
    "title_font_size": 20,
    "label_font_size": 16,
}

MIN_DIST_ATM = 1e-4
# --- Core Logic Functions ---


def find_fragments_geometric(
    atoms: Atoms, bond_multiplier: float, radius_type: str = "natural"
) -> tuple[int, np.ndarray]:
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
    """Merges fragments with geometric centers closer than the specified distance."""
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


# --- Visualization ---


def visualize_with_pyvista(
    atoms: Atoms,
    method: DetectionMethod,
    bond_data: float | np.ndarray,
    nonbond_cutoff: float = 0.05,
    bond_threshold: float = 0.8,
    radius_type: str = "natural",
) -> None:
    """Renders the molecular system with scalar-coded bond orders."""
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background("white")

    # CPK Colors
    cpk_colors = {
        1: "#FFFFFF",
        6: "#b5b5b5",
        7: "#0000FF",
        8: "#FF0000",
        9: "#90E050",
        15: "#FF8000",
        16: "#FFFF00",
        17: "#00FF00",
        35: "#A62929",
        53: "#940094",
    }
    default_color = "#FFC0CB"

    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    radii = covalent_radii[nums] * 0.45

    # Render Atoms
    for i, (p, n) in enumerate(zip(pos, nums)):
        sphere = pv.Sphere(
            radius=radii[i], center=p, theta_resolution=24, phi_resolution=24
        )
        plotter.add_mesh(
            sphere,
            color=cpk_colors.get(n, default_color),
            specular=0.5,
            smooth_shading=True,
        )

    # Render Bonds based on Method
    if method == DetectionMethod.GEOMETRIC:
        multiplier = float(bond_data)
        if radius_type == "covalent":
            cutoffs = covalent_radii[atoms.get_atomic_numbers()] * multiplier
        else:
            cutoffs = natural_cutoffs(atoms, mult=multiplier)
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, self_interaction=False)

        for i in range(len(atoms)):
            indices, _ = nl.get_neighbors(i)
            for j in indices:
                if i < j:
                    p1, p2 = pos[i], pos[j]
                    cyl = pv.Cylinder(
                        center=(p1 + p2) / 2,
                        direction=p2 - p1,
                        radius=0.15,
                        height=np.linalg.norm(p2 - p1),
                    )
                    plotter.add_mesh(cyl, color="darkgrey", specular=0.2)

    elif method == DetectionMethod.BOND_ORDER:
        matrix = bond_data
        # Ensure matrix is a numpy array
        matrix = np.asarray(matrix)

        # Identify pairs above threshold
        indices = np.argwhere(np.triu(matrix, k=1) > nonbond_cutoff)

        if indices.size == 0:
            logging.warning("No interactions found above cutoff.")
            plotter.show()
            return

        visible_wbo = matrix[indices[:, 0], indices[:, 1]]
        min_bo, max_bo = visible_wbo.min(), visible_wbo.max()

        # Avoid division by zero if all bond orders are equal
        bo_range = max_bo - min_bo if max_bo > min_bo else 1.0

        bonded_meshes = []
        weak_meshes = []

        for idx_pair in indices:
            i, j = idx_pair
            wbo = matrix[i, j]
            p1, p2 = pos[i], pos[j]
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            # Skip overlapping atoms
            if dist < MIN_DIST_ATM:
                continue

            if wbo >= bond_threshold:
                # Normalize radius: stronger bonds appear thicker
                norm_bo = np.clip((wbo - min_bo) / bo_range, 0.0, 1.0)
                radius = 0.08 + (0.01 * norm_bo)

                cyl = pv.Cylinder(
                    center=(p1 + p2) / 2,
                    direction=vec,
                    radius=radius,
                    height=dist,
                    resolution=15,
                )
                # Assign scalar to points for smoother rendering
                cyl.point_data["WBO"] = np.full(cyl.n_points, wbo)
                bonded_meshes.append(cyl)
            else:
                # Weak interaction dots
                n_dots = max(2, int(dist / 0.2))
                for k in range(n_dots + 1):
                    dot_pos = p1 + (k / n_dots) * vec
                    dot = pv.Sphere(radius=0.04, center=dot_pos)
                    dot.point_data["WBO"] = np.full(dot.n_points, wbo)
                    weak_meshes.append(dot)

        # Merge and Add to Plotter
        if bonded_meshes:
            plotter.add_mesh(
                pv.merge(bonded_meshes),
                scalars="WBO",
                cmap="batlow",
                clim=[min_bo, max_bo],
                smooth_shading=True,
                scalar_bar_args=SCALAR_BAR_ARGS,
            )

        if weak_meshes:
            plotter.add_mesh(
                pv.merge(weak_meshes),
                scalars="WBO",
                cmap="batlow",
                clim=[min_bo, max_bo],
                opacity=0.6,
                show_scalar_bar=False,
            )

    logging.info("Opening visualization...")
    plotter.show()


# --- CLI and Batch ---


def print_results(
    console: Console, atoms: Atoms, n_components: int, labels: np.ndarray
) -> None:
    """Displays analysis results in a structured table."""
    console.rule("[bold green]Analysis Summary[/]")
    table = Table(title="Detected Fragments")
    table.add_column("ID", justify="center")
    table.add_column("Hill Formula")
    table.add_column("Atom Count", justify="right")

    unique_labels = np.unique(labels)
    for i, lab in enumerate(unique_labels):
        indices = np.where(labels == lab)[0]
        mol = atoms[indices]
        table.add_row(str(i + 1), mol.get_chemical_formula(mode="hill"), str(len(mol)))
    console.print(table)


@click.group()
def main():
    """Fragment detection suite for physical chemistry simulations."""
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--multiplier", default=DEFAULT_BOND_MULTIPLIER, type=float)
@click.option(
    "--radius-type",
    type=click.Choice(["natural", "covalent"]),
    default="natural",
    help="Choose 'natural' for Cordero radii or 'covalent' for standard ASE radii.",
)
@click.option(
    "--min-dist", default=0.0, type=float, help="Merge threshold in Angstroms."
)
@click.option("--visualize", is_flag=True)
def geometric(filename, multiplier, radius_type, min_dist, visualize):
    """Executes geometric fragment detection."""
    atoms = read(filename)

    # Pass the new radius_type argument
    n, labels = find_fragments_geometric(atoms, multiplier, radius_type=radius_type)

    if min_dist > 0:
        n, labels = merge_fragments_by_distance(atoms, n, labels, min_dist)
    print_results(Console(), atoms, n, labels)

    if visualize:
        # Pass radius_type to visualization to ensure the drawn bonds match the logic
        visualize_with_pyvista(
            atoms,
            DetectionMethod.GEOMETRIC,
            multiplier,
            radius_type=radius_type,
        )


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--method",
    # why isn't IPEA-xTB and the rest present
    type=click.Choice(["GFN2-xTB", "GFN1-xTB", "IPEA-xTB"]),
    default="GFN2-xTB",
    help="The xTB Hamiltonian level for calculation.",
)
@click.option("--threshold", default=DEFAULT_BOND_ORDER_THRESHOLD, type=float)
@click.option("--charge", default=0, type=int)
@click.option("--multiplicity", default=1, type=int)
@click.option("--min-dist", default=0.0, type=float)
@click.option("--visualize", is_flag=True)
def bond_order(filename, method, threshold, charge, multiplicity, min_dist, visualize):
    """Execute fragment detection using quantum mechanical bond orders."""
    atoms = read(filename)
    n, labels, _, matrix = find_fragments_bond_order(
        atoms, threshold, charge, multiplicity, method=method
    )

    if min_dist > 0:
        n, labels = merge_fragments_by_distance(atoms, n, labels, min_dist)

    print_results(Console(), atoms, n, labels)

    if visualize:
        visualize_with_pyvista(
            atoms,
            DetectionMethod.BOND_ORDER,
            matrix,
            nonbond_cutoff=0.05,
            bond_threshold=threshold,
        )


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--method", type=click.Choice(["geometric", "bond-order"]), default="geometric"
)
@click.option("--pattern", default="*.xyz")
@click.option("--output", default="fragments.csv")
@click.option("--min-dist", default=0.0, type=float)
def batch(directory, method, pattern, output, min_dist):
    """Processes directories and outputs CSV summaries."""
    path = Path(directory)
    files = list(path.glob(pattern))
    results = []

    for f in files:
        atoms = read(f)
        if method == "geometric":
            n, labels = find_fragments_geometric(atoms, DEFAULT_BOND_MULTIPLIER)
        else:
            n, labels, _, _ = find_fragments_bond_order(
                atoms, DEFAULT_BOND_ORDER_THRESHOLD, 0, 1
            )

        if min_dist > 0:
            n, labels = merge_fragments_by_distance(atoms, n, labels, min_dist)

        results.append({"file": f.name, "fragments": n})

    with open(output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file", "fragments"])
        writer.writeheader()
        writer.writerows(results)
    logging.info(f"Batch results saved to {output}")


if __name__ == "__main__":
    main()
