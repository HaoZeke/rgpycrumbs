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
from pathlib import Path

import click
import numpy as np
from ase.atoms import Atoms
from ase.io import read
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rgpycrumbs.geom.fragment_visualization import visualize_with_pyvista
from rgpycrumbs.geom.fragments import (
    DEFAULT_BOND_MULTIPLIER,
    DEFAULT_BOND_ORDER_THRESHOLD,
    DetectionMethod,
    find_fragments_bond_order,
    find_fragments_geometric,
    merge_fragments_by_distance,
)

# --- Setup ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)


def print_results(
    console: Console,
    atoms: Atoms,
    n_components: int,  # noqa: ARG001
    labels: np.ndarray,
) -> None:
    """Displays analysis results in a structured table.

    .. versionadded:: 0.0.6
    """
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
@click.option("--min-dist", default=0.0, type=float, help="Merge threshold in Angstroms.")
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
