#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "ase>=3.22",
#   "numpy",
#   "rgpycrumbs",
# ]
# ///
"""Match atoms in structure file to target coordinates.

CLI for matching target coordinates to closest atoms in a structure file.
Useful for identifying atom IDs in eOn calculations.

.. versionadded:: 1.7.0
    Extracted from rgpycrumbs._aux to standalone CLI script.
"""

import logging
from pathlib import Path

import click
import numpy as np
from ase.io import read

from rgpycrumbs._aux import _import_from_parent_env

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# --- Configuration ---
POSCON_FILENAME = "pos.con"
_EXPECTED_COORD_COLS = 3

# --- Example target coordinates (can be overridden via CLI) ---
DEFAULT_TARGET_COORDS = """
19.7267 23.4973 21.4053
21.6919 21.7746 21.7746
19.7274 21.4053 23.4968
21.6918 21.7752 25.2201
21.6915 25.2205 21.7761
18.0999 23.4978 23.4978
19.7265 25.5905 23.4985
21.6915 25.2206 25.2206
22.7758 23.4966 23.4983
20.6077 23.4982 23.4977
19.7268 23.4982 25.5902
23.6566 21.4042 23.4962
23.6566 23.4979 21.4058
23.6561 25.5897 23.4984
23.6566 23.4984 25.5906
25.2834 23.4976 23.4978
"""


def parse_target_coords(text_block: str) -> np.ndarray:
    """Parse multiline string of target coordinates.

    Parameters
    ----------
    text_block
        Multiline string with space-separated x y z coordinates

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with target coordinates
    """
    coords = []
    lines = text_block.strip().split("\n")
    for i, line in enumerate(lines):
        try:
            parts = line.strip().split()
            if len(parts) == _EXPECTED_COORD_COLS:
                coords.append([float(p) for p in parts])
            elif parts:
                log.warning(
                    "Skipping target coordinate line %d due to incorrect format: %s",
                    i + 1,
                    line.strip(),
                )
        except ValueError:
            log.warning(
                "Skipping target coordinate line %d due to non-numeric value: %s",
                i + 1,
                line.strip(),
            )
    return np.array(coords)


def match_atoms(
    structure_file: Path,
    target_coords: np.ndarray,
) -> list[dict]:
    """Match target coordinates to closest atoms in structure.

    Parameters
    ----------
    structure_file
        Path to structure file (any ASE-readable format)
    target_coords
        Array of target coordinates to match

    Returns
    -------
    list[dict]
        List of match results with atom info and distances
    """
    try:
        atoms = read(structure_file)
        log.info("Successfully read %d atoms from %s.", len(atoms), structure_file)
    except FileNotFoundError:
        log.error("Error: File not found at %s", structure_file)
        return []
    except Exception as e:
        log.error("Error reading %s with ASE: %s", structure_file, e)
        return []

    if target_coords.size == 0:
        log.warning("No target coordinates provided.")
        return []

    atom_coords = atoms.get_positions()
    atom_symbols = atoms.get_chemical_symbols()

    results = []
    for i, target_pos in enumerate(target_coords):
        distances = np.linalg.norm(atom_coords - target_pos, axis=1)
        closest_idx = int(np.argmin(distances))
        min_dist = float(distances[closest_idx])

        results.append(
            {
                "target_index": i + 1,
                "target_pos": target_pos,
                "closest_atom_id": closest_idx,
                "closest_atom_symbol": atom_symbols[closest_idx],
                "closest_atom_pos": atom_coords[closest_idx],
                "distance": min_dist,
            }
        )

    return results


@click.command()
@click.option(
    "--structure",
    "-s",
    "structure_file",
    type=click.Path(exists=True, path_type=Path),
    default=POSCON_FILENAME,
    help=f"Structure file to match against (default: {POSCON_FILENAME})",
)
@click.option(
    "--target-file",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    help="File with target coordinates (one per line: x y z)",
)
@click.option(
    "--target-coords",
    "-c",
    type=str,
    default=DEFAULT_TARGET_COORDS,
    help="Target coordinates as multiline string (if --target-file not provided)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for results (default: stdout)",
)
def main(
    structure_file: Path,
    target_file: Path | None,
    target_coords: str,
    output: Path | None,
):
    """Match target coordinates to closest atoms in a structure.

    Reads a structure file and finds the closest atoms to specified target
    coordinates. Useful for identifying atom IDs in eOn calculations.
    """
    # Parse target coordinates
    if target_file:
        log.info("Reading target coordinates from %s", target_file)
        target_text = target_file.read_text()
    else:
        log.info("Using provided target coordinates")
        target_text = target_coords

    targets = parse_target_coords(target_text)
    log.info("Found %d target coordinates to match.", len(targets))

    # Match atoms
    results = match_atoms(structure_file, targets)

    if not results:
        log.warning("No matches found.")
        return

    # Output results
    output_lines = []
    output_lines.append("--- Results ---")
    for result in results:
        tp = result["target_pos"]
        output_lines.append(
            f"Target #{result['target_index']} ({tp[0]:.4f}, {tp[1]:.4f}, {tp[2]:.4f})"
        )
        output_lines.append(
            f"  -> Closest Atom ID: {result['closest_atom_id']} "
            f"(Symbol: {result['closest_atom_symbol']})"
        )
        cp = result["closest_atom_pos"]
        output_lines.append(f"     Position: ({cp[0]:.4f}, {cp[1]:.4f}, {cp[2]:.4f})")
        output_lines.append(f"     Distance: {result['distance']:.6f}")

    output_text = "\n".join(output_lines) + "\n"

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_text)
        log.info("Results written to %s", output)
    else:
        click.echo(output_text)


if __name__ == "__main__":
    main()
