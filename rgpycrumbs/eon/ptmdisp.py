#!/usr/bin/env python3

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
#   "click",
#   "numpy",
#   "ovito",
#   "rich",
# ]
# ///
"""
Identifies atoms in a structure file that do not match a specified crystal
structure (e.g., FCC) and prints their 0-based indices to standard output.
By default, the script is quiet. Use --verbose for progress messages.
"""

# 1. WARNING SUPPRESSION (before other imports)
import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*Please use atoms.calc.*"
)

# 2. IMPORTS (grouped by type)
# Standard Library
import logging
import sys
from enum import StrEnum

# Third-Party
import ase.io as aseio
import click
import numpy as np
from ovito.io.ase import ase_to_ovito
from ovito.modifiers import PolyhedralTemplateMatchingModifier, SelectTypeModifier
from ovito.pipeline import Pipeline, StaticSource
from rich.logging import RichHandler

# 3. CONSTANTS and ENUMERATIONS
# Set up logging to stderr using Rich.
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger(__name__)


# Use StrEnum for type-safe, readable choices for the crystal structure.
class CrystalStructure(StrEnum):
    OTHER = "Other"
    FCC = "FCC"
    HCP = "HCP"
    BCC = "BCC"
    ICO = "Icosahedral"


# Map the string choice to the actual OVITO library constant.
STRUCTURE_TYPE_MAP = {
    CrystalStructure.FCC: PolyhedralTemplateMatchingModifier.Type.FCC,
    CrystalStructure.HCP: PolyhedralTemplateMatchingModifier.Type.HCP,
    CrystalStructure.BCC: PolyhedralTemplateMatchingModifier.Type.BCC,
    CrystalStructure.ICO: PolyhedralTemplateMatchingModifier.Type.ICO,
    CrystalStructure.OTHER: PolyhedralTemplateMatchingModifier.Type.OTHER,
}
STRUCTURE_PROPERTY_NAME = "Structure Type"


def find_mismatch_indices(
    filename: str, target_structure: CrystalStructure
) -> np.ndarray:
    """
    Analyzes a structure file with PTM and returns indices of atoms that
    do NOT match the target crystal structure.
    """
    try:
        log.info(f"Reading structure from '{filename}'...")
        atoms = aseio.read(filename)
    except FileNotFoundError:
        log.critical(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    except Exception as e:
        log.critical(f"Failed to read or parse file '{filename}'. Error: {e}")
        sys.exit(1)

    # Set up the OVITO pipeline
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))

    ptm = PolyhedralTemplateMatchingModifier()
    pipeline.modifiers.append(ptm)

    # Select atoms that DO match the target structure
    ovito_type = STRUCTURE_TYPE_MAP[target_structure]
    select_modifier = SelectTypeModifier(
        operate_on="particles",
        property=STRUCTURE_PROPERTY_NAME,
        types={ovito_type},
    )
    pipeline.modifiers.append(select_modifier)

    log.info(f"Running PTM analysis to find non-{target_structure.value} atoms...")
    data = pipeline.compute()

    # The 'selection' array is 1 for selected atoms and 0 for others.
    # Find indices where the selection is 0 (i.e., the non-matching atoms).
    mismatch_indices = np.where(data.particles.selection.array == 0)[0]
    log.info(f"Found {len(mismatch_indices)} non-{target_structure.value} atoms.")
    return mismatch_indices


# 4. MAIN SCRIPT LOGIC (with Click for CLI)
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "filename",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "-s",
    "--structure-type",
    "structure",
    type=click.Choice(CrystalStructure),
    default=CrystalStructure.FCC,
    show_default=True,
    help="The crystal structure to identify and exclude.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose informational output to stderr.",
)
def main(filename: str, structure: CrystalStructure, verbose: bool):
    """
    Analyzes FILENAME to find all atoms that are NOT the specified
    crystal structure type and prints their 0-based indices as a
    comma-separated list, suitable for use in other programs.
    """
    if verbose:
        log.setLevel(logging.INFO)

    indices = find_mismatch_indices(filename, structure)

    # Final, clean output is printed to stdout.
    # All logs, errors, and status messages go to stderr.
    print(",".join(map(str, indices)))


# 5. SCRIPT ENTRY POINT
if __name__ == "__main__":
    main()
