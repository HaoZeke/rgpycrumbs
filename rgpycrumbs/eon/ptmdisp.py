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
from ovito.modifiers import (
    PolyhedralTemplateMatchingModifier,
    SelectTypeModifier,
    CentroSymmetryModifier,
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    InvertSelectionModifier,
)
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


def find_mismatch_indices(
    filename: str,
    target_structure: CrystalStructure,
    remove_fcc_vacancy: bool = False,
    view_selection: bool = True,
    cylinder_radius: float = 5,
) -> np.ndarray:
    """
    Analyzes a structure file with PTM and returns indices of atoms that
    do NOT match the target crystal structure.
    """
    try:
        log.info(f"Reading structure from '{filename}'...")
        atoms = aseio.read(filename)
        # XXX(rg): con readers in ase somehow lose this information, seems like an ase bug
        atoms.set_pbc([True] * 3)
    except FileNotFoundError:
        log.critical(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    except Exception as e:
        log.critical(f"Failed to read or parse file '{filename}'. Error: {e}")
        sys.exit(1)

    # Set up the OVITO pipeline
    # NOTE(rg): Can't call delete anywhere else because it'll lose the index information
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))

    ptm = PolyhedralTemplateMatchingModifier()
    pipeline.modifiers.append(ptm)

    # Select atoms that DO match the target structure
    ovito_type = STRUCTURE_TYPE_MAP[target_structure]
    select_modifier = SelectTypeModifier(
        operate_on="particles",
        property="Structure Type",
        types={ovito_type},
    )
    pipeline.modifiers.append(select_modifier)
    non_fcc = pipeline.compute()
    # The 'selection' array is 1 for selected atoms and 0 for others.
    # Find indices where the selection is 0 (i.e., the non-matching atoms).
    mismatch_indices = np.where(non_fcc.particles.selection.array == 0)[0]
    log.info(f"Running PTM analysis to find non-{target_structure.value} atoms...")
    if remove_fcc_vacancy:
        pipeline.modifiers.append(InvertSelectionModifier())
        csym = CentroSymmetryModifier(
            only_selected=True,
        )  # Default is conventional with 12 for FCC
        pipeline.modifiers.append(csym)
        pipeline.modifiers.append(
            ExpressionSelectionModifier(
                operate_on="particles",
                expression="Centrosymmetry < 70 || Centrosymmetry > 95",
            )
        )
    data = pipeline.compute()

    # The 'selection' is now the vacancy
    vacancy = np.where(data.particles.selection.array == 0)[0]
    interstitial = np.setdiff1d(mismatch_indices, vacancy)
    positions = data.particles.positions
    center_vacancy = np.mean(positions[vacancy], axis=0)
    center_interstitial = np.mean(positions[interstitial], axis=0)
    cylinder_radius_sq = cylinder_radius**2

    p1 = center_vacancy
    p2 = center_interstitial

    axis_vec = p2 - p1 # Vector points from vacancy TOWARDS interstitial
    axis_len_sq = np.sum(axis_vec**2)

    # Check for zero-length axis to avoid division by zero
    if axis_len_sq == 0:
        # Handle case where centers are identical
        between_indices = np.array([], dtype=int)
    else:
        # 4. Perform the cylinder selection math
        vec_from_p1 = positions - p1
        proj_len_sq = np.einsum("ij,j->i", vec_from_p1, axis_vec)**2

        # is_within_length: projection squared must be between 0 and axis_len_sq^2
        # We can't do this check on the squared projection easily. Let's use the non-squared version.
        proj_len = np.einsum("ij,j->i", vec_from_p1, axis_vec)
        is_within_length = (proj_len >= 0) & (proj_len <= axis_len_sq)

        # perp_dist_sq: |v|^2 - (projection)^2 / |axis|^2
        perp_dist_sq = np.sum(vec_from_p1**2, axis=1) - proj_len**2 / axis_len_sq
        is_within_radius = perp_dist_sq < cylinder_radius_sq

        # Final selection mask
        selection_mask = is_within_length & is_within_radius
        between_indices_all = np.where(selection_mask)[0]

        # 5. Exclude atoms that are part of the original defect clusters
        between_indices = np.setdiff1d(between_indices_all, mismatch_indices)

    breakpoint()

    log.info(
        f"Found {len(mismatch_indices)} non-{target_structure.value}, returning {len(interstitial)} frenkel pair atoms."
    )
    if view_selection:
        # Just to see what's being selected..
        all_selections = np.unique(np.hstack([between_indices, vacancy, interstitial]))
        pviz = Pipeline(
            source=StaticSource(data=ase_to_ovito(atoms[all_selections]))
        )
        pviz.add_to_scene()
        from ovito.vis import Viewport

        vp = Viewport()
        vp.type = Viewport.Type.Ortho
        vp.zoom_all()
        vp.render_image(
            size=(800, 600), filename="interstitial.png", background=(0, 0, 0), frame=0
        )
    return interstitial


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
@click.option(
    "--no-fcc-vacancy",
    "no_fcc_vacancy",
    default=False,
    help="Enable verbose informational output to stderr.",
)
def main(
    filename: str, structure: CrystalStructure, verbose: bool, no_fcc_vacancy: bool
):
    """
    Analyzes FILENAME to find all atoms that are NOT the specified
    crystal structure type and prints their 0-based indices as a
    comma-separated list, suitable for use in other programs.
    """
    if verbose:
        log.setLevel(logging.INFO)

    indices = find_mismatch_indices(filename, structure, no_fcc_vacancy)

    # Final, clean output is printed to stdout.
    # All logs, errors, and status messages go to stderr.
    print(", ".join(map(str, indices)))


# 5. SCRIPT ENTRY POINT
if __name__ == "__main__":
    main()
