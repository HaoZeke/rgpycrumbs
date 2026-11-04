#!/usr/bin/env python3

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
#   "click",
#   "numpy",
#   "ovito",
#   "rich",
#   "chemparseplot>=1.13.0,<2",
#   "readcon>=0.17.1",
#   "rgpycrumbs>=1.10.2",
# ]
# ///
"""Identifies non-crystalline atoms using polyhedral template matching.

.. versionadded:: 0.0.2

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
from pathlib import Path
from typing import Final

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport for Python 3.10."""


# Third-Party
import ase.io as aseio
import click
import numpy as np
from ase.geometry import find_mic
from ase.io.formats import filetype, parse_filename
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


def _load_ovito():
    """Import the Open Visualization Tool and the structure type map.

    Help and import succeed without the package. Analysis imports it.
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import (  # noqa: F401
        CentroSymmetryModifier,
        DeleteSelectedModifier,
        PolyhedralTemplateMatchingModifier,
        SelectTypeModifier,
    )
    from ovito.pipeline import Pipeline, StaticSource

    kind = PolyhedralTemplateMatchingModifier.Type
    return {
        "ase_to_ovito": ase_to_ovito,
        "CentroSymmetryModifier": CentroSymmetryModifier,
        "PolyhedralTemplateMatchingModifier": PolyhedralTemplateMatchingModifier,
        "SelectTypeModifier": SelectTypeModifier,
        "Pipeline": Pipeline,
        "StaticSource": StaticSource,
        "STRUCTURE_TYPE_MAP": {
            CrystalStructure.FCC: kind.FCC,
            CrystalStructure.HCP: kind.HCP,
            CrystalStructure.BCC: kind.BCC,
            CrystalStructure.ICO: kind.ICO,
            CrystalStructure.OTHER: kind.OTHER,
        },
    }


try:
    globals().update(_load_ovito())
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if missing != "ovito" and not missing.startswith("ovito."):
        raise
    ase_to_ovito = None
    CentroSymmetryModifier = None
    PolyhedralTemplateMatchingModifier = None
    SelectTypeModifier = None
    Pipeline = None
    StaticSource = None
    STRUCTURE_TYPE_MAP = None

# Centrosymmetry window of the atoms the vacancy filter drops, in units of a**2
# for the FCC lattice constant a. The parameter of Kelchner, Plimpton and
# Hamilton, Phys. Rev. B 58, 11085 (1998), sums |R_i + R_(i+6)|**2 over the six
# pairs of opposite nearest neighbors. It is 0 in the perfect lattice. Next to
# a vacancy one atom of a pair is missing and the other sits at a/sqrt(2), so
# the parameter is a**2/2. Neighbors of an unrelaxed octahedral interstitial
# come out at a**2/4, so the window reaches a**2/8, half that gap, to either
# side of a**2/2.
CSP_VACANCY_MIN_A2: Final = 3 / 8
CSP_VACANCY_MAX_A2: Final = 5 / 8


def _vacancy_neighbors(
    data, candidates: np.ndarray, csp_min_a2: float, csp_max_a2: float
) -> np.ndarray:
    """The *candidates* whose centrosymmetry, in units of a**2, is in the window."""
    structure = np.asarray(data.particles["Structure Type"])
    fcc = structure == PolyhedralTemplateMatchingModifier.Type.FCC
    if not fcc.any():
        return candidates[:0]
    # PTM gives an FCC atom its nearest-neighbor distance, a/sqrt(2).
    distance = np.asarray(data.particles["Interatomic Distance"])
    a2 = 2 * np.median(distance[fcc]) ** 2
    csp = np.asarray(data.particles["Centrosymmetry"])[candidates] / a2
    return candidates[(csp >= csp_min_a2) & (csp <= csp_max_a2)]


def _sphere_indices(atoms, defect: np.ndarray, radius: float) -> np.ndarray:
    """The atoms closer than *radius* to the center of the *defect* atoms.

    The center is the mean of the defect atoms, each taken at its image
    nearest to the first of them. Each distance goes to the nearest image of
    an atom. Both take images along the periodic axes of the cell alone.
    """
    positions = atoms.get_positions()
    first = positions[defect[0]]
    offsets, _ = find_mic(positions[defect] - first, atoms.cell, atoms.pbc)
    center = first + offsets.mean(axis=0)
    _, distances = find_mic(positions - center, atoms.cell, atoms.pbc)
    return np.flatnonzero(distances < radius)


def find_mismatch_indices(
    filename: str,
    target_structure: CrystalStructure,
    remove_fcc_vacancy: bool = False,
    view_selection: bool = False,
    selection_radius: float = 5,
    csp_vacancy_min: float = CSP_VACANCY_MIN_A2,
    csp_vacancy_max: float = CSP_VACANCY_MAX_A2,
    selection_image: str | Path = "selection.png",
) -> np.ndarray:
    """
    Analyzes a structure file with PTM, identifies the interstitial defect,
    and returns the indices of the interstitial atoms plus all atoms
    within a sphere of radius `selection_radius`.

    The sphere is centered on the mean of the defect atoms. Along each
    periodic axis of the cell, the mean and the distances take the nearest
    periodic image of each atom.
    The structure analysis and the sphere take the periodic axes that the
    file declares. readcon reads a ``.con`` file with the flags of its
    metadata line, and with all three axes periodic when that line has
    none. The eon reader of ase, which takes compressed ``.con`` files,
    drops the flags, so those files are periodic along all three axes.

    With `remove_fcc_vacancy` set, atoms whose centrosymmetry marks them as
    vacancy neighbors are dropped from the defect atoms.
    Those are the mismatched atoms whose centrosymmetry, divided by the
    squared FCC lattice constant, lies from `csp_vacancy_min` to
    `csp_vacancy_max`. A vacancy neighbor has 1/2. Atoms in a stacking fault
    and around a relaxed octahedral interstitial sit near 1/2 too, so they are
    dropped as well. Without FCC atoms nothing is dropped.
    Otherwise every atom that does not match `target_structure` is a defect atom.
    With `view_selection` set, the atoms of the sphere are rendered to the PNG
    file `selection_image`.
    The render then removes its pipeline from the OVITO scene, so the scene
    keeps only the pipelines that it had before the call.
    With no defect atom, the result is empty and nothing is rendered.

    .. versionchanged:: 1.14.0
       Without `remove_fcc_vacancy`, the mismatched atoms are kept instead of
       all being dropped.
       With no defect atom, the empty result comes back before any averaging.
       The vacancy window is in units of the squared lattice constant, set by
       `csp_vacancy_min` and `csp_vacancy_max`, and the parameter is taken
       over all atoms. It used to be 70 to 95 square angstroms of a parameter
       taken over the mismatched atoms alone. The neighbors of a lone vacancy
       have 11/2 there, so that window caught them only for lattice constants
       from 3.57 to 4.16 angstroms.
       `view_selection` defaults to False, and `selection_image` names the
       file. By default, every call used to render ``selection.png`` in the
       working directory.
       Each render used to leave its pipeline in the OVITO scene, so every
       later render in the process drew those atoms as well.
       The center and the distances of the sphere used to ignore the
       periodic boundaries. So a defect on a face, an edge or a corner of
       the cell returned other atoms than the same defect inside it.
       The atoms used to be periodic along all three axes after the read,
       whatever the file declared. A file without a cell then failed in
       the structure analysis.
    """
    try:
        log.info(f"Reading structure from '{filename}'...")
        path = Path(filename)
        if path.suffix.lower() in {".con", ".convel"}:
            from chemparseplot.parse.eon.con_io import read_first_atoms

            atoms = read_first_atoms(path)
        else:
            atoms = aseio.read(filename)
            # The eon reader of ase, which takes compressed .con files, drops
            # the periodic flags. All three are the default of the format.
            if filetype(parse_filename(str(filename))[0]) == "eon":
                atoms.set_pbc(True)
    except FileNotFoundError:
        log.critical(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    except Exception as e:
        log.critical(f"Failed to read or parse file '{filename}'. Error: {e}")
        sys.exit(1)

    structure_map = STRUCTURE_TYPE_MAP
    to_ovito = ase_to_ovito
    pipeline_cls = Pipeline
    source_cls = StaticSource
    select_cls = SelectTypeModifier
    ptm_cls = PolyhedralTemplateMatchingModifier
    csp_cls = CentroSymmetryModifier
    if (
        structure_map is None
        or to_ovito is None
        or pipeline_cls is None
        or source_cls is None
        or select_cls is None
        or ptm_cls is None
        or csp_cls is None
    ):
        raise ImportError(
            "ovito is required to analyze a structure with polyhedral template matching"
        )

    # Set up the OVITO pipeline
    # NOTE(rg): Can't call delete anywhere else because it'll lose the index information
    pipeline = pipeline_cls(source=source_cls(data=to_ovito(atoms)))

    ptm = ptm_cls(output_interatomic_distance=True)
    pipeline.modifiers.append(ptm)

    # Select atoms that DO match the target structure
    ovito_type = structure_map[target_structure]
    select_modifier = select_cls(
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
        # The default is the conventional parameter over 12 neighbors, as for FCC.
        pipeline.modifiers.append(csp_cls())
        data = pipeline.compute()
        vacancy = _vacancy_neighbors(
            data, mismatch_indices, csp_vacancy_min, csp_vacancy_max
        )
        interstitial = np.setdiff1d(mismatch_indices, vacancy)
    else:
        interstitial = mismatch_indices
    log.info(
        f"Found {len(mismatch_indices)} non-{target_structure.value}, returning {len(interstitial)} frenkel pair atoms."
    )
    if interstitial.size == 0:
        return interstitial
    sphere_indices = _sphere_indices(atoms, interstitial, selection_radius)
    final_active_indices = np.unique(np.hstack([sphere_indices, interstitial]))

    if view_selection:
        # Just to see what's being selected..
        pviz = pipeline_cls(source=source_cls(data=to_ovito(atoms[sphere_indices])))
        pviz.add_to_scene()
        try:
            from ovito.vis import Viewport

            vp = Viewport()
            vp.type = Viewport.Type.Ortho
            vp.zoom_all()
            vp.render_image(
                size=(800, 600),
                filename=str(selection_image),
                background=(0, 0, 0),
                frame=0,
            )
        finally:
            pviz.remove_from_scene()
    return final_active_indices


# 4. MAIN SCRIPT LOGIC (with Click for CLI)
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    is_flag=True,
    default=False,
    help="Disable FCC vacancy processing.",
)
@click.option(
    "--csp-vacancy-min",
    type=float,
    default=CSP_VACANCY_MIN_A2,
    show_default=True,
    help="Lowest centrosymmetry of a vacancy neighbor, in units of a**2.",
)
@click.option(
    "--csp-vacancy-max",
    type=float,
    default=CSP_VACANCY_MAX_A2,
    show_default=True,
    help="Highest centrosymmetry of a vacancy neighbor, in units of a**2.",
)
@click.option(
    "--selection-image",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Render the atoms around the defect to this PNG file.",
)
def main(
    filename: str,
    structure: CrystalStructure,
    verbose: bool,
    no_fcc_vacancy: bool,
    csp_vacancy_min: float,
    csp_vacancy_max: float,
    selection_image: Path | None,
):
    """
    Analyzes FILENAME to find all atoms that are NOT the specified
    crystal structure type and prints their 0-based indices as a
    comma-separated list, suitable for use in other programs.

    The vacancy filter drops atoms whose centrosymmetry, divided by the
    squared FCC lattice constant a**2, lies from --csp-vacancy-min to
    --csp-vacancy-max. A vacancy neighbor has 0.5.
    """
    if verbose:
        log.setLevel(logging.INFO)
    if csp_vacancy_min > csp_vacancy_max:
        raise click.BadParameter(
            "must not exceed --csp-vacancy-max", param_hint="--csp-vacancy-min"
        )

    indices = find_mismatch_indices(
        filename,
        structure,
        remove_fcc_vacancy=not no_fcc_vacancy,
        view_selection=selection_image is not None,
        csp_vacancy_min=csp_vacancy_min,
        csp_vacancy_max=csp_vacancy_max,
        selection_image=selection_image or "selection.png",
    )

    # Final, clean output is printed to stdout.
    # All logs, errors, and status messages go to stderr.
    print(", ".join(map(str, indices)))


# 5. SCRIPT ENTRY POINT
if __name__ == "__main__":
    main()
