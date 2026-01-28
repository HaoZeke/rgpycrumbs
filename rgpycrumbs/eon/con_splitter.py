#!/usr/bin/env python3

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "ase",
#   "rich",
# ]
# ///

import logging
import sys
from enum import Enum
from pathlib import Path

import click
from ase.io import read as aseread
from ase.io import write as asewrite
from rich.console import Console
from rich.logging import RichHandler

from rgpycrumbs.geom.api.alignment import IRAConfig, align_structure_robust

# Optional IRA import logic
try:
    from rgpycrumbs._aux import _import_from_parent_env

    ira_mod = _import_from_parent_env("ira_mod")
except ImportError:
    ira_mod = None

CONSOLE = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=CONSOLE,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            show_level=True,
            show_time=True,
        )
    ],
)


class AlignMode(Enum):
    """Defines structural alignment strategies."""

    NONE = "none"
    ALL = "all"
    ENDPOINTS = "endpoints"


class SplitMode(Enum):
    """Defines trajectory validation strictness."""

    NEB = "neb"  # Strict: must be a clean multiple of images_per_path
    FLEX = "flex"  # Flexible: allows partial paths or simple slicing


def align_path(frames, mode: AlignMode, iraconf: IRAConfig):
    """Applies the selected alignment strategy to the image sequence."""
    if mode == AlignMode.NONE or len(frames) < 2:
        return frames

    ref = frames[0]

    if mode == AlignMode.ALL:
        logging.info("Aligning [bold]all[/bold] images to reactant reference.")
        return [ref.copy()] + [
            align_structure_robust(ref, f.copy(), IRAConfig(use_ira, kmax)).atoms
            for f in frames[1:]
        ]

    if mode == AlignMode.ENDPOINTS:
        logging.info("Aligning [bold]endpoints[/bold] (product to reactant) only.")
        aligned_product = align_structure_robust(ref, frames[-1].copy(), iraconf).atoms
        # Intermediate frames remain unchanged in this specific mode logic,
        # Usually, endpoint alignment implies ensuring the BCs match.
        new_frames = [f.copy() for f in frames]
        new_frames[-1] = aligned_product
        return new_frames

    return frames


@click.command()
@click.argument(
    "neb_trajectory_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--mode",
    type=click.Choice([m.value for m in SplitMode]),
    default=SplitMode.FLEX.value,
    help="Validation mode: 'neb' (strict multiples) or 'normal' (flexible).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="Directory to save output files. Defaults to the input filename stem.",
)
@click.option(
    "--images-per-path",
    type=int,
    required=True,
    help="Number of images in a single NEB path (e.g., 7). [REQUIRED]",
)
@click.option(
    "--path-index",
    type=int,
    default=-1,
    show_default=True,
    help="Index of the NEB path to extract (0-based). Use -1 for the last path.",
)
@click.option(
    "--center/--no-center",
    default=False,
    help="Center the atomic coordinates around the origin.",
)
@click.option(
    "--box-diagonal",
    nargs=3,
    type=(float, float, float),
    default=(25.0, 25.0, 25.0),
    show_default=True,
    help="Override the unit cell dimensions (Å) during processing.",
)
@click.option(
    "--align-type",
    type=click.Choice([m.value for m in AlignMode]),
    default=AlignMode.NONE.value,
    help="Alignment: 'all' (every image), 'endpoints' (reactant/product), or 'none'.",
)
@click.option(
    "--use-ira",
    is_flag=True,
    help="Enable Iterative Reordering and Alignment (requires ira_mod).",
)
@click.option(
    "--ira-kmax",
    type=float,
    default=1.8,
    help="kmax factor for the IRA matching algorithm.",
)
@click.option(
    "--path-list-filename",
    default="ipath.dat",
    help="Name of the file listing the generated .con absolute paths.",
)
def con_splitter(
    neb_trajectory_file: Path,
    mode: str,
    output_dir: Path | None,
    images_per_path: int,
    path_index: int,
    center: bool,
    box_diagonal: tuple[float, float, float],
    align_type: str,
    use_ira: bool,
    ira_kmax: float,
    path_list_filename: str,
):
    """
    Splits a multi-step trajectory file (.traj, .con, etc.) into
    individual .con files for a *single* specified path.

    This script reads a trajectory file, which may contain multiple NEB
    optimization steps (paths), and extracts only the frames corresponding
    to a single specified path.

    It writes each frame of that path into a separate .con file
    (e.g., ipath_000.con, ipath_001.con, ...).

    It also generates a text file (default: 'ipath.dat') that lists the
    absolute paths of all created .con files.

    This utility extracts specific optimization steps and applies physical
    chemistry refinements such as centering, cell overrides, and structural
    alignment (RMSD minimization).
    """
    if output_dir is None:
        output_dir = Path(neb_trajectory_file.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    CONSOLE.rule(f"[bold green]Processing {neb_trajectory_file.name}[/bold green]")

    if images_per_path <= 0:
        logging.critical("--images-per-path must be a positive integer.")
        sys.exit(1)

    try:
        all_frames = aseread(neb_trajectory_file, index=":")
        if not all_frames:
            logging.error("No frames found in input file.")
            sys.exit(1)
    except Exception as e:
        logging.critical(f"Failed to read trajectory: {e}")
        sys.exit(1)

    total_frames = len(all_frames)
    num_paths = total_frames // images_per_path
    remainder = total_frames % images_per_path

    # Validation Logic based on Mode
    if mode == SplitMode.NEB.value and remainder != 0:
        logging.warning(
            f"Trajectory has {total_frames} frames,"
            f" which is not a multiple of {images_per_path}. "
            f"This often indicates an interrupted NEB calculation."
        )

    if total_frames < images_per_path:
        logging.critical(
            f"Total frames ({total_frames})"
            f" is less than images per path ({images_per_path})."
        )
        sys.exit(1)

    target_idx = num_paths - 1 if path_index == -1 else path_index
    if not (0 <= target_idx < num_paths):
        logging.critical(
            f"Path index {target_idx} is out of bounds (0 to {num_paths - 1})."
        )
        sys.exit(1)

    start, end = target_idx * images_per_path, (target_idx + 1) * images_per_path
    frames = all_frames[start:end]
    logging.info(f"Extracted [cyan]Path {target_idx}[/cyan] with {len(frames)} images.")

    if center:
        logging.info("Centering structures...")
    if box_diagonal:
        logging.info("Overriding box...")
    for atoms in frames:
        if center:
            atoms.center()
        if box_diagonal:
            atoms.set_cell(box_diagonal)

    align_strategy = AlignMode(align_type)
    if align_strategy != AlignMode.NONE:
        frames = align_path(
            frames, align_strategy, IRAConfig(enabled=use_ira, kmax=ira_kmax)
        )

    created_paths = []
    for i, atoms in enumerate(frames):
        name = f"ipath_{i:03d}.con"
        dest = output_dir / name
        asewrite(dest, atoms)
        created_paths.append(str(dest.resolve()))
        logging.info(f"  - Saved [green]{name}[/green]")

    with open(output_dir / path_list_filename, "w") as f:
        f.write("\n".join(created_paths) + "\n")

    logging.info(f"Path list saved to [magenta]{path_list_filename}[/magenta]")
    CONSOLE.rule("[bold green]Complete[/bold green]")


if __name__ == "__main__":
    con_splitter()
