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
    ALL = "all"  # Align every frame to the reactant
    ENDPOINTS = "endpoints"  # Align only reactant and product to each other


def align_path(frames, mode: AlignMode, use_ira=False, kmax=1.8):
    """Applies the selected alignment strategy to the image sequence."""
    if mode == AlignMode.NONE or len(frames) < 2:
        return frames

    ref = frames[0]

    if mode == AlignMode.ALL:
        logging.info("Aligning [bold]all[/bold] images to reactant.")
        return [ref.copy()] + [
            align_structure_robust(ref, f.copy(), IRAConfig(use_ira, kmax)).atoms
            for f in frames[1:]
        ]

    if mode == AlignMode.ENDPOINTS:
        logging.info("Aligning [bold]endpoints[/bold] (reactant and product) only.")
        # Only the product (last frame) undergoes alignment relative to the reactant
        aligned_product = align_structure_robust(
            ref, frames[-1].copy(), IRAConfig(use_ira, kmax)
        ).atoms
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
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="Directory to save the output files.",
)
@click.option(
    "--images-per-path",
    type=int,
    required=True,
    help="Number of images in a single NEB path.",
)
@click.option(
    "--path-index",
    type=int,
    default=-1,
    show_default=True,
    help="Index of the NEB path to extract. Use -1 for the last path.",
)
@click.option(
    "--center/--no-center",
    default=False,
    help="Center the atoms in each frame around the origin.",
)
@click.option(
    "--box-diagonal",
    nargs=3,
    type=(float, float, float),
    default=(25.0, 25.0, 25.0),
    show_default=True,
    help="Override the unit cell dimensions during processing.",
)
@click.option(
    "--align-type",
    type=click.Choice([m.value for m in AlignMode]),
    default=AlignMode.NONE.value,
    help="Alignment strategy: 'all' (every image), 'endpoints' (reactant/product), or 'none'.",
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
    help="Name of the file containing the list of output .con paths.",
)
def con_splitter(
    neb_trajectory_file: Path,
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
    Splits multi-step NEB trajectories into individual .con files.

    This utility extracts optimization steps and applies physical
    chemistry refinements to the structural coordinates.
    """
    if output_dir is None:
        output_dir = Path(neb_trajectory_file.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    CONSOLE.rule(f"[bold green]Processing {neb_trajectory_file.name}[/bold green]")

    try:
        all_frames = aseread(neb_trajectory_file, index=":")
    except Exception as e:
        logging.critical(f"Failed to read trajectory: {e}")
        sys.exit(1)

    num_paths = len(all_frames) // images_per_path
    target_idx = num_paths - 1 if path_index == -1 else path_index

    start, end = target_idx * images_per_path, (target_idx + 1) * images_per_path
    frames = all_frames[start:end]
    logging.info(f"Extracted [cyan]Path {target_idx}[/cyan] with {len(frames)} images.")

    for atoms in frames:
        if center:
            logging.info("Centering structures...")
            atoms.center()
        if box_diagonal:
            logging.info("Overriding box...")
            atoms.set_cell(box_diagonal)

    mode = AlignMode(align_type)
    if mode != AlignMode.NONE:
        frames = align_path(frames, mode, use_ira=use_ira, kmax=ira_kmax)

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
