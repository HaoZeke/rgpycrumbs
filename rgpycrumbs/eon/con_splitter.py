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
from pathlib import Path

import click
from ase.io import read as aseread
from ase.io import write as asewrite
from rich.console import Console
from rich.logging import RichHandler

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


def align_structures(frames, use_ira=False, kmax=1.8):
    """
    Aligns all frames in the path to the first frame (reactant).

    This procedure removes global rotation and translation. If use_ira equals
    True, the function also accounts for atom index permutations.
    """
    if len(frames) < 2:
        return frames

    ref = frames[0]
    ref_pos = ref.get_positions()
    ref_types = ref.get_atomic_numbers()
    ref_natoms = len(ref)

    aligned_frames = [ref.copy()]

    # Initialize IRA if requested
    ira_instance = None
    if use_ira and ira_mod:
        ira_instance = ira_mod.IRA()
        logging.info(
            "Using [bold magenta]IRA[/bold magenta] for optimal alignment and reordering."
        )

    for i in range(1, len(frames)):
        current = frames[i].copy()

        if ira_instance:
            # Perform IRA match
            r, t, p, _ = ira_instance.match(
                ref_natoms,
                ref_types,
                ref_pos,
                len(current),
                current.get_atomic_numbers(),
                current.get_positions(),
                kmax,
            )
            # Apply rotation, translation, and permutation
            new_pos = (current.get_positions() @ r.T) + t
            current.set_positions(new_pos[p])
            # Reorder atomic numbers to match reference after permutation
            current.set_atomic_numbers(current.get_atomic_numbers()[p])
        else:
            # Standard ASE alignment (Procrustes)
            current.euler_rotate(
                0, 0, 0
            )  # Placeholder for more complex ASE alignment if needed
            # For basic NEB, many users prefer simple RMSD alignment:
            from ase.build import minimize_rotation_and_translation

            minimize_rotation_and_translation(ref, current)

        aligned_frames.append(current)

    return aligned_frames


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
    "--center",
    is_flag=True,
    help="Center the atoms in each frame around the origin.",
)
@click.option(
    "--box-diagonal",
    nargs=3,
    type=(float, float, float),
    default=(25, 25, 25),
    show_default=True,
    help="Override box while centering.",
)
@click.option(
    "--align",
    is_flag=True,
    help="Align all frames to the first image of the path.",
)
@click.option(
    "--use-ira",
    is_flag=True,
    help="Use Iterative Reordering and Alignment (requires ira_mod).",
)
@click.option(
    "--ira-kmax",
    type=float,
    default=1.8,
    help="kmax factor for the IRA algorithm.",
)
@click.option(
    "--path-list-filename",
    default="ipath.dat",
    help="File listing the absolute paths of generated .con files.",
)
def con_splitter(
    neb_trajectory_file: Path,
    output_dir: Path | None,
    images_per_path: int,
    path_index: int,
    center: bool,
    box_diagonal: (float, float, float),
    align: bool,
    use_ira: bool,
    ira_kmax: float,
    path_list_filename: str,
):
    """
    Splits multi-step NEB trajectories into individual .con files.

    This tool extracts a specific optimization step and applies optional
    physical chemistry refinements like centering and structural alignment.
    """
    if output_dir is None:
        output_dir = Path(neb_trajectory_file.stem)

    output_dir.mkdir(parents=True, exist_ok=True)

    CONSOLE.rule(f"[bold green]Processing {neb_trajectory_file.name}[/bold green]")

    all_frames = aseread(neb_trajectory_file, index=":")
    total_frames = len(all_frames)
    num_paths = total_frames // images_per_path

    if path_index == -1:
        path_index = num_paths - 1

    start_idx = path_index * images_per_path
    end_idx = start_idx + images_per_path
    frames_to_process = all_frames[start_idx:end_idx]

    logging.info(
        f"Extracted [cyan]Path {path_index}[/cyan] ({len(frames_to_process)} images)."
    )

    for atoms in frames_to_process:
        if center:
            logging.info("Centering structures...")
            atoms.center()
        if box_diagonal:
            logging.info("Overriding box...")
            atoms.set_cell(box_diagonal)

    if align:
        logging.info("Aligning frames to reactant...")
        frames_to_process = align_structures(
            frames_to_process, use_ira=use_ira, kmax=ira_kmax
        )

    path_list_filepath = output_dir / path_list_filename
    created_paths = []

    with open(path_list_filepath, "w") as path_file:
        for i, atoms in enumerate(frames_to_process):
            out_name = f"ipath_{i:03d}.con"
            out_path = output_dir / out_name
            asewrite(out_path, atoms)

            abs_path = out_path.resolve()
            created_paths.append(str(abs_path))
            logging.info(f"  - Saved [green]{out_name}[/green]")

        path_file.write("\n".join(created_paths) + "\n")

    logging.info(f"Path list saved to [magenta]{path_list_filepath}[/magenta]")
    CONSOLE.rule("[bold green]Complete[/bold green]")


if __name__ == "__main__":
    con_splitter()
