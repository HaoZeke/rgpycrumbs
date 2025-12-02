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
from pathlib import Path

import click
from ase.io import read as aseread
from ase.io import write as asewrite
from rich.console import Console
from rich.logging import RichHandler

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


@click.command()
@click.argument(
    "neb_trajectory_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Directory to save the output files. "
        "Defaults to a new directory named after the input file (e.g., 'neb_path_001/')."
    ),
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
    default=0,
    show_default=True,
    help=(
        "Index of the NEB path to extract (0-based). "
        "Use -1 to extract the *last* available path."
    ),
)
@click.option(
    "--path-list-filename",
    default="ipath.dat",
    help="Name of the file that will list the paths to the generated .con files.",
)
def con_splitter(
    neb_trajectory_file: Path,
    output_dir: Path | None,
    images_per_path: int,
    path_index: int,
    path_list_filename: str,
):
    """
    Splits a multi-step NEB trajectory file (.traj, .con, etc.) into
    individual .con files for a *single* specified path.

    This script reads a trajectory file, which may contain multiple NEB
    optimization steps (paths), and extracts only the frames corresponding
    to a single specified path.

    It writes each frame of that path into a separate .con file
    (e.g., ipath_000.con, ipath_001.con, ...).

    It also generates a text file (default: 'ipath.dat') that lists the
    absolute paths of all created .con files.

    NEB_TRAJECTORY_FILE: Path to the input multi-image/multi-path trajectory.
    """
    # --- 1. Setup and Validation ---
    if output_dir is None:
        output_dir = Path(neb_trajectory_file.stem)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.critical(f"Error creating output directory [red]{output_dir}[/red]: {e}")
        sys.exit(1)

    if images_per_path <= 0:
        logging.critical("--images-per-path must be a positive number.")
        sys.exit(1)

    CONSOLE.rule(
        f"[bold green]Splitting {neb_trajectory_file.name} into .con files[/bold green]"
    )
    logging.info(f"Output directory: [cyan]{output_dir.resolve()}[/cyan]")
    logging.info(f"Images per path: [magenta]{images_per_path}[/magenta]")

    # --- 2. Read all frames using ASE ---
    try:
        logging.info(
            f"Reading all frames from [yellow]{neb_trajectory_file}[/yellow] "
            f"using [bold]ASE[/bold]..."
        )
        all_frames = aseread(neb_trajectory_file, index=":")

        total_frames = len(all_frames)
        if not all_frames:
            logging.warning("No frames found in the input file. Exiting.")
            sys.exit(0)

        logging.info(f"Found {total_frames} total frames in file.")

    except Exception as e:
        logging.critical(f"Failed to read trajectory file: {e}")
        sys.exit(1)

    # --- 3. Validate paths and select frames ---
    if total_frames < images_per_path:
        logging.critical(
            f"Error: Trajectory contains {total_frames} frames, "
            f"but {images_per_path} images per path were requested."
        )
        sys.exit(1)

    num_paths = total_frames // images_per_path
    if total_frames % images_per_path != 0:
        logging.warning(
            f"Total frames ({total_frames}) is not a clean multiple of "
            f"images per path ({images_per_path}). "
            f"Trajectory may be from an incomplete calculation. "
            f"Found {num_paths} complete path(s)."
        )
    else:
        logging.info(
            f"Found {num_paths} complete path(s) ({total_frames} / {images_per_path})."
        )

    target_path_index: int
    if path_index == -1:
        target_path_index = num_paths - 1
        logging.info(
            f"Using sentinel -1: extracting last complete path (index {target_path_index})."
        )
    else:
        target_path_index = path_index

    if not (0 <= target_path_index < num_paths):
        logging.critical(
            f"Error: Path index {target_path_index} is out of bounds. "
            f"Valid indices for this file are 0 to {num_paths - 1}."
        )
        sys.exit(1)

    start_frame = target_path_index * images_per_path
    end_frame = start_frame + images_per_path
    frames_to_process = all_frames[start_frame:end_frame]

    logging.info(
        f"Extracting path {target_path_index} "
        f"(frames [bold]{start_frame}..{end_frame - 1}[/bold]) "
        f"-> {len(frames_to_process)} images."
    )

    # --- 4. Write individual .con files and create the path list ---
    path_list_filepath = output_dir / path_list_filename
    created_paths = []

    try:
        logging.info(
            f"Writing {len(frames_to_process)} individual .con files "
            f"and creating [magenta]{path_list_filename}[/magenta]..."
        )
        with open(path_list_filepath, "w") as path_file:
            for i, atoms_frame in enumerate(frames_to_process):
                output_con_filename = f"ipath_{i:03d}.con"
                output_con_filepath = output_dir / output_con_filename

                asewrite(output_con_filepath, atoms_frame)

                logging.info(f"  - Created [green]{output_con_filepath.name}[/green]")

                absolute_path = output_con_filepath.resolve()
                created_paths.append(str(absolute_path))

            path_file.write("\n".join(created_paths) + "\n")

        logging.info(
            f"Successfully wrote {len(created_paths)} paths to [magenta]{path_list_filepath.resolve()}[/magenta]"
        )

    except Exception as e:
        logging.critical(f"An error occurred during file writing: {e}")
        sys.exit(1)

    CONSOLE.rule("[bold green]Processing Complete[/bold green]")


if __name__ == "__main__":
    con_splitter()
