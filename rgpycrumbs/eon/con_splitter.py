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
import sys

import click
from ase.io import read as aseread
from ase.io import write as asewrite
from rich.console import Console
from rich.logging import RichHandler

# Using a global console object is a common pattern with Rich
CONSOLE = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=False)],
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
    "--path-list-filename",
    default="ipath.dat",
    help="Name of the file that will list the paths to the generated .con files.",
)
def con_splitter(
    neb_trajectory_file: Path,
    output_dir: Path | None,
    path_list_filename: str,
):
    """
    Splits a multi-image .con trajectory file into individual .con files.

    This script reads a file like 'neb_path_XXX.con', which contains multiple
    atomic configurations (frames), and writes each frame into a separate file
    (e.g., ipath_000.con, ipath_001.con, ...).

    It also generates a text file (default: 'ipath.dat') that lists the
    absolute paths of all created .con files, which can be used as input
    for an NEB calculation.

    NEB_TRAJECTORY_FILE: Path to the input multi-image .con file.
    """
    # --- 1. Setup and Validation ---
    if output_dir is None:
        # If no output directory is specified, create one based on the input filename
        output_dir = Path(neb_trajectory_file.stem)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.critical(
            f"Error creating output directory [red]{output_dir}[/red]: {e}"
        )
        sys.exit(1)

    CONSOLE.rule(
        f"[bold green]Splitting {neb_trajectory_file.name} into individual frames[/bold green]"
    )
    logging.info(f"Output directory: [cyan]{output_dir.resolve()}[/cyan]")

    # --- 2. Read all frames from the trajectory ---
    try:
        logging.info(
            f"Reading all frames from [yellow]{neb_trajectory_file}[/yellow]..."
        )
        # The index=':' syntax tells ASE to read all images from the file
        all_frames = aseread(neb_trajectory_file, index=":")
        if not all_frames:
            logging.warning("No frames found in the input file. Exiting.")
            sys.exit(0)
        logging.info(f"Found {len(all_frames)} frames to process.")
    except Exception as e:
        logging.critical(f"Failed to read trajectory file: {e}")
        sys.exit(1)

    # --- 3. Write individual frames and create the path list ---
    path_list_filepath = output_dir / path_list_filename
    created_paths = []

    try:
        with open(path_list_filepath, "w") as path_file:
            CONSOLE.log(
                f"Writing individual .con files and creating [magenta]{path_list_filename}[/magenta]..."
            )
            for i, atoms_frame in enumerate(all_frames):
                # Define the output filename for the current frame
                output_con_filename = f"ipath_{i:03d}.con"
                output_con_filepath = output_dir / output_con_filename

                # Write the single frame to its own .con file
                asewrite(output_con_filepath, atoms_frame)
                CONSOLE.log(f"  - Created [green]{output_con_filepath.name}[/green]")

                # Store the absolute path for the list file
                absolute_path = output_con_filepath.resolve()
                created_paths.append(str(absolute_path))

            # Write all the collected absolute paths to the list file
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
