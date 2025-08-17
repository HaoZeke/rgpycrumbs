#!/usr/bin/env python3

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase>=3.25",
#   "click>=8.2.1",
#   "rich",
# ]
# ///

import configparser
import logging
import sys
from pathlib import Path

try:
    import click
    from ase.io import read as ase_read
    from rich.logging import RichHandler
except ImportError:
    print(
        "Error: Required libraries (ase, click, rich) are not installed.",
        file=sys.stderr,
    )
    print(
        "Please run this script using a PEP 723-compliant runner like 'uv run <script_name>.py'",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)


def generate_nwchem_input(
    pos_path: Path,
    settings_path: Path,
    socket_address: str,
    *,
    unix_mode: bool,
    mem_in_gb: int,
    output_path: Path,
):
    """
    Generates a complete NWChem input file by combining atom types from a .con
    file with a user-provided settings file.

    Args:
        pos_path: Path to the input geometry file (e.g., pos.con).
        settings_path: Path to the user's NWChem settings file.
        socket_address: The address for the socket (path for UNIX, host:port for TCP).
        unix_mode: Boolean indicating if a UNIX socket should be used.
        mem_in_gb: Memory to allocate for the NWChem calculation in Gigabytes.
        output_path: The path for the generated NWChem input file.
    """
    logging.info(f"Reading atom types from: [cyan]{pos_path}[/cyan]")
    try:
        atoms = ase_read(pos_path)
        if len(atoms) == 0:
            logging.critical(f"Input geometry file '{pos_path}' contains no atoms.")
            sys.exit(1)
    except FileNotFoundError:
        logging.critical(f"Geometry file not found at '{pos_path}'")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Failed to read geometry file with ASE: {e}")
        sys.exit(1)

    if not settings_path.is_file():
        logging.critical(f"NWChem settings file not found at '{settings_path}'")
        sys.exit(1)

    logging.info(f"Generating NWChem input file: [cyan]{output_path}[/cyan]")
    try:
        with open(output_path, "w") as f:
            f.write("start nwchem_socket_job\n")
            f.write('title "NWChem Server for EON"\n\n')

            f.write(f"memory {mem_in_gb} gb\n\n")

            # This geometry block is only a template for memory allocation.  The
            # atom types and count are what matter. Rather than confuse people
            # with possible unit related shenanigans, just dummy positions here.
            f.write("geometry units bohr noautosym nocenter noautoz\n")
            for i, atom in enumerate(atoms):
                f.write(
                    f"  {atom.symbol:<4s} {0.0:16.10f} {0.0:16.10f} {float(i):16.10f}\n"
                )
            f.write("end\n\n")

            f.write(f"include {settings_path.name}\n\n")

            f.write("driver\n")
            if unix_mode:
                f.write(f"  socket unix {socket_address}\n")
            else:
                f.write(f"  socket ipi_client {socket_address}\n")
            f.write("end\n\n")

            f.write("task scf optimize\n\n")

    except OSError as e:
        logging.critical(f"Could not write to output file '{output_path}': {e}")
        sys.exit(1)

    logging.info("[bold green]Success![/bold green] NWChem input file generated.")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--pos-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("pos.con"),
    show_default=True,
    help="Path to the input geometry file (e.g., in EON .con format).",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("config.ini"),
    show_default=True,
    help="Path to the eonclient config.ini file to read settings from.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("nwchem_socket.nwi"),
    show_default=True,
    help="Name of the final NWChem input file to be generated.",
)
def main(pos_file: Path, config: Path, output: Path):
    """Generate an NWChem input file for use with the EON SocketNWChemPot."""
    logging.info(f"Reading settings from [cyan]{config}[/cyan]")
    try:
        ini_parser = configparser.ConfigParser()
        ini_parser.read(config)

        settings_section = "SocketNWChemPot"

        # Read all required settings, providing the same defaults as the C++ code.
        settings_path_str = ini_parser.get(
            settings_section, "nwchem_settings", fallback="nwchem_settings.nwi"
        )
        settings_path = Path(settings_path_str)
        logging.info(f"Using NWChem settings file: [yellow]{settings_path}[/yellow]")

        mem_in_gb = ini_parser.getint(settings_section, "mem_in_gb", fallback=2)
        logging.info(f"Setting memory to: [yellow]{mem_in_gb} GB[/yellow]")

        is_unix_mode = ini_parser.getboolean(
            settings_section, "unix_socket_mode", fallback=False
        )

        if is_unix_mode:
            socket_address = ini_parser.get(
                settings_section, "unix_socket_path", fallback="eon_nwchem"
            )
            logging.info(
                f"Mode: [yellow]UNIX[/yellow], Socket Name: [yellow]{socket_address}[/yellow]"
            )
        else:
            host = ini_parser.get(settings_section, "host", fallback="127.0.0.1")
            port = ini_parser.get(settings_section, "port", fallback="9999")
            socket_address = f"{host}:{port}"
            logging.info(
                f"Mode: [yellow]TCP/IP[/yellow], Address: [yellow]{socket_address}[/yellow]"
            )

    except (configparser.NoSectionError, FileNotFoundError) as e:
        logging.critical(
            f"Could not read settings from '{config}'. Please ensure the file exists and contains a [SocketNWChemPot] section. Error: {e}"
        )
        sys.exit(1)

    generate_nwchem_input(
        pos_path=pos_file,
        settings_path=settings_path,
        socket_address=socket_address,
        unix_mode=is_unix_mode,
        mem_in_gb=mem_in_gb,
        output_path=output,
    )


if __name__ == "__main__":
    main()
