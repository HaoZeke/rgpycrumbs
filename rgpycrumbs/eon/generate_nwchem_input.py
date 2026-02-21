#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase>=3.25",
#   "click>=8.2.1",
#   "rich",
#   "pychum",
# ]
# ///

import configparser
import logging
import sys
from pathlib import Path

import click
from pychum import render_nwchem
from rich.logging import RichHandler

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--pos-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("pos.con"),
    show_default=True,
    help="Path to the input geometry file (e.g., in eOn .con format).",
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
    "-o",
    type=click.Path(path_type=Path),
    default=Path("nwchem_socket.nwi"),
    show_default=True,
    help="Name of the final NWChem input file to be generated.",
)
def main(pos_file: Path, config: Path, output: Path):
    """Generate an NWChem input file for use with the eOn SocketNWChemPot."""
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

        rendered_output = render_nwchem(
            pos_file=pos_file,
            settings_path=settings_path,
            socket_address=socket_address,
            unix_mode=is_unix_mode,
            mem_in_gb=mem_in_gb,
        )

        with open(output, "w") as f:
            f.write(rendered_output)

        logging.info("[bold green]Success![/bold green] NWChem input file generated.")

    except Exception as e:
        logging.critical(f"Error generating NWChem input: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
