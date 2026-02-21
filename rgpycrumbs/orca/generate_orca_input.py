#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "rich",
#   "pychum",
# ]
# ///

import logging
import sys
from pathlib import Path

import click
from rich.logging import RichHandler
from pychum import render_orca

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)


@click.command()
@click.argument("toml_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    is_flag=False,
    flag_value="orca.inp",
    help='Output file name (default is "orca.inp").',
)
def main(toml_path, output):
    """
    Render an Orca input file from a TOML configuration using pychum.

    TOML_PATH is the path to the TOML configuration file.
    """
    try:
        rendered_output = render_orca(Path(toml_path))

        if output is not None:
            with open(output, "w") as file:
                file.write(rendered_output)
            click.echo(f"Rendered ORCA input file written to '{output}'")
        else:
            click.echo(rendered_output)
    except Exception as e:
        logging.critical(f"Error rendering ORCA input: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
