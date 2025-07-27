import sys
import subprocess
from pathlib import Path
import click

# This gets the path to the directory where this cli.py file lives.
# We use it to find the other scripts in the package reliably.
PACKAGE_ROOT = Path(__file__).parent.resolve()


@click.group()
def cli():
    """A dispatcher that runs self-contained scripts using 'uv'."""
    pass


@cli.command(
    # Transparently send over all options
    context_settings=dict(
        ignore_unknown_options=True,
    ),
    # By disabling the default help option, '--help' will be passed
    # through to the target script instead of being handled by the dispatcher.
    add_help_option=False,
)
@click.argument("subcommand_name")
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
def eon(subcommand_name: str, script_args: tuple):
    """
    Dispatches to a script within the 'eon' submodule.

    Example: rgpycrumbs eon plt_neb --start 280
    """
    # Construct the full path to the target script
    script_filename = f"{subcommand_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / "eon" / script_filename

    if not script_path.is_file():
        click.echo(f"Error: Script not found at '{script_path}'", err=True)
        sys.exit(1)

    # Build the full command to be executed by the shell
    command = ["uv", "run", str(script_path)] + list(script_args)

    # TODO(rg): Consider adding verbosity
    click.echo(f"--> Dispatching to: {' '.join(command)}", err=True)

    try:
        # Execute the command. This will stream the output of the target
        # script directly to the user's terminal.
        # `check=True` ensures that if the script fails, this dispatcher will also fail.
        subprocess.run(command, check=True)
    except FileNotFoundError:
        click.echo("Error: 'uv' command not found.", err=True)
        click.echo(
            "Please ensure 'uv' is installed and in your system's PATH.", err=True
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # This catches errors from the script itself (e.g., it exited with status 1)
        # and ensures the dispatcher exits with the same error code.
        sys.exit(e.returncode)


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
    ),
    add_help_option=False,
)
@click.argument("subcommand_name")
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
def prefix(subcommand_name: str, script_args: tuple):
    """
    Dispatches to a script within the 'prefix' submodule.

    Example: rgpycrumbs prefix delete_packages --channel my-channel
    """
    # Construct the full path to the target script
    script_filename = f"{subcommand_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / "prefix" / script_filename

    if not script_path.is_file():
        click.echo(f"Error: Script not found at '{script_path}'", err=True)
        sys.exit(1)

    # Build the full command to be executed by the shell
    command = ["uv", "run", str(script_path)] + list(script_args)

    click.echo(f"--> Dispatching to: {' '.join(command)}", err=True)

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        click.echo("Error: 'uv' command not found.", err=True)
        click.echo(
            "Please ensure 'uv' is installed and in your system's PATH.", err=True
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    cli()
