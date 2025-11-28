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

    # Pass parent site-packages as an env var for fallback imports
    env = os.environ.copy()
    parent_paths = os.pathsep.join(site.getsitepackages() + [site.getusersitepackages()])
    env["RGPYCRUMBS_PARENT_SITE_PACKAGES"] = parent_paths

    click.echo(f"--> Dispatching to: {' '.join(command)}", err=True)

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        click.echo("Error: 'uv' command not found.", err=True)
        click.echo("Please ensure 'uv' is installed and in your system's PATH.", err=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    cli()
