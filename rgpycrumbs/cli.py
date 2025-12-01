import os
import site
import subprocess
import sys
from pathlib import Path
import click

# The directory where cli.py is located
PACKAGE_ROOT = Path(__file__).parent.resolve()


def _dispatch_to_script(folder_name: str, script_name: str, script_args: tuple):
    """
    Generic dispatcher:
    1. Looks for {folder_name}/{script_name}.py
    2. Sets up environment (fallback to parent site-packages).
    3. Runs via 'uv run'.
    """
    # Convert script-name to script_name.py (e.g., plt-neb -> plt_neb.py)
    filename = f"{script_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / folder_name / filename

    if not script_path.is_file():
        click.echo(f"Error: Script not found at '{script_path}'", err=True)
        # List available scripts in that folder to be helpful
        available = [
            f.stem
            for f in (PACKAGE_ROOT / folder_name).glob("*.py")
            if not f.name.startswith("_")
        ]
        if available:
            click.echo(
                f"Available scripts in '{folder_name}': {', '.join(available)}",
                err=True,
            )
        sys.exit(1)

    # Build command
    command = ["uv", "run", str(script_path)] + list(script_args)

    # Setup Environment (Parent path fallback)
    env = os.environ.copy()
    parent_paths = os.pathsep.join(
        site.getsitepackages() + [site.getusersitepackages()]
    )
    env["RGPYCRUMBS_PARENT_SITE_PACKAGES"] = parent_paths

    click.echo(f"--> Dispatching to: {' '.join(command)}", err=True)

    try:
        # Crucial: pass env=env so the subprocess sees the site-packages variable
        subprocess.run(command, check=True, env=env)
    except FileNotFoundError:
        click.echo("Error: 'uv' command not found. Is it installed?", err=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def make_command_function(module_name):
    """
    Factory function to create the click command function.
    We need this factory to avoid Python closure issues (variable binding).
    """

    @click.command(
        name=module_name,
        context_settings=dict(ignore_unknown_options=True),
        add_help_option=False,
        help=f"Dispatch to scripts in the '{module_name}' directory.",
    )
    @click.argument("subcommand_name")
    @click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
    def command_wrapper(subcommand_name, script_args):
        _dispatch_to_script(module_name, subcommand_name, script_args)

    return command_wrapper


@click.group()
def cli():
    """A dispatcher that runs self-contained scripts using 'uv'."""
    pass


# --- DYNAMIC REGISTRATION ---

# 1. Get all directories in the package root
# 2. Filter out __pycache__, dot-files, or files
for item in PACKAGE_ROOT.iterdir():
    if item.is_dir() and not item.name.startswith(("_", ".")):
        # 3. Create the function dynamically
        dynamic_cmd = make_command_function(item.name)

        # 4. Register it to the CLI group
        cli.add_command(dynamic_cmd)


if __name__ == "__main__":
    cli()
