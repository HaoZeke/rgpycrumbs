import logging
import os
import site
import subprocess
import sys
from pathlib import Path

import click

# Configure logging to output to stderr
logging.basicConfig(level=logging.INFO, format="%(message)s")

# The directory where cli.py is located
PACKAGE_ROOT = Path(__file__).parent.resolve()


def _get_scripts_in_folder(folder_name: str) -> list[str]:
    """Returns a sorted list of script names (without extension) in a folder."""
    folder_path = PACKAGE_ROOT / folder_name
    if not folder_path.is_dir():
        return []
    return sorted(f.stem for f in folder_path.glob("*.py") if not f.name.startswith("_"))


def _dispatch(
    group: str,
    script_name: str,
    script_args: tuple,
    is_dev: bool = False,
    is_verbose: bool = False,
):
    """
    Sets up the environment and runs the target script via 'uv run'.
    """
    # Convert script-name to filename (e.g., plt-neb -> plt_neb.py)
    filename = f"{script_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / group / filename

    if not script_path.is_file():
        click.echo(f"Error: Script not found at '{script_path}'", err=True)
        sys.exit(1)

    if not is_dev:
        command = ["uv", "run", str(script_path), *script_args]
    else:
        command = [sys.executable, str(script_path), *script_args]

    if is_verbose:
        click.echo(f"VERBOSE: Resolved script path -> {script_path}", err=True)
        click.echo(f"VERBOSE: Constructed command -> {' '.join(command)}", err=True)

    # --- SETUP ENVIRONMENT ---
    env = os.environ.copy()

    # Fallback imports
    try:
        site_packages = [*site.getsitepackages(), *site.getusersitepackages()]
        env["RGPYCRUMBS_PARENT_SITE_PACKAGES"] = os.pathsep.join(site_packages)
    except (AttributeError, ImportError):
        pass

    # Add parent dir to PYTHONPATH for internal imports (e.g. rgpycrumbs._aux)
    project_root = str(PACKAGE_ROOT.parent)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_pythonpath}"

    click.echo(f"--> Dispatching to: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, env=env)  # noqa: S603
    except FileNotFoundError:
        click.echo("Error: 'uv' command not found. Is it installed?", err=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def _make_script_command(group_name: str, script_stem: str) -> click.Command:
    """Creates a click command that dispatches to a PEP 723 script."""
    display_name = script_stem.replace("_", "-")

    @click.command(
        name=display_name,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.pass_context
    def cmd(ctx):
        # Retrieve the dev flag safely from the parent context
        is_dev = ctx.obj.get("is_dev", False) if ctx.obj else False
        is_verbose = ctx.obj.get("is_verbose", False) if ctx.obj else False
        _dispatch(
            group_name,
            display_name,
            tuple(ctx.args),
            is_dev=is_dev,
            is_verbose=is_verbose,
        )

    cmd.help = f"Run the {display_name} script."
    return cmd


@click.group()
@click.option(
    "--dev",
    is_flag=True,
    help="Run using sys.executable instead of 'uv run' for local development.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Print script paths and constructed commands before execution.",
)
@click.version_option(package_name="rgpycrumbs")
@click.pass_context
def main(ctx, dev, verbose):
    """A dispatcher that runs self-contained PEP 723 scripts using 'uv'."""
    # Ensure ctx.obj is a dictionary so we can store state in it
    ctx.ensure_object(dict)
    ctx.obj["is_dev"] = dev
    ctx.obj["is_verbose"] = verbose


# --- DYNAMIC DISCOVERY ---
# Scan the package directory for subfolders (groups) and register them
_valid_groups = sorted(
    d.name
    for d in PACKAGE_ROOT.iterdir()
    if d.is_dir() and not d.name.startswith(("_", "."))
)

for _group in _valid_groups:
    _file_stems = _get_scripts_in_folder(_group)
    if not _file_stems:
        continue

    # Create a click group for this category
    _group_cmd = click.Group(name=_group, help=f"Tools in the '{_group}' category.")

    for _stem in _file_stems:
        _group_cmd.add_command(_make_script_command(_group, _stem))

    main.add_command(_group_cmd)


if __name__ == "__main__":
    main()
