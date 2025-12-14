import argparse
import logging
import os
import site
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# The directory where cli.py is located
PACKAGE_ROOT = Path(__file__).parent.resolve()


def _get_scripts_in_folder(folder_name: str) -> list[str]:
    """Returns a sorted list of script names (without extension) in a folder."""
    folder_path = PACKAGE_ROOT / folder_name
    if not folder_path.is_dir():
        return []
    return sorted(
        f.stem for f in folder_path.glob("*.py") if not f.name.startswith("_")
    )


def _dispatch(group: str, script_name: str, script_args: list):
    """
    Sets up the environment and runs the target script via 'uv run'.
    """
    # Convert script-name to filename (e.g., plt-neb -> plt_neb.py)
    filename = f"{script_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / group / filename

    if not script_path.is_file():
        # Shouldn't happen thanks to argparse choices, but serves as a final
        # sanity check.
        rerr = f"Error: Script not found at '{script_path}'"
        raise (RuntimeError(rerr))
        sys.exit(1)

    command = ["uv", "run", str(script_path), *script_args]

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

    logging.info(f"--> Dispatching to: {' '.join(command)}", file=sys.stderr)

    try:
        # Use subprocess.run to block until completion
        subprocess.run(command, check=True, env=env)  # noqa: S603
    except FileNotFoundError:
        logging.info("Error: 'uv' command not found. Is it installed?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        prog="rgpycrumbs",
        description="A dispatcher that runs self-contained scripts using 'uv'.",
    )

    # Create subparsers for the groups (e.g., 'eon', 'prefix')
    subparsers = parser.add_subparsers(
        title="Command Groups", dest="group", required=True, metavar="GROUP"
    )

    # --- DYNAMIC DISCOVERY ---
    # Scan the package directory for subfolders (groups)
    valid_groups = sorted(
        d.name
        for d in PACKAGE_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )

    for group in valid_groups:
        available_scripts = _get_scripts_in_folder(group)
        if not available_scripts:
            continue

        # Create a subparser for this group (e.g., 'rgpycrumbs eon ...')
        group_parser = subparsers.add_parser(
            group,
            help=f"Tools in the '{group}' category.",
            description=f"Available scripts in '{group}': {', '.join(available_scripts)}",
        )

        # The script name is a positional argument, restricted to valid scripts
        group_parser.add_argument(
            "script", choices=available_scripts, help="The specific script to run."
        )

        # REMAINDER captures everything after the script name (flags, args, etc.)
        # and passes it raw to the target script.
        group_parser.add_argument(
            "script_args",
            nargs=argparse.REMAINDER,
            help="Arguments passed to the script.",
        )

    # Parse
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Dispatch
    _dispatch(args.group, args.script, args.script_args)


if __name__ == "__main__":
    main()
