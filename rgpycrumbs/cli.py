import argparse
import logging
import os
import site
import subprocess
import sys
from pathlib import Path

# Configure logging to output to stderr
logging.basicConfig(level=logging.INFO, format="%(message)s")

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
        rerr = f"Error: Script not found at '{script_path}'"
        logging.error(rerr)
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

    logging.info(f"--> Dispatching to: {' '.join(command)}")

    try:
        # Use subprocess.run to block until completion
        subprocess.run(command, check=True, env=env)  # noqa: S603
    except FileNotFoundError:
        logging.error("Error: 'uv' command not found. Is it installed?")
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
        file_stems = _get_scripts_in_folder(group)
        if not file_stems:
            continue

        # Create a display list with hyphens (e.g. plt-neb)
        display_scripts = [s.replace("_", "-") for s in file_stems]

        # Build a set of all valid inputs (both _ and - forms) for manual validation
        valid_inputs = set(file_stems) | set(display_scripts)

        # Create a subparser for this group
        group_parser = subparsers.add_parser(
            group,
            help=f"Tools in the '{group}' category.",
            description=f"Available scripts in '{group}':\n"
            f"  {', '.join(display_scripts)}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # The script name is a positional argument.
        # We REMOVE 'choices' to hide the ugly list from the usage string.
        # We manually validate later.
        group_parser.add_argument(
            "script",
            help="The specific script to run (e.g., 'plt-neb')."
            " Accepts names with '_' or '-'.",
            metavar="SCRIPT",
        )

        # REMAINDER captures everything after the script name (flags, args, etc.)
        group_parser.add_argument(
            "script_args",
            nargs=argparse.REMAINDER,
            help="Arguments passed to the script.",
        )

        # Attach the valid inputs to the parser instance so we can check later
        # (This is a bit hacky but keeps the main() logic clean)
        group_parser.set_defaults(valid_inputs=valid_inputs)

    # Parse
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Manual Validation of the script name
    if args.script not in args.valid_inputs:
        logging.error(
            f"Error: Invalid script '{args.script}' for group '{args.group}'."
        )
        # Re-generate the "nice" list for the error message
        file_stems = _get_scripts_in_folder(args.group)
        display_scripts = [s.replace("_", "-") for s in file_stems]
        logging.error(f"Available scripts: {', '.join(display_scripts)}")
        sys.exit(1)

    # Dispatch
    _dispatch(args.group, args.script, args.script_args)


if __name__ == "__main__":
    main()
