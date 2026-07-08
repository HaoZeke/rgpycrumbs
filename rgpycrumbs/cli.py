import contextlib
import importlib.util
import logging
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

import click

# Configure logging to output to stderr
logging.basicConfig(level=logging.INFO, format="%(message)s")

# The directory where cli.py is located
PACKAGE_ROOT = Path(__file__).parent.resolve()

# Modules that signal a usable in-env eOn/plot stack (readcon-native CON I/O).
_IN_ENV_STACK_MODULES = ("readcon", "matplotlib", "polars", "ase", "chemparseplot")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _in_env_stack_ready() -> bool:
    """True when the active interpreter already has the eOn plot/readcon stack.

    Prefer this over ``uv run`` isolation: a host ``uv`` cache can pull a different
    Python (e.g. 3.12) while site-packages still hold 3.13 extension modules
    (numpy), which breaks plotting of readcon-core CON outputs from eOn 2.16+.
    """
    for mod in _IN_ENV_STACK_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


def _prefer_in_env_interpreter(is_dev: bool) -> bool:
    """Decide whether to run scripts with ``sys.executable`` instead of ``uv run``."""
    if is_dev or _env_flag("RGPYCRUMBS_DEV"):
        return True
    if _env_flag("RGPYCRUMBS_FORCE_UV"):
        return False
    if shutil.which("uv") is None:
        return True
    # Active env already has readcon + plot deps: use it for CON metadata fidelity.
    return _in_env_stack_ready()


def _find_editable_source(package_name: str) -> Path | None:
    """Return the local project root for an editable package, if any."""
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None

    candidates: list[Path] = []
    if spec.origin:
        candidates.append(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        candidates.extend(Path(loc).resolve() for loc in spec.submodule_search_locations)

    for candidate in candidates:
        search_root = candidate if candidate.is_dir() else candidate.parent
        for parent in (search_root, *search_root.parents):
            if (parent / "pyproject.toml").is_file():
                return parent
    return None


def _uv_editable_sources() -> list[Path]:
    """Return local editable roots that should satisfy script dependencies."""
    sources: list[Path] = []
    for package_name in ("chemparseplot",):
        source = _find_editable_source(package_name)
        if source is not None:
            sources.append(source)
    return sources


def _get_scripts_in_folder(folder_name: str) -> list[str]:
    """Returns a sorted list of CLI script names (without extension) in a folder.

    Excludes library modules (hdf5_io, plotting) and internal files (__init__, _*).
    Includes actual CLI entry point scripts.
    """
    folder_path = PACKAGE_ROOT / folder_name
    if not folder_path.is_dir():
        return []

    # Library modules to exclude
    library_modules = {"hdf5_io", "plotting", "utils", "helpers", "seed_dimers"}

    scripts = []
    for f in folder_path.glob("*.py"):
        if f.name.startswith("_"):
            continue
        stem = f.stem
        # Skip library modules and __init__
        if stem in library_modules or stem == "__init__":
            continue
        # Strip 'cli_' prefix if present for cleaner command names
        if stem.startswith("cli_"):
            stem = stem[4:]
        scripts.append(stem)

    return sorted(scripts)


def _resolve_sbom_path(explicit: str | None = None) -> str | None:
    """Return SBOM path from *explicit* CLI value or ``RGPYCRUMBS_SBOM``."""
    from rgpycrumbs.sbom import SBOM_PATH_ENV

    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env_path = os.environ.get(SBOM_PATH_ENV, "").strip()
    return env_path or None


def _load_sbom_pins_or_exit(sbom_path: str) -> dict[str, str]:
    """Load PyPI pins from CycloneDX path; exit with a clear error on failure."""
    from rgpycrumbs.sbom import load_pypi_pins

    try:
        pins = load_pypi_pins(sbom_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error: invalid SBOM: {exc}", err=True)
        sys.exit(1)
    return pins


def _dispatch(
    group: str,
    script_name: str,
    script_args: tuple,
    is_dev: bool = False,
    is_verbose: bool = False,
    sbom_path: str | None = None,
):
    """
    Sets up the environment and runs the target script via 'uv run'.

    Preferred entry is this dispatcher (``rgpycrumbs`` / ``python -m
    rgpycrumbs.cli``). Raw ``uv run <script.py>`` is not the primary path.
    Optional CycloneDX SBOM (``--sbom`` / ``RGPYCRUMBS_SBOM``) constrains
    PyPI installs when provided.
    """
    import json
    import tempfile

    from rgpycrumbs.sbom import SBOM_PINS_ENV, pins_to_constraint_lines

    # Convert script-name to filename (e.g., plt-neb -> plt_neb.py)
    filename = f"{script_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / group / filename

    if not script_path.is_file():
        click.echo(f"Error: Script not found at '{script_path}'", err=True)
        sys.exit(1)

    resolved_sbom = _resolve_sbom_path(sbom_path)
    pins: dict[str, str] = {}
    if resolved_sbom is not None:
        pins = _load_sbom_pins_or_exit(resolved_sbom)
        click.echo(
            f"--> SBOM {resolved_sbom}: {len(pins)} PyPI pin(s) for install constraints",
            err=True,
        )

    # --- SETUP ENVIRONMENT ---
    env = os.environ.copy()

    # Fallback imports
    try:
        site_packages = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            site_packages.append(user_site)
        else:
            site_packages.extend(user_site)
        env["RGPYCRUMBS_PARENT_SITE_PACKAGES"] = os.pathsep.join(site_packages)
    except (AttributeError, ImportError):
        pass

    # Add parent dir to PYTHONPATH for internal imports (e.g. rgpycrumbs._aux)
    project_root = str(PACKAGE_ROOT.parent)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_pythonpath}"

    # CLI owns dependency resolution for dispatched scripts:
    # - uv run → PEP 723 header deps (+ optional SBOM constraints)
    # - in-env  → ensure_import cache installs for heavies (jax, adjustText)
    # Do not force hosts to pre-declare those. Explicit 0/false still disables.
    if env.get("RGPYCRUMBS_AUTO_DEPS", "").strip() == "":
        env["RGPYCRUMBS_AUTO_DEPS"] = "1"

    if pins:
        env[SBOM_PINS_ENV] = json.dumps(pins)

    use_in_env = _prefer_in_env_interpreter(is_dev)
    constraints_path: Path | None = None
    if use_in_env:
        command = [sys.executable, str(script_path), *script_args]
        if is_verbose or (not is_dev and _in_env_stack_ready()):
            click.echo(
                "--> Using active interpreter (readcon/plot stack present or --dev)",
                err=True,
            )
    else:
        command = ["uv", "run"]
        if pins:
            # Ephemeral constraints file for uv; cleaned after process exits.
            fd, tmp_name = tempfile.mkstemp(prefix="rgpycrumbs-sbom-", suffix=".txt")
            os.close(fd)
            constraints_path = Path(tmp_name)
            constraints_path.write_text(
                "\n".join(pins_to_constraint_lines(pins)) + "\n",
                encoding="utf-8",
            )
            command.extend(["--constraints", str(constraints_path)])
        for source in _uv_editable_sources():
            command.extend(["--with-editable", str(source)])
        command.extend([str(script_path), *script_args])

    if is_verbose:
        click.echo(f"VERBOSE: Resolved script path -> {script_path}", err=True)
        click.echo(f"VERBOSE: Constructed command -> {' '.join(command)}", err=True)

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
    finally:
        if constraints_path is not None:
            with contextlib.suppress(OSError):
                constraints_path.unlink(missing_ok=True)


def _make_script_command(group_name: str, script_stem: str) -> click.Command:
    """Creates a click command that dispatches to a PEP 723 script.

    For full option help, run the script directly:
        python -m rgpycrumbs.<group>.<script> --help
    """
    display_name = script_stem.replace("_", "-")

    @click.command(
        name=display_name,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        add_help_option=False,  # Pass --help to underlying script
    )
    @click.pass_context
    def cmd(ctx):
        # Retrieve flags safely from the parent context
        is_dev = ctx.obj.get("is_dev", False) if ctx.obj else False
        is_verbose = ctx.obj.get("is_verbose", False) if ctx.obj else False
        sbom_path = ctx.obj.get("sbom_path") if ctx.obj else None

        # Pass through --help to underlying script
        if "--help" in ctx.args or "-h" in ctx.args:
            # Run script with --help to show actual options
            _dispatch(
                group_name,
                display_name,
                tuple(ctx.args),
                is_dev=is_dev,
                is_verbose=False,  # Don't add verbose noise to help output
                sbom_path=sbom_path,
            )
            return

        _dispatch(
            group_name,
            display_name,
            tuple(ctx.args),
            is_dev=is_dev,
            is_verbose=is_verbose,
            sbom_path=sbom_path,
        )

    cmd.help = f"""Run the {display_name} script.

For full option documentation, run:
    python -m rgpycrumbs.{group_name}.{display_name} --help

Or use --help flag which will be passed to the script:
    rgpycrumbs {group_name} {display_name} --help
"""
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
@click.option(
    "--sbom",
    "sbom_path",
    type=click.Path(path_type=str),
    default=None,
    help=(
        "Optional CycloneDX JSON SBOM path (e.g. eb-stack --sbom-out). "
        "PyPI components become install pins for uv/AUTO_DEPS. "
        "Also set via RGPYCRUMBS_SBOM. Missing path fails clearly."
    ),
)
@click.version_option(package_name="rgpycrumbs")
@click.pass_context
def main(ctx, dev, verbose, sbom_path):
    """Dispatcher for PEP 723 scripts (preferred entry; not raw uv run).

    Each tool is a self-contained script. Prefer::

        rgpycrumbs eon plt-neb ...
        python -m rgpycrumbs.cli eon plt-neb ...

    over ``uv run path/to/plt_neb.py`` (the dispatcher sets PYTHONPATH,
    AUTO_DEPS, optional SBOM pins, and editable peers).
    """
    # Ensure ctx.obj is a dictionary so we can store state in it
    ctx.ensure_object(dict)
    ctx.obj["is_dev"] = dev
    ctx.obj["is_verbose"] = verbose
    ctx.obj["sbom_path"] = sbom_path


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
