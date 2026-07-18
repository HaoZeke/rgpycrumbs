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


def _prefer_in_env_interpreter(
    is_dev: bool,
    *,
    force_uv: bool | None = None,
) -> bool:
    """Decide whether to run scripts with ``sys.executable`` instead of ``uv run``."""
    if is_dev or _env_flag("RGPYCRUMBS_DEV"):
        return True
    # force_uv: CLI/env/config layer (None → re-read env only for back-compat)
    if force_uv is True or (force_uv is None and _env_flag("RGPYCRUMBS_FORCE_UV")):
        return False
    if force_uv is False:
        # config explicitly disabled force_uv; still allow stack-ready in-env
        pass
    if shutil.which("uv") is None:
        return True
    # Active env already has readcon + plot deps: use it for CON metadata fidelity.
    return _in_env_stack_ready()


def _normalize_dist_name(name: str) -> str:
    """PEP 503-ish normalize for comparing package names."""
    return name.strip().lower().replace("_", "-")


def _pyproject_project_name(pyproject: Path) -> str | None:
    """Return ``[project].name`` from a pyproject.toml, if present."""
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    # Prefer tomllib when available (3.11+); fall back to a tiny scan.
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py310
        tomllib = None  # type: ignore[assignment]
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = None
        if isinstance(data, dict):
            project = data.get("project")
            if isinstance(project, dict):
                name = project.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    # Minimal fallback: first bare name = "..." under [project]
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and stripped.startswith("name"):
            _, _, rhs = stripped.partition("=")
            name = rhs.strip().strip("\"'")
            if name:
                return name
    return None


def _find_editable_source(package_name: str) -> Path | None:
    """Return the local project root for an *editable* install of *package_name*.

    Wheel installs under ``site-packages`` must never map to an unrelated
    monorepo ``pyproject.toml`` above the env (e.g. eOn root named
    ``eon-akmc`` when looking up ``chemparseplot``). Only return a path when:

    1. The module lives outside site-/dist-packages (true editable / src layout), and
    2. The nearest ``pyproject.toml`` has ``[project].name`` matching *package_name*.

    eOn tooling only needs ``eon-schema`` + ``pyeonclient``; never imply ``eon-akmc``.
    """
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None

    want = _normalize_dist_name(package_name)
    candidates: list[Path] = []
    if spec.origin:
        candidates.append(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        candidates.extend(Path(loc).resolve() for loc in spec.submodule_search_locations)

    for candidate in candidates:
        search_root = candidate if candidate.is_dir() else candidate.parent
        # Regular wheels live under site-packages / dist-packages — not editable.
        if any(part in {"site-packages", "dist-packages"} for part in search_root.parts):
            continue
        for parent in (search_root, *search_root.parents):
            pyproject = parent / "pyproject.toml"
            if not pyproject.is_file():
                continue
            proj_name = _pyproject_project_name(pyproject)
            if proj_name is None:
                return None
            if _normalize_dist_name(proj_name) == want:
                return parent
            # First pyproject is a different project — do not climb further.
            return None
    return None


def _uv_editable_sources() -> list[Path]:
    """Return local editable roots that should satisfy script dependencies.

    Only true editable checkouts of chemparseplot (name-matched). Never the
    eOn monorepo / eon-akmc tree.
    """
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


def _load_lock_pins_or_exit(lock_path: str | Path) -> dict[str, str]:
    """Load PyPI pins from uv.lock / pylock / CycloneDX; exit on failure."""
    from rgpycrumbs.locks import detect_lock_format, load_pypi_pins

    path_str = str(lock_path)
    try:
        pins = load_pypi_pins(path_str)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error: invalid lock/SBOM: {exc}", err=True)
        sys.exit(1)
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    fmt = detect_lock_format(path_str)
    click.echo(
        f"--> Lock ({fmt.value}) {path_str}: {len(pins)} PyPI pin(s) for constraints",
        err=True,
    )
    return pins


def _dispatch(
    group: str,
    script_name: str,
    script_args: tuple,
    is_dev: bool = False,
    is_verbose: bool = False,
    lock_path: str | None = None,
    sbom_path: str | None = None,
):
    """
    Sets up the environment and runs the target script via 'uv run'.

    Preferred entry is this dispatcher (``rgpycrumbs`` / ``python -m
    rgpycrumbs.cli``). Raw ``uv run <script.py>`` is not the primary path.

    Pins/lock resolution (highest wins): CLI ``--lock``/``--sbom`` → env
    ``RGPYCRUMBS_LOCK``/``_SBOM`` → project ``rgpycrumbs.toml`` → user
    ``~/.config/rgpycrumbs/config.toml``. Formats: uv.lock, PEP 751 pylock,
    CycloneDX. TOML ``[pins.packages]`` merge on top of the lock file.
    """
    import json
    import tempfile

    from rgpycrumbs.config import (
        load_config,
        resolve_auto_deps_default,
        resolve_force_uv,
        resolve_lock_path_layered,
    )
    from rgpycrumbs.locks import (
        PINS_ENV,
        SBOM_PINS_ENV,
        normalize_pypi_name,
        pins_to_constraint_lines,
    )

    # Convert script-name to filename (e.g., plt-neb -> plt_neb.py)
    filename = f"{script_name.replace('-', '_')}.py"
    script_path = PACKAGE_ROOT / group / filename

    if not script_path.is_file():
        click.echo(f"Error: Script not found at '{script_path}'", err=True)
        sys.exit(1)

    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        click.echo(f"Error: config: {exc}", err=True)
        sys.exit(1)

    if is_verbose and cfg.sources:
        click.echo(
            "VERBOSE: config sources -> " + ", ".join(str(p) for p in cfg.sources),
            err=True,
        )

    resolved_lock = resolve_lock_path_layered(
        cli_lock=lock_path,
        cli_sbom=sbom_path,
        config=cfg,
    )
    pins: dict[str, str] = {}
    if resolved_lock is not None:
        pins = _load_lock_pins_or_exit(resolved_lock)
    # [pins.packages] from TOML override lock-file versions
    for name, ver in cfg.merged_package_pins_normalized().items():
        pins[name] = ver
    if cfg.package_pins and is_verbose:
        click.echo(
            f"VERBOSE: TOML package pins -> {cfg.merged_package_pins_normalized()}",
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
    # - uv run → PEP 723 header deps (+ optional lock/SBOM constraints)
    # - in-env  → ensure_import cache installs for heavies (jax, adjustText)
    if env.get("RGPYCRUMBS_AUTO_DEPS", "").strip() == "":
        env["RGPYCRUMBS_AUTO_DEPS"] = resolve_auto_deps_default(config=cfg)

    if pins:
        # ensure_import expects normalized keys; write all known env aliases
        from rgpycrumbs.locks import PINS_ENV_LEGACY

        pin_json = json.dumps({normalize_pypi_name(k): v for k, v in pins.items()})
        env[PINS_ENV] = pin_json
        env[PINS_ENV_LEGACY] = pin_json
        env[SBOM_PINS_ENV] = pin_json

    force_uv = resolve_force_uv(is_dev=is_dev, config=cfg)
    use_in_env = _prefer_in_env_interpreter(is_dev, force_uv=force_uv)
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
            # Ephemeral constraints for uv resolver; cleaned after process exits.
            fd, tmp_name = tempfile.mkstemp(prefix="rgpycrumbs-lock-", suffix=".txt")
            os.close(fd)
            constraints_path = Path(tmp_name)
            constraints_path.write_text(
                "\n".join(pins_to_constraint_lines(pins)) + "\n",
                encoding="utf-8",
            )
            command.extend(["--constraints", str(constraints_path)])
            env["UV_CONSTRAINT"] = str(constraints_path)
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
        lock_path = ctx.obj.get("lock_path") if ctx.obj else None
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
                lock_path=lock_path,
                sbom_path=sbom_path,
            )
            return

        _dispatch(
            group_name,
            display_name,
            tuple(ctx.args),
            is_dev=is_dev,
            is_verbose=is_verbose,
            lock_path=lock_path,
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
    "--lock",
    "lock_path",
    type=click.Path(path_type=str),
    default=None,
    help=(
        "Optional lock path: uv.lock, PEP 751 pylock.toml, or CycloneDX JSON. "
        "Precedence: CLI > env RGPKGS_LOCK > project rgpkgs.toml > "
        "~/.config/rgpkgs/config.toml (suite-wide). Missing path fails clearly."
    ),
)
@click.option(
    "--sbom",
    "sbom_path",
    type=click.Path(path_type=str),
    default=None,
    help=(
        "Alias for --lock (CycloneDX JSON, e.g. eb-stack --sbom-out). "
        "Also set via RGPYCRUMBS_SBOM."
    ),
)
@click.version_option(package_name="rgpycrumbs")
@click.pass_context
def main(ctx, dev, verbose, lock_path, sbom_path):
    """Dispatcher for PEP 723 scripts (preferred entry; not raw uv run).

    Each tool is a self-contained script. Prefer::

        rgpycrumbs eon plt-neb ...
        python -m rgpycrumbs.cli eon plt-neb ...

    Suite config (shared with chemparseplot and other rgpkgs): project
    ``rgpkgs.toml`` and user ``~/.config/rgpkgs/config.toml`` for lock pins
    and dispatch defaults. Legacy ``rgpycrumbs.toml`` paths still work.
    """
    # Ensure ctx.obj is a dictionary so we can store state in it
    ctx.ensure_object(dict)
    ctx.obj["is_dev"] = dev
    ctx.obj["is_verbose"] = verbose
    ctx.obj["lock_path"] = lock_path
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


@main.command("config")
@click.argument("action", default="show", type=click.Choice(["show"]))
def config_cmd(action: str) -> None:
    """Show merged suite config (rgpkgs.toml + ~/.config/rgpkgs/…)."""
    from rgpycrumbs.config import (
        load_config,
        resolve_auto_deps_default,
        resolve_force_uv,
        resolve_lock_path_layered,
        user_config_path,
    )

    if action != "show":
        return
    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    lock = resolve_lock_path_layered(config=cfg)
    click.echo(f"user_config:     {user_config_path()}")
    click.echo(f"sources:         {', '.join(str(p) for p in cfg.sources) or '(none)'}")
    click.echo(f"lock_path:       {lock or '(none)'}")
    click.echo(f"force_uv:        {resolve_force_uv(is_dev=False, config=cfg)}")
    click.echo(f"auto_deps_def:   {resolve_auto_deps_default(config=cfg)}")
    pins = cfg.merged_package_pins_normalized()
    click.echo(f"package_pins:    {pins or '{}'}")
    if cfg.tool_tables:
        click.echo(f"tool_tables:     {list(cfg.tool_tables)}")


if __name__ == "__main__":
    main()
