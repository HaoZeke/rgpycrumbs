import contextlib
import importlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency registry
# ---------------------------------------------------------------------------
# Maps importable module names to (pip_spec, extra_name).
# Conda-only deps (ira_mod, tblite, ovito) are intentionally absent;
# they fall through to a helpful error suggesting pixi.
_DEPENDENCY_MAP: dict[str, tuple[str, str]] = {
    "jax": ("jax>=0.4", "surfaces"),
    "jaxlib": ("jax>=0.4", "surfaces"),
    "scipy": ("scipy>=1.11", "interpolation"),
    "scipy.interpolate": ("scipy>=1.11", "interpolation"),
    "scipy.spatial": ("scipy>=1.11", "analysis"),
    "scipy.spatial.distance": ("scipy>=1.11", "analysis"),
    "ase": ("ase>=3.22", "analysis"),
    "ase.data": ("ase>=3.22", "analysis"),
    "ase.neighborlist": ("ase>=3.22", "analysis"),
}

# CPU-only pip spec overrides for packages with heavy GPU backends.
# Applied when no CUDA device is detected to avoid pulling hundreds of
# megabytes of CUDA libraries.
_CPU_OVERRIDES: dict[str, str] = {
    "jax": "jax[cpu]>=0.4",
    "jaxlib": "jax[cpu]>=0.4",
}

# Cache the result of the CUDA probe so it runs at most once per process.
_cuda_available: bool | None = None


def _has_cuda() -> bool:
    """Return True when a usable NVIDIA GPU is present.

    Checks for ``nvidia-smi`` on PATH and verifies it exits cleanly. The
    result is cached for the lifetime of the process.
    """
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        _cuda_available = False
        return False
    try:
        subprocess.run(  # noqa: S603
            [nvsmi],
            check=True,
            capture_output=True,
            timeout=5,
        )
        _cuda_available = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        _cuda_available = False
    return _cuda_available


def _get_dep_cache_dir() -> Path:
    """Return the per-user dependency cache directory.

    Defaults to ``$XDG_CACHE_HOME/rgpycrumbs/deps/``
    (typically ``~/.cache/rgpycrumbs/deps/``).
    """
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "rgpycrumbs" / "deps"


def _resolve_pip_spec(module_name: str) -> str:
    """Return the pip install spec for *module_name*, respecting CUDA.

    If the host lacks a CUDA device and a CPU-only override exists, the
    override is returned instead of the default spec.
    """
    spec, _extra = _DEPENDENCY_MAP[module_name]
    if not _has_cuda():
        base_pkg = module_name.split(".", maxsplit=1)[0]
        spec = _CPU_OVERRIDES.get(base_pkg, spec)
    return spec


def _uv_install(package_spec: str, target: Path) -> None:
    """Install *package_spec* into *target* using uv (falling back to pip).

    Raises ``RuntimeError`` if both installers fail.
    """
    target.mkdir(parents=True, exist_ok=True)
    for installer in ("uv", "pip"):
        exe = shutil.which(installer)
        if exe is None:
            continue
        cmd = [exe, "pip", "install", "--target", str(target), package_spec]
        if installer == "pip":
            cmd = [exe, "install", "--target", str(target), package_spec]
        logger.info("rgpycrumbs: installing %s via %s", package_spec, installer)
        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
            )
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.debug("%s install failed: %s", installer, exc)
            continue
    msg = f"Failed to install {package_spec}. Ensure uv or pip is available on PATH."
    raise RuntimeError(msg)


def ensure_import(module_name: str):
    """Import *module_name* through a 5-step priority chain.

    1. Current environment (importlib)
    2. Parent environment (RGPYCRUMBS_PARENT_SITE_PACKAGES)
    3. uv cache directory on sys.path
    4. uv/pip install into cache (opt-in via RGPYCRUMBS_AUTO_DEPS=1)
    5. Raise ImportError with an actionable message

    Returns the imported module object.

    .. versionadded:: 1.3.0
    """
    # Step 1: current env
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    # Step 2: parent env
    mod = _import_from_parent_env(module_name)
    if mod is not None:
        return mod

    # Step 3: check uv cache
    cache_dir = _get_dep_cache_dir()
    cache_str = str(cache_dir)
    if cache_dir.is_dir() and cache_str not in sys.path:
        sys.path.insert(0, cache_str)
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass

    # Step 4: auto-install (opt-in)
    auto = os.environ.get("RGPYCRUMBS_AUTO_DEPS", "").strip()
    if auto == "1" and module_name in _DEPENDENCY_MAP:
        spec = _resolve_pip_spec(module_name)
        _uv_install(spec, cache_dir)
        if cache_str not in sys.path:
            sys.path.insert(0, cache_str)
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass

    # Step 5: actionable error
    if module_name in _DEPENDENCY_MAP:
        _spec, extra = _DEPENDENCY_MAP[module_name]
        msg = (
            f"Module '{module_name}' is not installed. Options:\n"
            f"  pip install rgpycrumbs[{extra}]\n"
            f"  RGPYCRUMBS_AUTO_DEPS=1 to auto-resolve via uv"
        )
    else:
        msg = (
            f"Module '{module_name}' is not installed and is not a "
            "pip-installable dependency of rgpycrumbs. "
            "For conda-only packages (ira_mod, tblite, ovito), "
            "use pixi: pixi install"
        )
    raise ImportError(msg)


class _LazyModule:
    """Proxy that defers ``ensure_import`` until first attribute access.

    After resolution the proxy replaces its own ``__dict__`` with the real
    module's attributes so subsequent access carries zero overhead.

    .. versionadded:: 1.3.0
    """

    def __init__(self, module_name: str):
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)

    def _resolve(self):
        mod = object.__getattribute__(self, "_module")
        if mod is None:
            name = object.__getattribute__(self, "_module_name")
            mod = ensure_import(name)
            object.__setattr__(self, "_module", mod)
        return mod

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)

    def __repr__(self):
        name = object.__getattribute__(self, "_module_name")
        mod = object.__getattribute__(self, "_module")
        if mod is None:
            return f"<LazyModule '{name}' (unresolved)>"
        return repr(mod)


def lazy_import(module_name: str) -> _LazyModule:
    """Return a lazy proxy for *module_name*.

    The actual import (via :func:`ensure_import`) is deferred until the
    first attribute access on the returned object.

    .. versionadded:: 1.3.0
    """
    return _LazyModule(module_name)


def getstrform(pathobj):
    """Return the absolute path as a string.

    .. versionadded:: 0.0.1
    """
    return str(pathobj.absolute())


def get_gitroot():
    """Return the root of the current git repository as a Path.

    .. versionadded:: 0.0.1
    """
    git_path = shutil.which("git") or "git"
    gitroot = Path(
        subprocess.run(  # noqa: S603
            [git_path, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            cwd=Path.cwd(),
        )
        .stdout.decode("utf-8")
        .strip()
    )
    return gitroot


@contextlib.contextmanager
def switchdir(path):
    """Context manager that temporarily changes the working directory.

    .. versionadded:: 0.0.1
    """
    curpath = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curpath)


def _import_from_parent_env(module_name: str):
    """
    Import a module from parent interpreter's site-packages as a fallback.
    Uses importlib to correctly handle nested modules (e.g. 'tblite.interface').
    """
    # 1. Try current environment
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    # 2. Check parent environment
    parent_paths = os.environ.get("RGPYCRUMBS_PARENT_SITE_PACKAGES", "")
    if not parent_paths:
        return None

    # 3. Temporarily extend sys.path
    # Filter out empty strings and paths already in sys.path
    paths_to_add = [p for p in parent_paths.split(os.pathsep) if p and p not in sys.path]
    sys.path.extend(paths_to_add)

    try:
        # importlib.import_module returns the actual leaf module (interface)
        # __import__ would have returned the top-level package (tblite)
        return importlib.import_module(module_name)
    except ImportError:
        return None
    finally:
        # Clean up sys.path
        for p in paths_to_add:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
