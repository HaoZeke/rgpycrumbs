import contextlib
import importlib
import os
import subprocess
import sys
from pathlib import Path


def getstrform(pathobj):
    return str(pathobj.absolute())


def get_gitroot():
    gitroot = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
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
    paths_to_add = [
        p for p in parent_paths.split(os.pathsep) if p and p not in sys.path
    ]
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
