import importlib
import importlib.util

import pytest

# Define the requirements for each suite/marker
ENVIRONMENT_REQUIREMENTS = {
    "fragments": ["ase", "tblite"],
    "ptm": ["ase", "ovito"],
    "eon": ["ase", "eon", "polars", "chemparseplot"],
    "align": ["ase", "numpy"],
    "pure": ["numpy"],
    "interpolation": ["numpy", "scipy"],
    "surfaces": ["jax"],
    "ira": ["ira_mod"],
}


def _try_import(module_name):
    """Return True if *module_name* can actually be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


# Test files that import PEP 723 dispatcher modules with heavy or
# conda-only dependencies.  Prevent pytest collection when the source
# module itself cannot import (regardless of what find_spec reports for
# individual package names).
_GUARDED_IMPORTS = {
    "test_detect_fragments.py": "rgpycrumbs.geom.detect_fragments",
    "test_eon_cli.py": "rgpycrumbs.eon.plt_neb",
    "test_ptmdisp.py": "rgpycrumbs.eon.ptmdisp",
    "test_ira.py": "rgpycrumbs.geom.ira",
    "test_surfaces.py": "jax",
}

collect_ignore = [f for f, mod in _GUARDED_IMPORTS.items() if not _try_import(mod)]


def check_missing_modules(marker_name):
    """
    Returns a list of missing modules for a given marker.
    """
    modules = ENVIRONMENT_REQUIREMENTS.get(marker_name, [])
    return [mod for mod in modules if importlib.util.find_spec(mod) is None]


def skip_if_not_env(marker_name):
    """
    Skips the entire module if dependencies for the marker remain uninstalled.
    """
    missing = check_missing_modules(marker_name)
    if missing:
        pytest.skip(
            f"Missing dependencies for '{marker_name}': {', '.join(missing)}",
            allow_module_level=True,
        )
