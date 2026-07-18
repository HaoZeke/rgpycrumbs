import importlib
import importlib.util

import pytest

from tests._optional_imports import optional_import_available

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
        return optional_import_available(module_name)
    except Exception:
        # Never let collection die on optional heavy stacks (ovito/Qt/libEGL).
        return False


# Test files that import PEP 723 dispatcher modules with heavy or
# pixi-only dependencies.  Prevent pytest collection when the source
# module itself cannot import (regardless of what find_spec reports for
# individual package names).
#
# Prefer lightweight probes (package name) over importing full scripts that
# pull GUI native libs at module level (ptmdisp → ovito → PySide6).
_GUARDED_IMPORTS = {
    "test_detect_fragments.py": "rgpycrumbs.geom.detect_fragments",
    # Do not import plt_neb at collection (PEP 723 script pulls adjustText/etc.).
    "test_eon_cli.py": "chemparseplot",
    "test_ptmdisp.py": "ovito",
    "test_ira.py": "rgpycrumbs.geom.ira",
    "test_surfaces.py": "jax",
}

collect_ignore = [f for f, mod in _GUARDED_IMPORTS.items() if not _try_import(mod)]


def check_missing_modules(marker_name):
    """
    Returns a list of missing modules for a given marker.
    """
    modules = ENVIRONMENT_REQUIREMENTS.get(marker_name, [])
    missing = []
    for mod in modules:
        if mod in {"rgpycrumbs", "chemparseplot"}:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
            elif not optional_import_available(mod):
                missing.append(mod)
        elif importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return missing


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
