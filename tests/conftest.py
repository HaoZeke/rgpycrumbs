import importlib

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


def check_missing_modules(marker_name):
    """
    Returns a list of missing modules for a given marker.
    Uses actual import rather than find_spec to catch broken installs.
    """
    modules = ENVIRONMENT_REQUIREMENTS.get(marker_name, [])
    missing = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except (ImportError, ModuleNotFoundError):
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
