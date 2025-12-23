import importlib.util

import pytest

# Define the requirements for each suite/marker
ENVIRONMENT_REQUIREMENTS = {
    "fragments": ["ase", "tblite"],
    "ptm": ["ase", "ovito"],
    "eon": ["ase", "eon"],
}


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
