import importlib.util

import pytest


def require_module(module_name):
    """
    Checks if a module exists. If not, skips the test or module.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        pytest.skip(
            f"Module '{module_name}' not found. Skipping.", allow_module_level=True
        )


def get_missing_modules(*modules):
    """Returns a list of modules not currently installed."""
    return [mod for mod in modules if importlib.util.find_spec(mod) is None]
