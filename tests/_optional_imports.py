import importlib
import importlib.util

FIRST_PARTY_PREFIXES = ("rgpycrumbs", "chemparseplot")


def has_module_spec(module_name: str) -> bool:
    """Return True when Python can locate *module_name*."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Incomplete namespace packages (e.g. missing ``tblite``) raise on find_spec.
        return False


def optional_import_available(module_name: str) -> bool:
    """Return False only for genuinely missing third-party dependencies.

    Broken first-party imports should fail loudly instead of turning into test skips.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing and not missing.startswith(FIRST_PARTY_PREFIXES):
            return False
        raise
