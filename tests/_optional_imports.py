import importlib
import importlib.util

# Only rgpycrumbs itself is first-party for "fail loudly" purposes.
# chemparseplot is an optional peer dependency for eOn plot scripts and may be
# absent in the pure/test-extra CI matrix (PEP 723 scripts resolve it at runtime).
FIRST_PARTY_PREFIXES = ("rgpycrumbs",)


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
    Shared-library failures (e.g. ovito/PySide6 missing libEGL) count as unavailable.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing and not missing.startswith(FIRST_PARTY_PREFIXES):
            return False
        # First-party ModuleNotFoundError (broken package layout) fails loudly.
        if missing and missing.startswith(FIRST_PARTY_PREFIXES):
            raise
        return False
    except ImportError:
        # Optional peers missing, or native deps (libEGL) not on the runner.
        return False
    except OSError:
        # dlopen failures for optional GUI stacks (ovito / Qt).
        return False
