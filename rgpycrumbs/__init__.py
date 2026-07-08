try:
    from rgpycrumbs._version import __version__
except ImportError:
    __version__ = "unknown"


def __getattr__(name):
    import importlib

    _LAZY = {"surfaces", "basetypes", "interpolation", "geom"}
    if name in _LAZY:
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError as exc:
            msg = (
                f"rgpycrumbs.{name} requires optional deps. "
                "Set RGPYCRUMBS_AUTO_DEPS=1 to auto-resolve via uv, "
                "or install the needed packages (e.g. jax, scipy) yourself. "
                "CLI tools use PEP 723 + uv run instead."
            )
            raise ImportError(msg) from exc
    aerr = f"module 'rgpycrumbs' has no attribute {name}"
    raise AttributeError(aerr)
