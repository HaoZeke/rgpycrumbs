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
            hints = {
                "surfaces": "pip install rgpycrumbs[surfaces]",
                "interpolation": "pip install rgpycrumbs[interpolation]",
            }
            if name in hints:
                msg = (
                    f"rgpycrumbs.{name} requires optional deps. Install with:\n"
                    f"  {hints[name]}\n"
                    "Or set RGPYCRUMBS_AUTO_DEPS=1 to auto-resolve via uv."
                )
                raise ImportError(msg) from exc
            raise
    aerr = f"module 'rgpycrumbs' has no attribute {name}"
    raise AttributeError(aerr)
