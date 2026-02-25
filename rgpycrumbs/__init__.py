try:
    from rgpycrumbs._version import __version__
except ImportError:
    __version__ = "unknown"


def __getattr__(name):
    if name == "surfaces":
        try:
            from rgpycrumbs import surfaces
        except ImportError as exc:
            msg = (
                "rgpycrumbs.surfaces requires jax. Install with:\n"
                "  pip install rgpycrumbs[surfaces]\n"
                "Or set RGPYCRUMBS_AUTO_DEPS=1 to auto-resolve via uv."
            )
            raise ImportError(msg) from exc
        return surfaces
    if name == "basetypes":
        from rgpycrumbs import basetypes

        return basetypes
    if name == "interpolation":
        try:
            from rgpycrumbs import interpolation
        except ImportError as exc:
            msg = (
                "rgpycrumbs.interpolation requires scipy. Install with:\n"
                "  pip install rgpycrumbs[interpolation]\n"
                "Or set RGPYCRUMBS_AUTO_DEPS=1 to auto-resolve via uv."
            )
            raise ImportError(msg) from exc
        return interpolation
    aerr = f"module 'rgpycrumbs' has no attribute {name}"
    raise AttributeError(aerr)
