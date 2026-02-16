try:
    from rgpycrumbs._version import __version__
except ImportError:
    __version__ = "unknown"


def __getattr__(name):
    if name == "surfaces":
        from rgpycrumbs import surfaces

        return surfaces
    if name == "basetypes":
        from rgpycrumbs import basetypes

        return basetypes
    if name == "interpolation":
        from rgpycrumbs import interpolation

        return interpolation
    raise AttributeError(f"module 'rgpycrumbs' has no attribute {name}")
