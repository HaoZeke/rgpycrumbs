import importlib

_LAZY_IMPORTS = {
    # _base
    "BaseGradientSurface": "_base",
    "BaseSurface": "_base",
    "generic_negative_mll": "_base",
    "safe_cholesky_solve": "_base",
    # _kernels
    "_imq_kernel_matrix": "_kernels",
    "_matern_kernel_matrix": "_kernels",
    "_tps_kernel_matrix": "_kernels",
    # gradient (requires jax)
    "GradientIMQ": "gradient",
    "GradientMatern": "gradient",
    "GradientRQ": "gradient",
    "GradientSE": "gradient",
    "NystromGradientIMQ": "gradient",
    # standard (requires jax)
    "FastIMQ": "standard",
    "FastMatern": "standard",
    "FastTPS": "standard",
}

# Submodules that require jax at import time
_JAX_SUBMODULES = frozenset({"gradient", "standard", "_kernels"})

NYSTROM_THRESHOLD = 1000
NYSTROM_N_INDUCING_DEFAULT = 300


def nystrom_paths_needed(n_inducing, images_per_step):
    """Number of optimization steps the Nystrom approximation actually samples.

    Mirrors the structured sampling in :class:`NystromGradientIMQ._fit`:
    ``paths_to_sample = max(1, n_inducing // nimags)``, plus one buffer step.
    """
    return max(1, -(-n_inducing // images_per_step)) + 1  # ceil div + buffer


__all__ = [
    "NYSTROM_N_INDUCING_DEFAULT",
    "NYSTROM_THRESHOLD",
    "BaseGradientSurface",
    "BaseSurface",
    "FastIMQ",
    "FastMatern",
    "FastTPS",
    "GradientIMQ",
    "GradientMatern",
    "GradientRQ",
    "GradientSE",
    "NystromGradientIMQ",
    "_imq_kernel_matrix",
    "_matern_kernel_matrix",
    "_tps_kernel_matrix",
    "generic_negative_mll",
    "get_surface_model",
    "nystrom_paths_needed",
    "safe_cholesky_solve",
]


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        target = _LAZY_IMPORTS[name]
        if target in _JAX_SUBMODULES:
            from rgpycrumbs._aux import ensure_import

            ensure_import("jax")
        submod = importlib.import_module(f"rgpycrumbs.surfaces.{target}")
        return getattr(submod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_surface_model(name):
    """
    Factory function to retrieve surface model classes by name.

    .. versionadded:: 1.0.0

    Args:
        name: Model identifier (e.g., 'grad_matern', 'tps', 'imq').

    Returns:
        type: The model class. Defaults to GradientMatern.
    """
    _models = {
        "grad_matern": "GradientMatern",
        "grad_rq": "GradientRQ",
        "grad_se": "GradientSE",
        "grad_imq": "GradientIMQ",
        "grad_imq_ny": "NystromGradientIMQ",
        "matern": "FastMatern",
        "imq": "FastIMQ",
        "tps": "FastTPS",
        "rbf": "FastTPS",
    }
    class_name = _models.get(name, "GradientMatern")
    return __getattr__(class_name)
