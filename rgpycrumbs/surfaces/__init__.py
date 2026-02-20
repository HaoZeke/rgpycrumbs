from rgpycrumbs.surfaces._base import (
    BaseGradientSurface,
    BaseSurface,
    generic_negative_mll,
    safe_cholesky_solve,
)
from rgpycrumbs.surfaces._kernels import (
    _imq_kernel_matrix,
    _matern_kernel_matrix,
    _tps_kernel_matrix,
)
from rgpycrumbs.surfaces.gradient import (
    GradientIMQ,
    GradientMatern,
    GradientRQ,
    GradientSE,
    NystromGradientIMQ,
)
from rgpycrumbs.surfaces.standard import FastIMQ, FastMatern, FastTPS

__all__ = [
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
    "safe_cholesky_solve",
]


def get_surface_model(name):
    """
    Factory function to retrieve surface model classes by name.

    Args:
        name: Model identifier (e.g., 'grad_matern', 'tps', 'imq').

    Returns:
        type: The model class. Defaults to GradientMatern.
    """
    models = {
        "grad_matern": GradientMatern,
        "grad_rq": GradientRQ,
        "grad_se": GradientSE,
        "grad_imq": GradientIMQ,
        "grad_imq_ny": NystromGradientIMQ,
        "matern": FastMatern,
        "imq": FastIMQ,
        "tps": FastTPS,
        "rbf": FastTPS,
    }
    return models.get(name, GradientMatern)
