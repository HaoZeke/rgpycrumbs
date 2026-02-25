"""Kernel functions for Gaussian process surface fitting.

Provides element-wise and matrix kernel functions for Matern 5/2, inverse
multiquadric (IMQ), squared exponential (SE), rational quadratic (RQ), and
thin-plate spline (TPS) kernels.  Each kernel family includes a ``full_covariance_*``
function that builds the gradient-enhanced covariance block
``[[k, dk/dx2], [dk/dx1, d2k/dx1dx2]]`` via JAX automatic differentiation.

.. versionadded:: 1.0.0
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap

# ==============================================================================
# TPS KERNELS
# ==============================================================================


@jit
def _tps_kernel_matrix(x):
    """Compute the thin-plate spline kernel matrix.

    Evaluates ``K_{ij} = r_{ij}^2 * log(r_{ij})`` where ``r_{ij}`` is the
    Euclidean distance between points ``x_i`` and ``x_j``.

    Args:
        x: Input points, shape ``(N, D)``.

    Returns:
        Kernel matrix, shape ``(N, N)``.

    .. versionadded:: 1.0.0
    """
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K = r**2 * jnp.log(r)
    return K


# ==============================================================================
# MATERN KERNELS
# ==============================================================================


@jit
def _matern_kernel_matrix(x, length_scale):
    """Compute the Matern 5/2 kernel matrix.

    Evaluates k(r) = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l).

    Args:
        x: Input points, shape ``(N, D)``.
        length_scale: Kernel length scale parameter.

    Returns:
        Kernel matrix, shape ``(N, N)``.

    .. versionadded:: 1.0.0
    """
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)

    # Matern 5/2 Kernel
    # k(r) = (1 + sqrt(5)r/l + 5r^2/3l^2) * exp(-sqrt(5)r/l)
    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    K = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(-sqrt5_r_l)
    return K


def matern_kernel_elem(x1, x2, length_scale=1.0):
    """Evaluate the Matern 5/2 kernel for a single pair of points.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        length_scale: Kernel length scale parameter.

    Returns:
        Scalar kernel value.

    .. versionadded:: 1.0.0
    """
    d2 = jnp.sum((x1 - x2) ** 2)
    r = jnp.sqrt(d2 + 1e-12)
    ls = jnp.squeeze(length_scale)
    sqrt5_r_l = jnp.sqrt(5.0) * r / ls
    val = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * ls**2)) * jnp.exp(-sqrt5_r_l)
    return val


def full_covariance_matern(x1, x2, length_scale):
    """Build the gradient-enhanced covariance block for the Matern 5/2 kernel.

    Constructs a ``(D+1, D+1)`` matrix containing the energy-energy,
    energy-gradient, gradient-energy, and gradient-gradient covariance entries
    using JAX automatic differentiation.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        length_scale: Kernel length scale parameter.

    Returns:
        Covariance block, shape ``(D+1, D+1)``.

    .. versionadded:: 1.1.0
    """
    k_ee = matern_kernel_elem(x1, x2, length_scale)
    k_ed = jax.grad(matern_kernel_elem, argnums=1)(x1, x2, length_scale)
    k_de = jax.grad(matern_kernel_elem, argnums=0)(x1, x2, length_scale)
    k_dd = jax.jacfwd(jax.grad(matern_kernel_elem, argnums=1), argnums=0)(
        x1, x2, length_scale
    )
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_matern_grad_map = vmap(
    vmap(full_covariance_matern, (None, 0, None)), (0, None, None)
)


# ==============================================================================
# IMQ KERNELS
# ==============================================================================


@jit
def _imq_kernel_matrix(x, epsilon):
    """Compute the inverse multiquadric (IMQ) kernel matrix.

    Evaluates k(r) = 1 / sqrt(r^2 + epsilon^2).

    Args:
        x: Input points, shape ``(N, D)``.
        epsilon: Shape parameter controlling kernel width.

    Returns:
        Kernel matrix, shape ``(N, N)``.

    .. versionadded:: 1.0.0
    """
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    K = 1.0 / jnp.sqrt(d2 + epsilon**2)
    return K


def imq_kernel_elem(x1, x2, epsilon=1.0):
    """Evaluate the IMQ kernel for a single pair of points.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        epsilon: Shape parameter controlling kernel width.

    Returns:
        Scalar kernel value.

    .. versionadded:: 1.0.0
    """
    d2 = jnp.sum((x1 - x2) ** 2)
    val = 1.0 / jnp.sqrt(d2 + epsilon**2)
    return val


def full_covariance_imq(x1, x2, epsilon):
    """Build the gradient-enhanced covariance block for the IMQ kernel.

    Constructs a ``(D+1, D+1)`` matrix containing energy-energy,
    energy-gradient, gradient-energy, and gradient-gradient covariance entries.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        epsilon: Shape parameter controlling kernel width.

    Returns:
        Covariance block, shape ``(D+1, D+1)``.

    .. versionadded:: 1.1.0
    """
    k_ee = imq_kernel_elem(x1, x2, epsilon)
    k_ed = jax.grad(imq_kernel_elem, argnums=1)(x1, x2, epsilon)
    k_de = jax.grad(imq_kernel_elem, argnums=0)(x1, x2, epsilon)
    k_dd = jax.jacfwd(jax.grad(imq_kernel_elem, argnums=1), argnums=0)(x1, x2, epsilon)
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_imq_grad_map = vmap(vmap(full_covariance_imq, (None, 0, None)), (0, None, None))


# ==============================================================================
# SE KERNELS
# ==============================================================================


def se_kernel_elem(x1, x2, length_scale=1.0):
    """Evaluate the squared exponential (SE) kernel for a single pair of points.

    Computes k(r) = exp(-r^2 / (2 * l^2)).

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        length_scale: Kernel length scale parameter.

    Returns:
        Scalar kernel value.

    .. versionadded:: 1.0.0
    """
    d2 = jnp.sum((x1 - x2) ** 2)
    ls = jnp.maximum(length_scale, 1e-5)
    val = jnp.exp(-d2 / (2.0 * ls**2))
    return val


def full_covariance_se(x1, x2, length_scale):
    """Build the gradient-enhanced covariance block for the SE kernel.

    Constructs a ``(D+1, D+1)`` matrix containing energy-energy,
    energy-gradient, gradient-energy, and gradient-gradient covariance entries.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        length_scale: Kernel length scale parameter.

    Returns:
        Covariance block, shape ``(D+1, D+1)``.

    .. versionadded:: 1.1.0
    """
    k_ee = se_kernel_elem(x1, x2, length_scale)
    k_ed = jax.grad(se_kernel_elem, argnums=1)(x1, x2, length_scale)
    k_de = jax.grad(se_kernel_elem, argnums=0)(x1, x2, length_scale)
    k_dd = jax.jacfwd(jax.grad(se_kernel_elem, argnums=1), argnums=0)(
        x1, x2, length_scale
    )
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_se_grad_map = vmap(vmap(full_covariance_se, (None, 0, None)), (0, None, None))


# ==============================================================================
# RQ KERNELS
# ==============================================================================


def rq_kernel_base(x1, x2, length_scale, alpha):
    """Evaluate the rational quadratic (RQ) base kernel.

    Computes k(r) = (1 + r^2 / (2 * alpha * l^2))^(-alpha).

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        length_scale: Kernel length scale parameter.
        alpha: Shape parameter controlling the mixture of length scales.

    Returns:
        Scalar kernel value.

    .. versionadded:: 1.1.0
    """
    d2 = jnp.sum((x1 - x2) ** 2)
    base = 1.0 + d2 / (2.0 * alpha * (length_scale**2) + 1e-6)
    val = base ** (-alpha)
    return val


def rq_kernel_elem(x1, x2, params):
    """Evaluate the RQ kernel with mirror symmetry for a single pair.

    Computes k(x1, x2) + k(flip(x1), x2) to enforce symmetry under
    coordinate reversal.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        params: Array of ``[length_scale, alpha]``.

    Returns:
        Scalar kernel value (sum of direct and mirrored terms).

    .. versionadded:: 1.1.0
    """
    length_scale = params[0]
    alpha = params[1]
    k_direct = rq_kernel_base(x1, x2, length_scale, alpha)
    k_mirror = rq_kernel_base(x1[::-1], x2, length_scale, alpha)
    return k_direct + k_mirror


def full_covariance_rq(x1, x2, params):
    """Build the gradient-enhanced covariance block for the RQ kernel.

    Constructs a ``(D+1, D+1)`` matrix containing energy-energy,
    energy-gradient, gradient-energy, and gradient-gradient covariance entries.

    Args:
        x1: First point, shape ``(D,)``.
        x2: Second point, shape ``(D,)``.
        params: Array of ``[length_scale, alpha]``.

    Returns:
        Covariance block, shape ``(D+1, D+1)``.

    .. versionadded:: 1.1.0
    """
    k_ee = rq_kernel_elem(x1, x2, params)
    k_ed = jax.grad(rq_kernel_elem, argnums=1)(x1, x2, params)
    k_de = jax.grad(rq_kernel_elem, argnums=0)(x1, x2, params)
    k_dd = jax.jacfwd(jax.grad(rq_kernel_elem, argnums=1), argnums=0)(x1, x2, params)
    row1 = jnp.concatenate([k_ee[None], k_ed])
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)
    return jnp.concatenate([row1[None, :], row2], axis=0)


k_matrix_rq_grad_map = vmap(vmap(full_covariance_rq, (None, 0, None)), (0, None, None))
