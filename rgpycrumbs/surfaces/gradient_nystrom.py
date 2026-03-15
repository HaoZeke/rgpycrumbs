import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np
from jax import jit, vmap

from rgpycrumbs.surfaces._base import BaseGradientSurface
from rgpycrumbs.surfaces._kernels import (
    imq_kernel_elem,
    k_matrix_imq_grad_map,
)

# ==============================================================================
# NYSTROM GRADIENT-ENHANCED IMQ HELPERS
# ==============================================================================


@jit
def _stable_nystrom_grad_imq_solve(x, y_full, x_inducing, noise_scalar, epsilon):
    N = x.shape[0]
    M = x_inducing.shape[0]
    D_plus_1 = x.shape[1] + 1
    K_mm_blocks = k_matrix_imq_grad_map(x_inducing, x_inducing, epsilon)
    K_mm = K_mm_blocks.transpose(0, 2, 1, 3).reshape(M * D_plus_1, M * D_plus_1)
    jitter = (noise_scalar + 1e-4) * jnp.eye(M * D_plus_1)
    K_mm = K_mm + jitter
    L = jnp.linalg.cholesky(K_mm)
    K_nm_blocks = k_matrix_imq_grad_map(x, x_inducing, epsilon)
    K_nm = K_nm_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, M * D_plus_1)
    K_mn = K_nm.T
    V = jlinalg.solve_triangular(L, K_mn, lower=True)
    sigma2 = noise_scalar + 1e-6
    S = V @ V.T + sigma2 * jnp.eye(M * D_plus_1)
    L_S = jnp.linalg.cholesky(S)
    Vy = V @ y_full.flatten()
    beta = jlinalg.cho_solve((L_S, True), Vy)
    alpha_m = jlinalg.solve_triangular(L.T, beta, lower=False)
    I_M = jnp.eye(M * D_plus_1)
    S_inv = jlinalg.cho_solve((L_S, True), I_M)
    inner = I_M - sigma2 * S_inv
    L_inv = jlinalg.solve_triangular(L, I_M, lower=True)
    W = L_inv.T @ inner @ L_inv
    return alpha_m, W


@jit
def _nystrom_grad_imq_predict(x_query, x_inducing, alpha_m, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_qm = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_inducing)
    Q, M, D_plus_1 = K_qm.shape
    return K_qm.reshape(Q, M * D_plus_1) @ alpha_m


@jit
def _nystrom_grad_imq_var(x_query, x_inducing, W, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_qm = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_inducing)
    Q, M, D_plus_1 = K_qm.shape
    K_qm_flat = K_qm.reshape(Q, M * D_plus_1)
    var = (1.0 / epsilon) - jnp.sum((K_qm_flat @ W) * K_qm_flat, axis=1)
    return jnp.maximum(var, 0.0)


class NystromGradientIMQ(BaseGradientSurface):
    """Memory-efficient Nystrom-approximated gradient-enhanced IMQ surface.

    .. versionadded:: 1.1.0
    """

    def __init__(
        self,
        x,
        y,
        gradients=None,
        n_inducing=300,
        nimags=None,
        smoothing=1e-3,
        length_scale=None,
        optimize=True,
        **kwargs,
    ):
        """
        Initializes the Nystrom-approximated model.

        Args:
            x: Training inputs.
            y: Training values.
            gradients: Training gradients.
            n_inducing: Number of inducing points to sample.
            nimags: Path-based image count for structured sampling.
            smoothing: Noise level.
            length_scale: Initial epsilon for IMQ.
            optimize: Optimization toggle.
        """
        self.n_inducing = n_inducing
        self.nimags = nimags
        super().__init__(x, y, gradients, smoothing, length_scale, optimize, **kwargs)

    def _fit(self, smoothing, length_scale, _optimize):
        N_total = self.x.shape[0]
        if N_total <= self.n_inducing:
            self.x_inducing = self.x
            self.y_full_inducing = self.y_full
        else:
            rng = np.random.RandomState(42)
            if self.nimags is not None and self.nimags > 0:
                n_paths = N_total // self.nimags
                paths_to_sample = max(1, self.n_inducing // self.nimags)
                if n_paths > 1:
                    start_idx = max(0, n_paths - paths_to_sample)
                    path_indices = np.arange(start_idx, n_paths)
                else:
                    path_indices = np.array([0])
                idx = np.concatenate(
                    [
                        np.arange(p * self.nimags, (p + 1) * self.nimags)
                        for p in path_indices
                    ]
                )
                idx = idx[idx < N_total]
                self.x_inducing = self.x[idx]
                self.y_full_inducing = self.y_full[idx]
            else:
                idx = rng.choice(N_total, min(self.n_inducing, N_total), replace=False)
                self.x_inducing = self.x[idx]
                self.y_full_inducing = self.y_full[idx]
        self.epsilon = length_scale if length_scale is not None else 0.5
        self.noise = smoothing

    def _solve(self):
        self.alpha_m, self.W = _stable_nystrom_grad_imq_solve(
            self.x, self.y_full, self.x_inducing, self.noise, self.epsilon
        )

    def _predict_chunk(self, chunk):
        return _nystrom_grad_imq_predict(
            chunk, self.x_inducing, self.alpha_m, self.epsilon
        )

    def _var_chunk(self, chunk):
        return _nystrom_grad_imq_var(chunk, self.x_inducing, self.W, self.epsilon)
