import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt
from jax import jit, vmap

from rgpycrumbs.surfaces._base import BaseGradientSurface, generic_negative_mll
from rgpycrumbs.surfaces._kernels import (
    k_matrix_matern_grad_map,
    matern_kernel_elem,
)

# ==============================================================================
# GRADIENT-ENHANCED MATERN HELPERS
# ==============================================================================


def negative_mll_matern_grad(log_params, x, y_flat, D_plus_1):
    length_scale = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K_blocks = k_matrix_matern_grad_map(x, x, length_scale)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    return generic_negative_mll(K_full, y_flat, noise_scalar)


@jit
def _grad_matern_solve(x, y_full, noise_scalar, length_scale):
    K_blocks = k_matrix_matern_grad_map(x, x, length_scale)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_matern_predict(x_query, x_obs, alpha, length_scale):
    def get_query_row(xq, xo):
        kee = matern_kernel_elem(xq, xo, length_scale)
        ked = jax.grad(matern_kernel_elem, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_matern_var(x_query, x_obs, K_inv, length_scale):
    def get_query_row(xq, xo):
        kee = matern_kernel_elem(xq, xo, length_scale)
        ked = jax.grad(matern_kernel_elem, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)
    var = 1.0 - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientMatern(BaseGradientSurface):
    """Gradient-enhanced Matern 5/2 surface implementation.

    .. versionadded:: 1.0.0
    """

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
            init_ls = jnp.mean(span) * 0.5
        else:
            init_ls = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_matern_grad(log_p, self.x, self.y_flat, self.D_plus_1)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.ls = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))
            if jnp.isnan(self.ls) or jnp.isnan(self.noise):
                self.ls, self.noise = init_ls, init_noise
        else:
            self.ls, self.noise = init_ls, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _grad_matern_solve(
            self.x, self.y_full, self.noise, self.ls
        )

    def _predict_chunk(self, chunk):
        return _grad_matern_predict(chunk, self.x, self.alpha, self.ls)

    def _var_chunk(self, chunk):
        return _grad_matern_var(chunk, self.x, self.K_inv, self.ls)
