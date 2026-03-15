import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt
from jax import jit, vmap

from rgpycrumbs.surfaces._base import BaseGradientSurface, generic_negative_mll
from rgpycrumbs.surfaces._kernels import (
    k_matrix_rq_grad_map,
    rq_kernel_elem,
)

# ==============================================================================
# GRADIENT-ENHANCED RQ HELPERS
# ==============================================================================


def negative_mll_rq_map(log_params, x, y_flat, D_plus_1):
    log_ls = log_params[0]
    log_alpha = log_params[1]
    log_noise = log_params[2]
    length_scale = jnp.exp(log_ls)
    alpha = jnp.exp(log_alpha)
    noise_scalar = jnp.exp(log_noise)

    params = jnp.array([length_scale, alpha])
    K_blocks = k_matrix_rq_grad_map(x, x, params)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    mll_cost = generic_negative_mll(K_full, y_flat, noise_scalar)

    ls_target = jnp.log(1.5)
    ls_penalty = (log_ls - ls_target) ** 2 / 0.05
    noise_target = jnp.log(1e-2)
    noise_penalty = (log_noise - noise_target) ** 2 / 1.0
    alpha_target = jnp.log(0.8)
    alpha_penalty = (log_alpha - alpha_target) ** 2 / 0.5
    return mll_cost + ls_penalty + noise_penalty + alpha_penalty


@jit
def _grad_rq_solve(x, y_full, noise_scalar, params):
    K_blocks = k_matrix_rq_grad_map(x, x, params)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_rq_predict(x_query, x_obs, alpha, params):
    def get_query_row(xq, xo):
        kee = rq_kernel_elem(xq, xo, params)
        ked = jax.grad(rq_kernel_elem, argnums=1)(xq, xo, params)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_rq_var(x_query, x_obs, K_inv, params):
    def get_query_row(xq, xo):
        kee = rq_kernel_elem(xq, xo, params)
        ked = jax.grad(rq_kernel_elem, argnums=1)(xq, xo, params)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)

    def self_var(xq):
        return rq_kernel_elem(xq, xq, params)

    base_var = vmap(self_var)(x_query)
    var = base_var - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientRQ(BaseGradientSurface):
    """Symmetric Gradient-enhanced Rational Quadratic (RQ) surface implementation.

    .. versionadded:: 1.0.0
    """

    def _fit(self, _smoothing, length_scale, optimize):
        init_ls = length_scale if length_scale is not None else 1.5
        init_alpha = 1.0
        init_noise = 1e-2

        if optimize:
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_alpha), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_rq_map(log_p, self.x, self.y_flat, self.D_plus_1)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.ls = float(jnp.exp(results.x[0]))
            self.alpha_param = float(jnp.exp(results.x[1]))
            self.noise = float(jnp.exp(results.x[2]))

            if jnp.isnan(self.ls) or jnp.isnan(self.noise):
                self.ls, self.alpha_param, self.noise = init_ls, init_alpha, init_noise
        else:
            self.ls, self.alpha_param, self.noise = init_ls, init_alpha, init_noise
        self.params = jnp.array([self.ls, self.alpha_param])

    def _solve(self):
        self.alpha, self.K_inv = _grad_rq_solve(
            self.x, self.y_full, self.noise, self.params
        )

    def _predict_chunk(self, chunk):
        return _grad_rq_predict(chunk, self.x, self.alpha, self.params)

    def _var_chunk(self, chunk):
        return _grad_rq_var(chunk, self.x, self.K_inv, self.params)
