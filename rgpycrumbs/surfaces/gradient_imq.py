import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt
from jax import jit, vmap

from rgpycrumbs.surfaces._base import BaseGradientSurface, generic_negative_mll
from rgpycrumbs.surfaces._kernels import (
    imq_kernel_elem,
    k_matrix_imq_grad_map,
)

# ==============================================================================
# GRADIENT-ENHANCED IMQ HELPERS
# ==============================================================================


def negative_mll_imq_grad(log_params, x, y_flat, D_plus_1):
    epsilon = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])
    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    return generic_negative_mll(K_full, y_flat, noise_scalar)


def negative_mll_imq_map(log_params, init_eps, x, y_flat, D_plus_1):
    log_eps = log_params[0]
    log_noise = log_params[1]
    epsilon = jnp.exp(log_eps)
    noise_scalar = jnp.exp(log_noise)

    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    mll_cost = generic_negative_mll(K_full, y_flat, noise_scalar)

    alpha_g = 2.0
    beta_g = 1.0 / (init_eps + 1e-6)
    eps_penalty = -(alpha_g - 1.0) * log_eps + beta_g * epsilon

    noise_target = jnp.log(1e-2)
    noise_penalty = (log_noise - noise_target) ** 2 / 0.5
    return mll_cost + eps_penalty + noise_penalty


@jit
def _grad_imq_solve(x, y_full, noise_scalar, epsilon):
    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise
    K_inv = jnp.linalg.inv(K_full)
    alpha = jnp.linalg.solve(K_full, y_full.flatten())
    return alpha, K_inv


@jit
def _grad_imq_predict(x_query, x_obs, alpha, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    return K_q.reshape(M, N * D_plus_1) @ alpha


@jit
def _grad_imq_var(x_query, x_obs, K_inv, epsilon):
    def get_query_row(xq, xo):
        kee = imq_kernel_elem(xq, xo, epsilon)
        ked = jax.grad(imq_kernel_elem, argnums=1)(xq, xo, epsilon)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)
    var = (1.0 / epsilon) - jnp.sum((K_q_flat @ K_inv) * K_q_flat, axis=1)
    return jnp.maximum(var, 0.0)


class GradientIMQ(BaseGradientSurface):
    """Gradient-enhanced Inverse Multi-Quadratic (IMQ) surface implementation.

    .. versionadded:: 1.0.0
    """

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            span = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
            init_eps = jnp.mean(span) * 0.8
        else:
            init_eps = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_eps), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_imq_map(
                    log_p, init_eps, self.x, self.y_flat, self.D_plus_1
                )

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)
            self.epsilon = float(jnp.exp(results.x[0]))
            self.noise = float(jnp.exp(results.x[1]))
            if jnp.isnan(self.epsilon) or jnp.isnan(self.noise):
                self.epsilon, self.noise = init_eps, init_noise
        else:
            self.epsilon, self.noise = init_eps, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _grad_imq_solve(
            self.x, self.y_full, self.noise, self.epsilon
        )

    def _predict_chunk(self, chunk):
        return _grad_imq_predict(chunk, self.x, self.alpha, self.epsilon)

    def _var_chunk(self, chunk):
        return _grad_imq_var(chunk, self.x, self.K_inv, self.epsilon)
