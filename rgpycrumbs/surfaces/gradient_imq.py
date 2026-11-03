import jax
import jax.numpy as jnp
from jax import jit, vmap

from rgpycrumbs.surfaces._base import BaseGradientSurface, generic_negative_mll
from rgpycrumbs.surfaces._hyperparameters import (
    IMQ_EPSILON_PRIOR_SHAPE,
    covariance_roundoff_jitter,
    median_pairwise_distance,
    minimize_log_parameters,
)
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

    # Gamma(shape, rate) prior on epsilon with its mode (shape - 1) / rate at
    # the starting value.
    rate = (IMQ_EPSILON_PRIOR_SHAPE - 1.0) / init_eps
    eps_penalty = -(IMQ_EPSILON_PRIOR_SHAPE - 1.0) * log_eps + rate * epsilon

    return mll_cost + eps_penalty


@jit
def _grad_imq_solve(x, y_full, noise_scalar, epsilon, noise_per_obs=None):
    K_blocks = k_matrix_imq_grad_map(x, x, epsilon)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    if noise_per_obs is not None:
        # Heteroscedastic: each observation has its own noise level,
        # repeated for the value + D gradient components.
        # Gradient components get 10x less noise than energy (trust gradients).
        D = D_plus_1 - 1
        noise_blocks = []
        for i in range(N):
            obs_noise = noise_per_obs[i]
            block = jnp.concatenate(
                [
                    jnp.array([obs_noise]),  # energy noise
                    jnp.full(D, obs_noise * 0.1),  # gradient noise (10x lower)
                ]
            )
            noise_blocks.append(block)
        noise_diag = jnp.concatenate(noise_blocks)
    else:
        noise_diag = jnp.full(N * D_plus_1, noise_scalar)
    jitter = covariance_roundoff_jitter(K_full)
    K_full = K_full + jnp.diag(noise_diag + jitter)
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

    .. versionchanged:: 1.14.0
       Optimized fits no longer pull the noise toward ``1e-2``.
       The default ``epsilon`` starts at the median distance between inputs.
    """

    def _fit(self, smoothing, length_scale, optimize):
        if length_scale is None:
            init_eps = median_pairwise_distance(self.x)
        else:
            init_eps = length_scale
        init_noise = max(smoothing, 1e-4)

        if optimize:
            x0 = jnp.array([jnp.log(init_eps), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll_imq_map(
                    log_p, init_eps, self.x, self.y_flat, self.D_plus_1
                )

            log_params = minimize_log_parameters(loss_fn, x0)
            self.epsilon = float(jnp.exp(log_params[0]))
            self.noise = float(jnp.exp(log_params[1]))
            # Cap noise at smoothing input -- the MLL tends to overestimate
            # noise for gradient-enhanced IMQ, destroying basin accuracy.
            self.noise = min(self.noise, max(smoothing, 1e-2))
            if jnp.isnan(self.epsilon) or jnp.isnan(self.noise):
                self.epsilon, self.noise = init_eps, init_noise
        else:
            self.epsilon, self.noise = init_eps, init_noise

    def _solve(self):
        self.alpha, self.K_inv = _grad_imq_solve(
            self.x,
            self.y_full,
            self.noise,
            self.epsilon,
            noise_per_obs=getattr(self, "_noise_per_obs", None),
        )

    def _predict_chunk(self, chunk):
        return _grad_imq_predict(chunk, self.x, self.alpha, self.epsilon)

    def _var_chunk(self, chunk):
        return _grad_imq_var(chunk, self.x, self.K_inv, self.epsilon)
