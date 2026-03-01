"""ASV benchmarks for rgpycrumbs.surfaces (JAX kernel regression)."""

import jax

from rgpycrumbs.surfaces import GradientMatern
from rgpycrumbs.surfaces._kernels import _matern_kernel_matrix


class TimeKernelMatrix:
    """Benchmark raw Matern 5/2 kernel matrix computation."""

    params = [50, 200]
    param_names = ["N"]
    warmup_time = 0

    def setup(self, N):
        key = jax.random.PRNGKey(0)
        self.x = jax.random.normal(key, (N, 3))
        self.length_scale = 1.0
        # JIT warmup
        _matern_kernel_matrix(self.x, self.length_scale).block_until_ready()

    def time_matern_kernel_matrix(self, N):
        _matern_kernel_matrix(self.x, self.length_scale).block_until_ready()


class TimeGradientMaternFitPredict:
    """Benchmark fit + predict cycle for GradientMatern (no optimizer)."""

    warmup_time = 0

    def setup(self):
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        N, D = 30, 3
        self.x_train = jax.random.normal(k1, (N, D))
        self.y_train = jax.random.normal(k2, (N,))
        self.grads_train = jax.random.normal(k3, (N, D))
        self.x_query = jax.random.normal(k4, (20, D))
        # JIT warmup: throwaway fit/predict
        model = GradientMatern(
            self.x_train,
            self.y_train,
            gradients=self.grads_train,
            optimize=False,
        )
        model(self.x_query).block_until_ready()

    def time_fit_predict(self):
        model = GradientMatern(
            self.x_train,
            self.y_train,
            gradients=self.grads_train,
            optimize=False,
        )
        model(self.x_query).block_until_ready()


class TimeGradientMaternPredict:
    """Benchmark prediction only (model fitted in setup)."""

    params = [100, 500]
    param_names = ["n_predict"]
    warmup_time = 0

    def setup(self, n_predict):
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        N, D = 30, 3
        x_train = jax.random.normal(k1, (N, D))
        y_train = jax.random.normal(k2, (N,))
        grads_train = jax.random.normal(k3, (N, D))
        self.model = GradientMatern(
            x_train,
            y_train,
            gradients=grads_train,
            optimize=False,
        )
        self.x_query = jax.random.normal(k4, (n_predict, D))
        # JIT warmup
        self.model(self.x_query).block_until_ready()

    def time_predict(self, n_predict):
        self.model(self.x_query).block_until_ready()
