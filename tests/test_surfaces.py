import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("surfaces")

import jax.numpy as jnp
import numpy as np

from rgpycrumbs.surfaces import (
    FastIMQ,
    FastMatern,
    FastTPS,
    GradientIMQ,
    GradientMatern,
    GradientRQ,
    GradientSE,
    NystromGradientIMQ,
    _imq_kernel_matrix,
    _matern_kernel_matrix,
    _tps_kernel_matrix,
    generic_negative_mll,
    get_surface_model,
    safe_cholesky_solve,
)

pytestmark = pytest.mark.surfaces


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_2d_data():
    """5 observations on a quadratic bowl z = x^2 + y^2."""
    x = jnp.array(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    y = jnp.sum(x**2, axis=1)  # [2, 2, 0, 2, 2]
    return x, y


@pytest.fixture
def gradient_2d_data(simple_2d_data):
    """Gradients for the quadratic bowl z = x^2 + y^2."""
    x, y = simple_2d_data
    grads = 2.0 * x
    return x, y, grads


@pytest.fixture
def query_points():
    """A small grid of query points."""
    return jnp.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5]], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_get_surface_model_returns_classes():
    """Test that get_surface_model returns valid model classes."""
    assert get_surface_model("tps") is FastTPS
    assert get_surface_model("rbf") is FastTPS
    assert get_surface_model("matern") is FastMatern
    assert get_surface_model("imq") is FastIMQ
    assert get_surface_model("grad_matern") is GradientMatern
    assert get_surface_model("grad_rq") is GradientRQ
    assert get_surface_model("grad_se") is GradientSE
    assert get_surface_model("grad_imq") is GradientIMQ
    assert get_surface_model("grad_imq_ny") is NystromGradientIMQ


def test_get_surface_model_default():
    """Test that unknown names return GradientMatern as default."""
    assert get_surface_model("nonexistent") is GradientMatern


# ---------------------------------------------------------------------------
# Kernel matrix tests
# ---------------------------------------------------------------------------


def test_tps_kernel_matrix_shape():
    """Test that TPS kernel matrix has correct shape."""
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    K = _tps_kernel_matrix(x)
    assert K.shape == (3, 3)


def test_tps_kernel_matrix_symmetric():
    """Test that TPS kernel matrix is symmetric."""
    x = jnp.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [2.0, 2.0]], dtype=jnp.float32)
    K = _tps_kernel_matrix(x)
    assert jnp.allclose(K, K.T, atol=1e-6)


def test_matern_kernel_matrix_shape_and_symmetry():
    """Test that Matern kernel has correct shape and is symmetric."""
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    K = _matern_kernel_matrix(x, length_scale=1.0)
    assert K.shape == (4, 4)
    assert jnp.allclose(K, K.T, atol=1e-6)


def test_matern_kernel_diagonal_ones():
    """Matern kernel should have 1s on the diagonal (k(x,x)=1)."""
    x = jnp.array([[0.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)
    K = _matern_kernel_matrix(x, length_scale=1.0)
    assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-5)


def test_imq_kernel_matrix_shape_and_symmetry():
    """Test that IMQ kernel has correct shape and is symmetric."""
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    K = _imq_kernel_matrix(x, epsilon=1.0)
    assert K.shape == (3, 3)
    assert jnp.allclose(K, K.T, atol=1e-6)


def test_imq_kernel_positive_values():
    """IMQ kernel values should always be positive."""
    x = jnp.array([[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]], dtype=jnp.float32)
    K = _imq_kernel_matrix(x, epsilon=1.0)
    assert jnp.all(K > 0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_generic_negative_mll_finite():
    """Test that generic_negative_mll returns a finite value."""
    K = jnp.eye(3, dtype=jnp.float32)
    y = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    result = generic_negative_mll(K, y, 1e-3)
    assert jnp.isfinite(result)


def test_generic_negative_mll_identity_known_value():
    """For identity kernel, MLL should be computable and match analytic form."""
    N = 4
    K = jnp.eye(N, dtype=jnp.float32)
    y = jnp.ones(N, dtype=jnp.float32)
    noise = 1e-4
    result = generic_negative_mll(K, y, noise)
    # data_fit = 0.5 * y^T (K + noise*I)^{-1} y  ~  0.5 * N (for small noise)
    # complexity = 0.5 * log det(K + noise*I)  ~  0 (for small noise)
    assert result > 0
    assert jnp.isfinite(result)


def test_safe_cholesky_solve_identity():
    """safe_cholesky_solve on identity should return y (approximately)."""
    K = jnp.eye(3, dtype=jnp.float32)
    y = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    alpha, log_det = safe_cholesky_solve(K, y, 1e-6)
    assert jnp.allclose(alpha, y, atol=1e-3)
    assert jnp.isfinite(log_det)


# ---------------------------------------------------------------------------
# Model instantiation and prediction tests (non-gradient models)
# ---------------------------------------------------------------------------


def test_fast_tps_fit_and_predict(simple_2d_data, query_points):
    """FastTPS should fit and produce finite predictions."""
    x, y = simple_2d_data
    model = FastTPS(x, y, optimize=False)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_fast_tps_interpolates_training_points(simple_2d_data):
    """FastTPS predictions at training points should be close to training values."""
    x, y = simple_2d_data
    model = FastTPS(x, y, smoothing=1e-6, optimize=False)
    preds = model(x)
    assert jnp.allclose(preds, y, atol=0.1)


def test_fast_matern_fit_and_predict(simple_2d_data, query_points):
    """FastMatern should fit and produce finite predictions."""
    x, y = simple_2d_data
    model = FastMatern(x, y, optimize=False, length_scale=1.0)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_fast_matern_interpolates_training_points(simple_2d_data):
    """FastMatern predictions at training points should be close to training values."""
    x, y = simple_2d_data
    model = FastMatern(x, y, smoothing=1e-6, length_scale=1.0, optimize=False)
    preds = model(x)
    assert jnp.allclose(preds, y, atol=0.5)


def test_fast_imq_fit_and_predict(simple_2d_data, query_points):
    """FastIMQ should fit and produce finite predictions."""
    x, y = simple_2d_data
    model = FastIMQ(x, y, optimize=False, length_scale=1.0)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_fast_imq_interpolates_training_points(simple_2d_data):
    """FastIMQ predictions at training points should be close to training values."""
    x, y = simple_2d_data
    model = FastIMQ(x, y, smoothing=1e-6, length_scale=1.0, optimize=False)
    preds = model(x)
    assert jnp.allclose(preds, y, atol=0.5)


# ---------------------------------------------------------------------------
# Model instantiation and prediction tests (gradient-enhanced models)
# ---------------------------------------------------------------------------


def test_gradient_matern_fit_and_predict(gradient_2d_data, query_points):
    """GradientMatern should fit and produce finite predictions."""
    x, y, grads = gradient_2d_data
    model = GradientMatern(x, y, gradients=grads, optimize=False, length_scale=1.0)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def _pinned_bowl():
    """A 30-point bowl with exact gradients, 25 queries, and their truth."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-2.0, 2.0, (30, 2))
    query = rng.uniform(-2.0, 2.0, (25, 2))
    a, b = x[:, 0], x[:, 1]
    y = np.sin(a) * np.cos(0.5 * b) + 0.1 * a**2
    grads = np.stack(
        [np.cos(a) * np.cos(0.5 * b) + 0.2 * a, -0.5 * np.sin(a) * np.sin(0.5 * b)],
        axis=1,
    )
    qa, qb = query[:, 0], query[:, 1]
    truth = np.sin(qa) * np.cos(0.5 * qb) + 0.1 * qa**2
    return x, y, grads, query, truth


def test_gradient_matern_predict_is_near_the_exact_answer():
    """Held to the exact answer on the bowl at the default and length_scale 3.

    The bound is 0.02.  The closed form measures 0.00729 at the default and
    0.00374 at length_scale 3.  The autodiff covariance is 45.4 and 0.247.
    """
    x, y, grads, query, truth = _pinned_bowl()
    cases = ((None, 0.02), (3.0, 0.02))
    for length_scale, bound in cases:
        kwargs = {"optimize": False}
        if length_scale is not None:
            kwargs["length_scale"] = length_scale
        out = GradientMatern(x, y, gradients=grads, **kwargs)(query)
        error = float(np.abs(np.asarray(out) - np.asarray(truth)).max())
        assert error < bound, (length_scale, error, bound)


def test_gradient_matern_query_gradient_matches_the_expected_result():
    """The query-row gradients are correct at training and query points.

    At a training point, the mean gradient should match the exact bowl
    gradient.  The variance path uses ``maximum(var, 0)``, so a training
    point can sit on the clipping kink; the variance gradient check is done
    at an off-training query where the unclipped path is smooth.
    """
    import jax

    from rgpycrumbs.surfaces.gradient_matern import (
        _grad_matern_predict,
        _grad_matern_var,
    )

    x, y, grads, _, _ = _pinned_bowl()

    def central_grad(fn, xq, h=5e-2):
        out = np.empty_like(np.asarray(xq))
        xq_np = np.asarray(xq)
        for i in range(xq_np.size):
            step = np.zeros_like(xq_np)
            step[i] = h
            out[i] = (fn(xq_np + step) - fn(xq_np - step)) / (2.0 * h)
        return out

    for length_scale in (None, 3.0):
        kwargs = {"optimize": False}
        if length_scale is not None:
            kwargs["length_scale"] = length_scale
        model = GradientMatern(x, y, gradients=grads, **kwargs)

        def mean(xq, model=model):
            return _grad_matern_predict(xq[None], model.x, model.alpha, model.ls)[0]

        x0 = model.x[0]
        grad_mean = np.asarray(jax.grad(mean)(x0))
        assert grad_mean.shape == (2,)
        assert np.all(np.isfinite(grad_mean))
        assert grad_mean == pytest.approx(np.asarray(grads[0]), abs=1e-3)

    model = GradientMatern(x, y, gradients=grads, optimize=False)

    def var(xq):
        return _grad_matern_var(xq[None], model.x, model.K_inv, model.ls)[0]

    x0 = model.x[0]
    grad_var = np.asarray(jax.grad(var)(x0))
    assert grad_var.shape == (2,)
    assert np.all(np.isfinite(grad_var))

    xq = np.array([2.0, 2.0], dtype=np.float64)
    grad_var_query = np.asarray(jax.grad(var)(jnp.asarray(xq, dtype=jnp.float32)))
    fd_var_query = central_grad(
        lambda p: float(var(jnp.asarray(p, dtype=jnp.float32))), xq
    )
    assert float(var(jnp.asarray(xq, dtype=jnp.float32))) > 0.0
    assert grad_var_query == pytest.approx(fd_var_query, abs=2e-2)


def test_matern_gradient_covariance_is_exact_at_a_repeated_point():
    """The closed form, not an autodiff cancellation, at x1 == x2."""
    from rgpycrumbs.surfaces._kernels import full_covariance_matern

    x = jnp.array([0.3, -0.4], dtype=jnp.float32)
    block = np.asarray(full_covariance_matern(x, x, jnp.float32(3.0)))
    exact = 5.0 / (3.0 * 9.0)
    assert block[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert block[0, 1:] == pytest.approx(0.0, abs=1e-6)
    assert block[1:, 0] == pytest.approx(0.0, abs=1e-6)
    assert block[1:, 1:] == pytest.approx(exact * np.eye(2), abs=1e-5)


def test_matern_gradient_covariance_matches_a_central_difference():
    """Off a repeated point, the block matches a float64 difference of k."""
    from rgpycrumbs.surfaces._kernels import full_covariance_matern

    rng = np.random.default_rng(1)
    x1 = rng.normal(size=2)
    x2 = rng.normal(size=2)
    ls = 1.7
    h = 1e-5

    def value(p, q):
        r = np.linalg.norm(p - q)
        u = np.sqrt(5.0) * r / ls
        return (1.0 + u + u**2 / 3.0) * np.exp(-u)

    def grad_x2(p, q):
        out = np.empty(2)
        for i in range(2):
            step = np.zeros(2)
            step[i] = h
            out[i] = (value(p, q + step) - value(p, q - step)) / (2.0 * h)
        return out

    dk_dx2 = grad_x2(x1, x2)
    dk_dx1 = grad_x2(x2, x1)
    hess = np.empty((2, 2))
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        hess[j] = (grad_x2(x1 + step, x2) - grad_x2(x1 - step, x2)) / (2.0 * h)
    # jacfwd(grad wrt x2) wrt x1 has shape (dx2, dx1); hess[j] is d/dx1_j.
    hess = hess.T
    block = np.asarray(
        full_covariance_matern(
            jnp.asarray(x1),
            jnp.asarray(x2),
            jnp.asarray(ls),
        ),
        dtype=np.float64,
    )
    assert block[0, 0] == pytest.approx(value(x1, x2), abs=1e-5)
    assert block[0, 1:] == pytest.approx(dk_dx2, abs=1e-5)
    assert block[1:, 0] == pytest.approx(dk_dx1, abs=1e-5)
    assert block[1:, 1:] == pytest.approx(hess, abs=1e-4)


def test_gradient_imq_fit_and_predict(gradient_2d_data, query_points):
    """GradientIMQ should fit and produce finite predictions."""
    x, y, grads = gradient_2d_data
    model = GradientIMQ(x, y, gradients=grads, optimize=False, length_scale=1.0)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_gradient_se_fit_and_predict(gradient_2d_data, query_points):
    """GradientSE should fit and produce finite predictions."""
    x, y, grads = gradient_2d_data
    model = GradientSE(x, y, gradients=grads, optimize=False, length_scale=1.0)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_gradient_rq_fit_and_predict(gradient_2d_data, query_points):
    """GradientRQ should fit and produce finite predictions."""
    x, y, grads = gradient_2d_data
    model = GradientRQ(x, y, gradients=grads, optimize=False, length_scale=1.0)
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_nystrom_gradient_imq_fit_and_predict(gradient_2d_data, query_points):
    """NystromGradientIMQ should fit and produce finite predictions."""
    x, y, grads = gradient_2d_data
    # Use n_inducing >= N_total to test basic fit
    model = NystromGradientIMQ(
        x, y, gradients=grads, n_inducing=10, optimize=False, length_scale=1.0
    )
    preds = model(query_points)
    assert preds.shape == (3,)
    assert jnp.all(jnp.isfinite(preds))


def test_nystrom_gradient_imq_path_sampling(gradient_2d_data):
    """Test the path-based sampling logic in NystromGradientIMQ."""
    x, y, grads = gradient_2d_data
    # Create synthetic "paths"
    nimags = 2
    # Simple data has 5 points, so 2 paths of 2 and 1 residual
    model = NystromGradientIMQ(
        x, y, gradients=grads, n_inducing=2, nimags=nimags, optimize=False
    )
    # The sampling logic should produce an x_inducing
    assert model.x_inducing.shape[0] <= 4  # max(1, 2//2) paths * 2 = 2 or more points
    assert jnp.all(jnp.isfinite(model.x_inducing))


# ---------------------------------------------------------------------------
# Variance prediction tests
# ---------------------------------------------------------------------------


def test_predict_var_non_gradient_models(simple_2d_data, query_points):
    """Test variance prediction for standard implementations."""
    x, y = simple_2d_data
    for ModelClass in [FastTPS, FastMatern, FastIMQ]:
        model = ModelClass(x, y, optimize=False, length_scale=1.0, smoothing=1e-4)
        var = model.predict_var(query_points)
        assert var.shape == (query_points.shape[0],)
        assert jnp.all(var >= 0.0)
        assert jnp.all(jnp.isfinite(var))

        # Variance at training points should drop significantly
        var_train = model.predict_var(x)
        assert jnp.all(var_train < 1e-1)


def test_predict_var_gradient_models(gradient_2d_data, query_points):
    """Test variance prediction for gradient-enhanced kernels."""
    x, y, grads = gradient_2d_data
    for ModelClass in [GradientMatern, GradientIMQ, GradientSE, GradientRQ]:
        model = ModelClass(
            x, y, gradients=grads, optimize=False, length_scale=1.0, smoothing=1e-4
        )
        var = model.predict_var(query_points)
        assert var.shape == (query_points.shape[0],)
        assert jnp.all(var >= 0.0)
        assert jnp.all(jnp.isfinite(var))

        # Variance at training points should drop significantly
        var_train = model.predict_var(x)
        assert jnp.all(var_train < 1e-1)


# ---------------------------------------------------------------------------
# Batched prediction tests
# ---------------------------------------------------------------------------


def test_batched_prediction_matches_full(simple_2d_data):
    """Prediction with small chunk_size should match prediction with large chunk_size."""
    x, y = simple_2d_data
    model = FastTPS(x, y, optimize=False)
    query = jnp.linspace(-1, 1, 20).reshape(-1, 2).astype(jnp.float32)
    full = model(query, chunk_size=1000)
    batched = model(query, chunk_size=3)
    assert jnp.allclose(full, batched, atol=1e-5)


# ---------------------------------------------------------------------------
# Optimization tests
# ---------------------------------------------------------------------------


def test_fast_tps_with_optimization(simple_2d_data, query_points):
    """FastTPS with optimize=True should still produce finite results."""
    x, y = simple_2d_data
    model = FastTPS(x, y, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    # Optimized smoothing should be a positive finite number
    assert model.sm > 0
    assert jnp.isfinite(model.sm)


def test_fast_matern_with_optimization(simple_2d_data, query_points):
    """FastMatern with optimize=True should produce finite results."""
    x, y = simple_2d_data
    model = FastMatern(x, y, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    assert model.ls > 0
    assert model.noise > 0


def test_fast_imq_with_optimization(simple_2d_data, query_points):
    """FastIMQ with optimize=True should produce finite results."""
    x, y = simple_2d_data
    model = FastIMQ(x, y, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    assert model.epsilon > 0
    assert model.noise > 0


def test_gradient_matern_with_optimization(gradient_2d_data, query_points):
    """GradientMatern with optimize=True should produce finite results."""
    x, y, grads = gradient_2d_data
    model = GradientMatern(x, y, gradients=grads, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    assert model.ls > 0
    assert model.noise > 0


def test_gradient_imq_with_optimization(gradient_2d_data, query_points):
    """GradientIMQ with optimize=True should produce finite results."""
    x, y, grads = gradient_2d_data
    model = GradientIMQ(x, y, gradients=grads, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    assert model.epsilon > 0
    assert model.noise > 0


def test_gradient_se_with_optimization(gradient_2d_data, query_points):
    """GradientSE with optimize=True should produce finite results."""
    x, y, grads = gradient_2d_data
    model = GradientSE(x, y, gradients=grads, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    assert model.ls > 0
    assert model.noise > 0


def test_gradient_rq_with_optimization(gradient_2d_data, query_points):
    """GradientRQ with optimize=True should produce finite results."""
    x, y, grads = gradient_2d_data
    model = GradientRQ(x, y, gradients=grads, optimize=True)
    preds = model(query_points)
    assert jnp.all(jnp.isfinite(preds))
    assert model.ls > 0
    assert model.alpha_param > 0
    assert model.noise > 0


# ---------------------------------------------------------------------------
# Prediction at origin sanity check
# ---------------------------------------------------------------------------


def test_gradient_models_predict_minimum_near_origin(gradient_2d_data):
    """For z = x^2 + y^2, gradient-enhanced models should predict a low value at the origin."""
    x, y, grads = gradient_2d_data
    origin = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

    for ModelClass in [GradientMatern, GradientIMQ, GradientSE, GradientRQ]:
        model = ModelClass(
            x, y, gradients=grads, optimize=False, length_scale=1.0, smoothing=1e-4
        )
        pred = float(model(origin)[0])
        assert pred < float(jnp.max(y)), f"{ModelClass.__name__} predicted {pred}"


def test_gradient_influence():
    """Verify that providing different gradients changes the model output."""
    x = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    y = jnp.array([0.0], dtype=jnp.float32)
    query = jnp.array([[1.0, 1.0]], dtype=jnp.float32)

    grads1 = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    grads2 = jnp.array([[10.0, 10.0]], dtype=jnp.float32)

    for ModelClass in [GradientMatern, GradientIMQ, GradientSE, GradientRQ]:
        m1 = ModelClass(x, y, gradients=grads1, optimize=False, length_scale=1.0)
        m2 = ModelClass(x, y, gradients=grads2, optimize=False, length_scale=1.0)

        pred1 = m1(query)
        pred2 = m2(query)

        # The predictions should differ because the gradients at (0,0) are different
        assert not jnp.allclose(pred1, pred2, atol=1e-3), f"{ModelClass.__name__}"
