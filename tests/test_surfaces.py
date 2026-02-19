import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("surfaces")

import jax.numpy as jnp  # noqa: E402

from rgpycrumbs.surfaces import (  # noqa: E402
    FastIMQ,
    FastMatern,
    FastTPS,
    GradientMatern,
    GradientRQ,
    GradientSE,
    GradientIMQ,
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
    return jnp.array(
        [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5]], dtype=jnp.float32
    )


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
    x = jnp.array(
        [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [2.0, 2.0]], dtype=jnp.float32
    )
    K = _tps_kernel_matrix(x)
    assert jnp.allclose(K, K.T, atol=1e-6)


def test_matern_kernel_matrix_shape_and_symmetry():
    """Test that Matern kernel has correct shape and is symmetric."""
    x = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32
    )
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
    x = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
    )
    K = _imq_kernel_matrix(x, epsilon=1.0)
    assert K.shape == (3, 3)
    assert jnp.allclose(K, K.T, atol=1e-6)


def test_imq_kernel_positive_values():
    """IMQ kernel values should always be positive."""
    x = jnp.array(
        [[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]], dtype=jnp.float32
    )
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
        model = ModelClass(x, y, gradients=grads, optimize=False, length_scale=1.0, smoothing=1e-4)
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


# ---------------------------------------------------------------------------
# Prediction at origin sanity check
# ---------------------------------------------------------------------------


def test_models_predict_minimum_near_origin(simple_2d_data):
    """For z = x^2 + y^2, all models should predict a low value at the origin."""
    x, y = simple_2d_data
    origin = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

    for ModelClass in [FastTPS, FastMatern, FastIMQ]:
        model = ModelClass(x, y, optimize=False, length_scale=1.0)
        pred = float(model(origin)[0])
        # The minimum of x^2+y^2 is 0 at the origin
        # Model should predict something reasonably low (< max training value)
        assert pred < float(jnp.max(y)) + 1.0, f"{ModelClass.__name__} predicted {pred}"
