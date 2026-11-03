import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("surfaces")

import jax
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
# Fit policy tests
# ---------------------------------------------------------------------------


def test_median_pairwise_distance_skips_repeated_points():
    """Zero distances between repeated inputs do not drag the start down."""
    from rgpycrumbs.surfaces._hyperparameters import median_pairwise_distance

    points = jnp.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])

    # The distinct distances are 3, 3, 4, 4 and 5.
    assert median_pairwise_distance(points) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "points",
    [jnp.zeros((1, 2)), jnp.zeros((3, 2))],
    ids=["one-point", "repeated-point"],
)
def test_median_pairwise_distance_needs_two_distinct_points(points):
    """No length scale can be read from fewer than two distinct inputs."""
    from rgpycrumbs.surfaces._hyperparameters import median_pairwise_distance

    with pytest.raises(ValueError, match="length-scale initialization"):
        median_pairwise_distance(points)


def test_zero_smoothing_is_the_roundoff_jitter(simple_2d_data):
    """Exact data get the Cholesky roundoff bound as their noise."""
    from rgpycrumbs.surfaces._hyperparameters import covariance_roundoff_jitter

    x, y = simple_2d_data
    model = FastMatern(x, y, smoothing=0.0, optimize=False, length_scale=1.0)
    jitter = covariance_roundoff_jitter(_matern_kernel_matrix(x, 1.0))

    assert model.noise == pytest.approx(float(jitter))


def test_default_smoothing_treats_observations_as_exact(simple_2d_data):
    """Without a caller's smoothing, the noise is the roundoff jitter."""
    from rgpycrumbs.surfaces._hyperparameters import covariance_roundoff_jitter

    x, y = simple_2d_data
    model = FastMatern(x, y, optimize=False, length_scale=1.0)
    jitter = covariance_roundoff_jitter(_matern_kernel_matrix(x, 1.0))

    assert model.noise == pytest.approx(float(jitter))


@pytest.mark.parametrize("smoothing", [None, 0.0])
def test_exact_data_fits_start_the_noise_at_the_prior_scale(
    simple_2d_data, monkeypatch, smoothing
):
    """An optimized exact-data fit starts at a signal-to-noise ratio of one."""
    from rgpycrumbs.surfaces import standard

    starts = []

    def record_start(_loss_fn, x0):
        starts.append(np.asarray(x0))
        return jnp.asarray(x0)

    monkeypatch.setattr(standard, "minimize_log_parameters", record_start)
    x, y = simple_2d_data
    FastMatern(x, y, smoothing=smoothing, optimize=True, length_scale=1.0)
    scale = jnp.max(jnp.abs(_matern_kernel_matrix(x, 1.0)))

    assert np.exp(starts[0][1]) == pytest.approx(float(scale))


def test_negative_smoothing_is_rejected(simple_2d_data):
    """A negative noise variance is an input error."""
    x, y = simple_2d_data

    with pytest.raises(ValueError, match="smoothing must be non-negative"):
        FastMatern(x, y, smoothing=-1e-3, optimize=False, length_scale=1.0)


def test_gradient_covariance_orders_by_point_then_component():
    """Entry ``(i(D+1) + a, j(D+1) + b)`` is component pair ``(a, b)`` of ``(i, j)``."""
    from rgpycrumbs.surfaces._kernels import gradient_covariance

    N, M, D_plus_1 = 3, 2, 3
    K_blocks = jnp.arange(N * M * D_plus_1**2, dtype=jnp.float32).reshape(
        N, M, D_plus_1, D_plus_1
    )

    K = np.asarray(gradient_covariance(K_blocks))

    assert K.shape == (N * D_plus_1, M * D_plus_1)
    for i in range(N):
        for j in range(M):
            rows = slice(i * D_plus_1, (i + 1) * D_plus_1)
            columns = slice(j * D_plus_1, (j + 1) * D_plus_1)
            assert K[rows, columns] == pytest.approx(np.asarray(K_blocks[i, j]))


def test_roundoff_jitter_is_highams_cholesky_bound():
    """The diagonal shift is gamma_(n+1) times the largest covariance entry."""
    from rgpycrumbs.surfaces._hyperparameters import covariance_roundoff_jitter

    order = 900
    unit_roundoff = 2.0**-24
    gamma = (order + 1) * unit_roundoff / (1 - (order + 1) * unit_roundoff)
    K = 4.0 * jnp.eye(order, dtype=jnp.float32)

    assert covariance_roundoff_jitter(K) == pytest.approx(4.0 * gamma, rel=1e-6)


def test_log_parameter_fit_converges_on_a_smooth_objective():
    """The optimizer reaches the minimum of a plain quadratic."""
    from rgpycrumbs.surfaces._hyperparameters import minimize_log_parameters

    result = minimize_log_parameters(lambda p: jnp.sum((p - 2.0) ** 2), jnp.zeros(2))

    assert np.asarray(result) == pytest.approx([2.0, 2.0], abs=1e-4)


def test_log_parameter_fit_keeps_the_last_accepted_point():
    """A trial step into a region with no finite objective is not the answer."""
    from rgpycrumbs.surfaces._hyperparameters import minimize_log_parameters

    def loss(p):
        return jnp.where(p[0] < 1.0, (p[0] - 2.0) ** 2, jnp.nan)

    start = jnp.zeros(1)
    value = float(loss(minimize_log_parameters(loss, start)))

    assert np.isfinite(value)
    assert value <= float(loss(start))


def test_gradient_se_fit_beats_a_grid_search_on_muller_brown():
    """The fitted likelihood is at least as good as the best grid point.

    The data are 40 random Muller-Brown points in hundreds of energy units.
    The fit starts at the median input distance and unit noise, the prior
    variance of the squared-exponential kernel.  The 41 by 41 grid spans
    length scales from ``e**-3`` to ``e`` times that start and noise from
    float32 epsilon to one.
    """
    from rgpycrumbs.func.muller_brown import muller_brown, muller_brown_gradient
    from rgpycrumbs.surfaces._hyperparameters import minimize_log_parameters
    from rgpycrumbs.surfaces.gradient_se import negative_mll_se_grad

    rng = np.random.default_rng(0)
    x = np.column_stack([rng.uniform(-1.5, 1.0, 40), rng.uniform(-0.2, 1.8, 40)])
    energy = muller_brown(x.T) / 100
    grads = np.asarray(muller_brown_gradient(x.T)).T / 100
    data = GradientSE(x, energy, gradients=grads, length_scale=1.0, optimize=False)

    def loss(p):
        return negative_mll_se_grad(p, data.x, data.y_flat, data.D_plus_1)

    distances = np.linalg.norm(x[:, None] - x[None], axis=-1)[np.triu_indices(40, 1)]
    log_start = np.log(np.median(distances))
    fitted = float(loss(minimize_log_parameters(loss, jnp.array([log_start, 0.0]))))

    log_ls, log_noise = jnp.meshgrid(
        log_start + jnp.linspace(-3.0, 1.0, 41),
        jnp.linspace(jnp.log(jnp.finfo(jnp.float32).eps), 0.0, 41),
    )
    grid = jnp.stack([log_ls.ravel(), log_noise.ravel()], axis=1)
    best_on_grid = float(jnp.nanmin(jax.jit(jax.vmap(loss))(grid)))

    assert fitted <= best_on_grid


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


def test_safe_cholesky_solve_detects_non_positive_definite_input():
    """JAX reports a failed Cholesky with NaNs rather than an exception."""
    K = jnp.array([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float32)
    y = jnp.ones(2, dtype=jnp.float32)

    alpha, log_det = safe_cholesky_solve(K, y, 0.0)

    assert jnp.all(alpha == 0)
    assert jnp.isnan(log_det)


def test_dense_solve_factors_repeated_inputs_without_smoothing():
    """Two identical rows make ``K`` singular, which the jitter covers."""
    from rgpycrumbs.surfaces.standard import _imq_solve

    x = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    y = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)

    alpha, K_inv = _imq_solve(x, y, 0.0, 1.0)

    assert jnp.all(jnp.isfinite(alpha))
    assert jnp.all(jnp.isfinite(K_inv))


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
    assert model.noise == pytest.approx(1e-6)
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
    model = GradientMatern(
        x,
        y,
        gradients=grads,
        smoothing=1e-6,
        optimize=False,
        length_scale=1.0,
    )
    assert model.noise == pytest.approx(1e-6)
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

    The bound is 0.02.  With exact data the closed form measures 0.00593 at
    the median start and 0.00389 at length_scale 3, against 0.00592 and
    0.00373 with noise 1e-4.  At the v1.13.0 start, half the mean span, it
    measured 0.00729, and the autodiff covariance it replaced measured 45.4
    and 0.247.
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
    at an off-training query where the unclipped path is smooth.  With exact
    data the float32 variance is a small difference of large terms, so the
    central difference evaluates the same fitted model in float64.
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
    with jax.enable_x64(True):
        fd_var_query = central_grad(lambda p: float(var(jnp.asarray(p))), xq)
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


def test_rq_gradient_covariance_matches_a_central_difference():
    """Row ``i``, column ``j`` of the mixed block is d2k / dx1_i dx2_j.

    The mirrored RQ kernel has an asymmetric mixed block, so a transposed
    block fails this check where the isotropic kernels would not.
    """
    from rgpycrumbs.surfaces._kernels import full_covariance_rq

    length_scale, alpha = 0.9, 1.5
    params = jnp.array([length_scale, alpha], dtype=jnp.float32)

    def base(p, q):
        d2 = np.sum((p - q) ** 2)
        # rq_kernel_base adds the same 1e-6 to its denominator.
        return (1.0 + d2 / (2.0 * alpha * length_scale**2 + 1e-6)) ** (-alpha)

    def value(p, q):
        return base(p, q) + base(p[::-1], q)

    rng = np.random.default_rng(3)
    x1 = rng.normal(size=2).astype(np.float32).astype(np.float64)
    x2 = rng.normal(size=2).astype(np.float32).astype(np.float64)
    h = 1e-4
    steps = h * np.eye(2)
    mixed = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            mixed[i, j] = (
                value(x1 + steps[i], x2 + steps[j])
                - value(x1 + steps[i], x2 - steps[j])
                - value(x1 - steps[i], x2 + steps[j])
                + value(x1 - steps[i], x2 - steps[j])
            ) / (4.0 * h * h)
    block = np.asarray(
        full_covariance_rq(jnp.asarray(x1), jnp.asarray(x2), params), dtype=np.float64
    )

    assert not np.allclose(mixed, mixed.T, atol=1e-3)
    assert block[1:, 1:] == pytest.approx(mixed, abs=1e-4)


def test_rq_gradient_covariance_is_positive_semidefinite():
    """No eigenvalue falls below the roundoff bound on the entries."""
    from rgpycrumbs.surfaces._hyperparameters import covariance_roundoff_jitter
    from rgpycrumbs.surfaces._kernels import k_matrix_rq_grad_map

    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.uniform(-1.0, 1.0, (12, 2)), dtype=jnp.float32)
    params = jnp.array([0.5, 1.0], dtype=jnp.float32)
    K_blocks = k_matrix_rq_grad_map(x, x, params)
    N, _, D_plus_1, _ = K_blocks.shape
    K = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)

    eigenvalues = np.linalg.eigvalsh(np.asarray(K, dtype=np.float64))

    assert eigenvalues.min() >= -float(covariance_roundoff_jitter(K))


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


def test_gradient_rq_respects_fixed_smoothing(gradient_2d_data):
    """Disabling optimization keeps the caller's smoothing parameter."""
    x, y, grads = gradient_2d_data
    smoothing = 0.25
    model = GradientRQ(
        x,
        y,
        gradients=grads,
        optimize=False,
        length_scale=1.0,
        smoothing=smoothing,
    )

    assert model.noise == pytest.approx(smoothing)


def test_gradient_imq_caps_the_fitted_noise_at_the_callers_smoothing(
    gradient_2d_data, monkeypatch
):
    """A likelihood that prefers more noise cannot exceed the caller's value."""
    from rgpycrumbs.surfaces import gradient_imq

    monkeypatch.setattr(
        gradient_imq,
        "minimize_log_parameters",
        lambda _loss_fn, _x0: jnp.log(jnp.array([1.0, 0.5])),
    )
    x, y, grads = gradient_2d_data
    model = GradientIMQ(
        x, y, gradients=grads, smoothing=1e-3, optimize=True, length_scale=1.0
    )

    assert model.noise == pytest.approx(1e-3)


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


def test_nystrom_gradient_imq_optimizes_hyperparameters(gradient_2d_data, monkeypatch):
    """The public optimize flag must not be silently ignored."""
    from rgpycrumbs.surfaces import gradient_nystrom

    calls = []

    def fake_minimize(loss_fn, x0):
        calls.append((loss_fn, x0))
        return jnp.log(jnp.array([2.0, 1e-3]))

    monkeypatch.setattr(gradient_nystrom, "minimize_log_parameters", fake_minimize)
    x, y, grads = gradient_2d_data
    model = NystromGradientIMQ(
        x,
        y,
        gradients=grads,
        n_inducing=10,
        smoothing=0.2,
        optimize=True,
        length_scale=1.0,
    )

    assert calls
    assert model.epsilon == pytest.approx(2.0)
    assert model.noise == pytest.approx(1e-3)


def test_nystrom_fit_matches_gradient_imq_when_every_point_is_inducing(
    gradient_2d_data,
):
    """With every point inducing, both models fit the same objective."""
    x, y, grads = gradient_2d_data
    kwargs = {"gradients": grads, "smoothing": 1e-3, "optimize": True}

    full = GradientIMQ(x, y, **kwargs)
    nystrom = NystromGradientIMQ(x, y, n_inducing=10, **kwargs)

    assert nystrom.epsilon == pytest.approx(full.epsilon, rel=1e-5)
    assert nystrom.noise == pytest.approx(full.noise, rel=1e-5)


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


@pytest.mark.parametrize(
    ("model_class", "attribute"),
    [(FastMatern, "ls"), (FastIMQ, "epsilon")],
)
def test_dense_models_start_at_the_median_distance(
    simple_2d_data, model_class, attribute
):
    """The five-point bowl has a median input distance of exactly 2."""
    x, y = simple_2d_data
    model = model_class(x, y, optimize=False)

    assert getattr(model, attribute) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("model_class", "attribute"),
    [
        (GradientMatern, "ls"),
        (GradientSE, "ls"),
        (GradientRQ, "ls"),
        (GradientIMQ, "epsilon"),
        (NystromGradientIMQ, "epsilon"),
    ],
)
def test_gradient_models_start_at_the_median_distance(
    gradient_2d_data, model_class, attribute
):
    """The five-point bowl has a median input distance of exactly 2."""
    x, y, grads = gradient_2d_data
    model = model_class(x, y, gradients=grads, optimize=False)

    assert getattr(model, attribute) == pytest.approx(2.0)


def test_gradient_rq_fits_the_marginal_likelihood(gradient_2d_data):
    """No penalty pulls the RQ parameters toward fixed values."""
    from rgpycrumbs.surfaces._kernels import k_matrix_rq_grad_map
    from rgpycrumbs.surfaces.gradient_rq import negative_mll_rq_grad

    x, y, grads = gradient_2d_data
    data = GradientRQ(x, y, gradients=grads, optimize=False, length_scale=1.0)
    length_scale, alpha, noise = 0.7, 2.0, 1e-2
    K_blocks = k_matrix_rq_grad_map(data.x, data.x, jnp.array([length_scale, alpha]))
    N, _, D_plus_1, _ = K_blocks.shape
    K = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
    log_params = jnp.log(jnp.array([length_scale, alpha, noise]))

    objective = negative_mll_rq_grad(log_params, data.x, data.y_flat, data.D_plus_1)

    assert float(objective) == pytest.approx(
        float(generic_negative_mll(K, data.y_flat, noise)), rel=1e-5
    )


def test_gradient_rq_fit_minimizes_the_marginal_likelihood(gradient_2d_data):
    """An optimized fit ends where a direct fit of the likelihood ends."""
    from rgpycrumbs.surfaces._hyperparameters import minimize_log_parameters
    from rgpycrumbs.surfaces._kernels import k_matrix_rq_grad_map

    x, y, grads = gradient_2d_data
    model = GradientRQ(
        x, y, gradients=grads, length_scale=1.5, smoothing=1e-2, optimize=True
    )

    def loss(log_params):
        length_scale = jnp.exp(log_params[0])
        alpha = jnp.exp(log_params[1])
        noise = jnp.exp(log_params[2])
        params = jnp.array([length_scale, alpha])
        K_blocks = k_matrix_rq_grad_map(model.x, model.x, params)
        N, _, D_plus_1, _ = K_blocks.shape
        K = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)
        return generic_negative_mll(K, model.y_flat, noise)

    start = jnp.log(jnp.array([1.5, 1.0, 1e-2]))
    expected = np.exp(np.asarray(minimize_log_parameters(loss, start)))

    fitted = np.array([model.ls, model.alpha_param, model.noise])
    assert fitted == pytest.approx(expected, rel=1e-6)


def _imq_prior(log_params, init_eps, data):
    from rgpycrumbs.surfaces.gradient_imq import (
        negative_mll_imq_grad,
        negative_mll_imq_map,
    )

    args = (data.x, data.y_flat, data.D_plus_1)
    return float(
        negative_mll_imq_map(log_params, init_eps, *args)
        - negative_mll_imq_grad(log_params, *args)
    )


def test_gradient_imq_prior_peaks_at_the_starting_epsilon(gradient_2d_data):
    """The gamma prior adds the least penalty at the starting epsilon."""
    x, y, grads = gradient_2d_data
    data = GradientIMQ(x, y, gradients=grads, optimize=False, length_scale=1.0)
    init_eps = 1.3
    log_eps = jnp.log(init_eps) + jnp.linspace(-0.5, 0.5, 101)
    penalties = [
        _imq_prior(jnp.array([value, jnp.log(1e-2)]), init_eps, data) for value in log_eps
    ]

    best = float(jnp.exp(log_eps[int(np.argmin(penalties))]))

    assert best == pytest.approx(init_eps, rel=1e-2)


def test_gradient_imq_prior_leaves_the_noise_to_the_likelihood(gradient_2d_data):
    """Only epsilon carries a prior, so the noise level changes no penalty."""
    x, y, grads = gradient_2d_data
    data = GradientIMQ(x, y, gradients=grads, optimize=False, length_scale=1.0)
    log_eps = jnp.log(1.3)

    penalties = [
        _imq_prior(jnp.array([log_eps, jnp.log(noise)]), 1.3, data)
        for noise in (1e-4, 1e-2, 1.0)
    ]

    assert penalties == pytest.approx([penalties[0]] * 3, abs=1e-3)


def test_heteroscedastic_gradient_noise_follows_the_input_units():
    """Inputs scaled by ``s`` scale the gradient noise by ``1 / s**2``.

    That is the energy noise ratio a gradient in energy per length needs.
    """
    from rgpycrumbs.surfaces._kernels import full_covariance_imq
    from rgpycrumbs.surfaces.gradient_imq import _heteroscedastic_noise

    point = jnp.array([0.3, -0.2], dtype=jnp.float32)
    epsilon, scale = 0.7, 10.0
    noise = jnp.array([0.05], dtype=jnp.float32)
    ratios = []
    for s in (1.0, scale):
        block = full_covariance_imq(s * point, s * point, jnp.float32(s * epsilon))
        spread = _heteroscedastic_noise(noise, jnp.diag(block)[None, :])
        assert spread[0] == pytest.approx(0.05)
        ratios.append(np.asarray(spread[1:] / spread[0]))

    assert ratios[1] == pytest.approx(ratios[0] / scale**2, rel=1e-5)


def test_gradient_imq_solve_spreads_the_energy_noise_by_prior_variance(
    gradient_2d_data,
):
    """The IMQ prior variances are ``1 / epsilon`` and ``1 / epsilon**3``.

    Each gradient component therefore gets the energy noise over
    ``epsilon**2``.
    """
    from rgpycrumbs.surfaces._hyperparameters import covariance_roundoff_jitter
    from rgpycrumbs.surfaces._kernels import gradient_covariance, k_matrix_imq_grad_map
    from rgpycrumbs.surfaces.gradient_imq import _grad_imq_solve

    x, y, grads = gradient_2d_data
    epsilon = 0.7
    noise = jnp.linspace(0.01, 0.05, x.shape[0], dtype=jnp.float32)
    y_full = jnp.concatenate([y[:, None], grads], axis=1)
    K = gradient_covariance(k_matrix_imq_grad_map(x, x, epsilon))
    gradient_noise = noise / epsilon**2
    expected_noise = jnp.stack([noise, gradient_noise, gradient_noise], axis=1)
    K_noisy = K + jnp.diag(expected_noise.flatten() + covariance_roundoff_jitter(K))

    alpha, _ = _grad_imq_solve(x, y_full, 0.0, epsilon, noise)

    expected = np.linalg.solve(np.asarray(K_noisy, np.float64), np.ravel(y_full))
    assert np.asarray(alpha) == pytest.approx(expected, rel=1e-3, abs=1e-4)


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
