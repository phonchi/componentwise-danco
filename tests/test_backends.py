"""T2 -- the GPU backend agrees with the CPU one.

The point of the layer is that ``backend`` is a *performance* choice, not a
numerical one.  So every test here fixes the input and varies only the backend:
neighbour distances to ``rtol=1e-6`` (the GPU path is float32 by default),
angles to ``atol=1e-4``, and -- the one that actually matters for a table --
identical integer estimates with the fractal refinement agreeing to 0.05.

Data is deliberately mid-sized (``N=500``, ``D=2000``): small enough to keep the
suite quick, large enough that the chunked kernels take more than one pass and
the ``float32`` error has room to show up.

Everything is skipped without CUDA.  The backend refuses to run on the CPU by
design, and a green tick from a suite that silently tested nothing would be
worse than a skip.
"""

import numpy as np
import pytest

from danco import DANCo
from danco._kernels.angles import indnComb
from danco.backends import get_backend
from danco.calibration import hyper_ball

K = 10
CUDA = False
try:  # torch may not be installed at all
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - depends on the install
    torch = None

requires_cuda = pytest.mark.skipif(
    not CUDA, reason="no CUDA device; the gpu backend raises rather than falling back"
)


def embedded_ball(n=500, d=12, D=2000, seed=0):
    """``d``-dimensional ball randomly rotated into ``D`` dimensions."""
    rng = np.random.RandomState(seed)
    X = hyper_ball(rng, n, d, center=[0] * d)
    X = np.hstack([X, np.zeros((n, D - d))])
    rotation = np.linalg.qr(rng.normal(size=(D, D)))[0]
    return X @ rotation


@pytest.fixture(scope="module")
def data():
    return embedded_ball()


@pytest.fixture(scope="module")
def cpu():
    return get_backend("cpu")


# ----------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------
@requires_cuda
def test_knn_matches_cpu(data, cpu):
    gpu = get_backend("gpu")
    d_cpu, i_cpu = cpu.knn(data, K)
    d_gpu, i_gpu = gpu.knn(data, K)

    assert d_gpu.shape == d_cpu.shape == (len(data), K + 1)
    assert np.array_equal(i_gpu, i_cpu), "neighbour ordering differs"
    np.testing.assert_allclose(d_gpu, d_cpu, rtol=1e-6)


@requires_cuda
def test_angle_stats_matches_cpu(data, cpu):
    gpu = get_backend("gpu")
    combs = indnComb(K)
    _, idx = cpu.knn(data, K)

    a_cpu = cpu.angle_stats(data, idx, combs, K)
    a_gpu = gpu.angle_stats(data, idx, combs, K)

    assert a_gpu.shape == a_cpu.shape == (len(data), K * (K - 1) // 2)
    np.testing.assert_allclose(a_gpu, a_cpu, atol=1e-4)


@requires_cuda
def test_float64_is_exact(data, cpu):
    """``dtype='float64'`` is not an approximation of the CPU path at all."""
    gpu = get_backend("gpu", dtype="float64")
    d_cpu, i_cpu = cpu.knn(data, K)
    d_gpu, i_gpu = gpu.knn(data, K)
    assert np.array_equal(i_gpu, i_cpu)
    np.testing.assert_allclose(d_gpu, d_cpu, rtol=1e-12)


@requires_cuda
def test_small_batches_change_nothing(data):
    """Chunking is an implementation detail; the batch sizes must not leak.

    Not bit-identical, and cannot be: cuBLAS picks a different reduction for a
    ``37 x n`` matmul than for a ``1024 x n`` one, so the last float32 bit
    moves.  What must not move is the neighbour *set* -- an OOM retry halves the
    batch mid-run, and a run whose answer depended on how much VRAM happened to
    be free would not be reproducible.
    """
    big = get_backend("gpu", query_batch=1024, angle_batch=64)
    small = get_backend("gpu", query_batch=37, angle_batch=3)
    d_big, i_big = big.knn(data, K)
    d_small, i_small = small.knn(data, K)
    assert np.array_equal(i_big, i_small)
    np.testing.assert_allclose(d_big, d_small, rtol=1e-5)

    combs = indnComb(K)
    np.testing.assert_allclose(
        big.angle_stats(data, i_big, combs, K),
        small.angle_stats(data, i_big, combs, K),
        atol=1e-4,
    )


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------
@requires_cuda
@pytest.mark.parametrize("version", ["DANCo_fast", "DANCo_nustrip_fast",
                                     "DANCo_gride_nustrip_fast"])
def test_evaluate_matches_cpu(data, version):
    """Same estimate from both backends -- integer exactly, fractal to 0.05.

    The ``_fast`` versions keep the runtime down (spline calibration, no
    Monte-Carlo loop) and are the configuration the CNN experiments run; the
    Profiled versions exercise the profiled angular curve, which reads
    ``mu_tau`` and so is the one sensitive to the angle kernel.
    """
    spline_kwargs = dict(N=500, k=K, random_state=35, version=version)

    for fractal in (False, True):
        de_cpu, s_cpu = DANCo(fractal=fractal, backend="cpu", **spline_kwargs
                              ).evaluate(data, maxDim=100)
        de_gpu, s_gpu = DANCo(fractal=fractal, backend="gpu", **spline_kwargs
                              ).evaluate(data, maxDim=100)
        key = "estimate"
        if not fractal:
            assert int(s_gpu[key]) == int(s_cpu[key]) == int(de_gpu)
        else:
            assert abs(s_gpu[key] - s_cpu[key]) < 0.05
        assert abs(s_gpu["dhat"] - s_cpu["dhat"]) < 1e-3
        assert abs(s_gpu["mu_tau"] - s_cpu["mu_tau"]) < 1e-4


@requires_cuda
def test_monte_carlo_calibration_stays_on_cpu(data):
    """The device must not touch the calibration, bit for bit.

    Same seed, same hyperballs, same backend for them -- so the two caches have
    to be identical even though one estimator is nominally a GPU one.  If this
    ever fails, the GPU has started contributing float32 noise to the reference
    curve and every calibrated number moves with it.
    """
    kwargs = dict(N=500, k=K, random_state=35, version="DANCo", fractal=False)
    s_cpu = DANCo(backend="cpu", **kwargs).evaluate_all(data, maxDim=25)
    s_gpu = DANCo(backend="gpu", **kwargs).evaluate_all(data, maxDim=25)

    cal_cpu = s_cpu["calibration_statistic"]["statistics"]
    cal_gpu = s_gpu["calibration_statistic"]["statistics"]
    assert len(cal_gpu) == len(cal_cpu) == 25
    for d, (a, b) in enumerate(zip(cal_cpu, cal_gpu), start=1):
        for key in ("dhat", "mu_nu", "mu_tau"):
            assert a[key] == b[key], f"calibration dim {d} key {key} drifted"
    assert int(s_gpu["de_standard"]) == int(s_cpu["de_standard"])


# ----------------------------------------------------------------------
# the deliberate divergences from backends.cpu
# ----------------------------------------------------------------------
@requires_cuda
def test_duplicate_filter_drops_coincident_points(cpu):
    """``dup_threshold`` is the GPU answer to ``_clean_zero_nb`` (opt-in)."""
    X = embedded_ball(n=200, d=6, D=50, seed=1)
    X[50] = X[10]  # exact duplicate

    d_cpu, i_cpu = cpu.knn(X, K)
    assert d_cpu[10, 0] == 0.0  # the CPU backend happily returns the twin

    d_gpu, i_gpu = get_backend("gpu", dup_threshold=1e-4).knn(X, K)
    assert d_gpu[10, 0] > 1e-4
    assert 50 not in i_gpu[10]

    # ... and the default (None) keeps plain kNN semantics, like the CPU.
    assert get_backend("gpu").dup_threshold is None
    d_off, i_off = get_backend("gpu").knn(X, K)
    assert d_off[10, 0] == 0.0
    # looser than the headline 1e-6: these distances are ~0.2 in a D=50 cloud,
    # where the expanded float32 form cancels harder than it does at D=2000.
    np.testing.assert_allclose(d_off, d_cpu, rtol=1e-5, atol=1e-7)


@requires_cuda
def test_extra_neighbors_zero_returns_only_k_columns(data, cpu):
    """``extra_neighbors=0`` is the original GPU off-by-one, opt-in only."""
    d_cpu, _ = cpu.knn(data, K)
    d_gpu, i_gpu = get_backend("gpu", extra_neighbors=0).knn(data, K)
    assert d_gpu.shape == (len(data), K)
    np.testing.assert_allclose(d_gpu, d_cpu[:, :K], rtol=1e-6)
    # the ratio formed from k columns is r_1 / r_k, not r_1 / r_(k+1)
    assert not np.allclose(d_gpu[:, -1], d_cpu[:, -1])


@requires_cuda
def test_too_few_survivors_is_an_error():
    """A filter that eats the neighbourhood must say so, not return inf."""
    X = np.repeat(np.arange(20.0)[:, None], 5, axis=1)
    X = np.vstack([X, X])  # every point has an exact twin
    backend = get_backend("gpu", dup_threshold=1e6)  # eats the whole cloud
    with pytest.raises(RuntimeError, match="duplicate filter"):
        backend.knn(X, K)


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
def test_cpu_backend_takes_no_options():
    with pytest.raises(TypeError, match="takes no options"):
        get_backend("cpu", query_batch=8)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_missing_cuda_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        get_backend("gpu")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_backend_options_reach_the_backend():
    est = DANCo(backend="gpu", backend_options={"device": "cpu",
                                                "query_batch": 64})
    assert est._backend.query_batch == 64
    assert est._backend.device.type == "cpu"
    # the calibration backend is the real CPU one either way
    assert est._calibration_backend is get_backend("cpu")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_bad_options_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        get_backend("gpu", device="cpu", dtype="float16")
    with pytest.raises(TypeError):
        get_backend("gpu", device="cpu", nonsense=1)
