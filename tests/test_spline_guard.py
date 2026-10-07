"""The ``DANCo_fast`` range guard.

``RectBivariateSpline`` extrapolates outside its fitted grid without warning, so
an unguarded ``DANCo_fast`` happily produces "calibration statistics" for
N=2500 from a surface fitted on N in [450, 700] -- numbers with no data behind
them.  The guard turns that into a ``ValueError``.

The guard must be a pure gate: inside the fitted grid, turning it on or off has
to give bit-identical results, otherwise it would itself be a source of drift.
"""

import numpy as np
import pytest

from danco import DANCo
from danco.calibration import hyper_ball
from danco.spline import (
    grow_spline_calibration,
    load_splines,
    spline_statistic,
)

K = 10


@pytest.fixture(scope="module")
def splines():
    return load_splines()


# ---------------------------------------------------------------- module level

def test_inside_grid_is_bitwise_identical(splines):
    """The guard changes control flow only, never a number."""
    for N in (450, 500, 623, 700):
        for d in (1, 2, 17, 200, 400):
            checked = spline_statistic(splines, d, N, check_range=True)
            unchecked = spline_statistic(splines, d, N, check_range=False)
            assert checked == unchecked  # exact dict equality, no tolerance
            assert all(isinstance(v, float) for v in checked.values())


@pytest.mark.parametrize("N", [0, 100, 449, 701, 2500, 10000])
def test_out_of_range_N_raises(splines, N):
    with pytest.raises(ValueError, match=r"N=.*outside the range"):
        spline_statistic(splines, 10, N)


@pytest.mark.parametrize("d", [0, -1, 401, 1000])
def test_out_of_range_d_raises(splines, d):
    with pytest.raises(ValueError, match=r"d=.*outside the range"):
        spline_statistic(splines, d, 500)


def test_error_message_names_the_fitted_range(splines):
    with pytest.raises(ValueError) as excinfo:
        spline_statistic(splines, 10, 2500)
    message = str(excinfo.value)
    assert "450" in message and "700" in message
    assert "DANCo" in message  # points at the workaround


def test_growth_raises_only_when_it_actually_leaves_the_grid():
    """Growing to exactly the last fitted dimension is fine; one past it is not."""
    cache = dict(statistics=list(), maxdim=0)
    grow_spline_calibration(cache, 400, 500)
    assert cache["maxdim"] == 400
    assert len(cache["statistics"]) == 400

    with pytest.raises(ValueError, match=r"d=401.*outside the range"):
        grow_spline_calibration(cache, 401, 500)


# ------------------------------------------------------------- estimator level

@pytest.fixture(scope="module")
def data_500():
    return hyper_ball(np.random.RandomState(4242), 500, 12, center=[0] * 12)


def test_estimator_in_range_is_bitwise_identical(data_500):
    """Full ``evaluate`` at N=500 (inside the grid) with the guard on vs off."""
    guarded = DANCo(N=500, k=K, random_state=35, version="DANCo_fast",
                    check_spline_range=True)
    unguarded = DANCo(N=500, k=K, random_state=35, version="DANCo_fast",
                      check_spline_range=False)
    stats_a = guarded.evaluate_all(data_500, maxDim=30)
    stats_b = unguarded.evaluate_all(data_500, maxDim=30)
    de_a, de_b = stats_a["de_standard"], stats_b["de_standard"]

    assert de_a == de_b
    for key in ("dhat", "mu_nu", "mu_tau"):
        assert stats_a[key] == stats_b[key]
    np.testing.assert_array_equal(stats_a["kl_divergence"], stats_b["kl_divergence"])
    np.testing.assert_array_equal(stats_a["kl_ang_strip"], stats_b["kl_ang_strip"])


def test_estimator_at_default_N_refuses_to_extrapolate():
    """N=2500 -- the estimator default -- is pure extrapolation and must raise."""
    X = hyper_ball(np.random.RandomState(99), 2500, 12, center=[0] * 12)
    est = DANCo(N=2500, k=K, random_state=35, version="DANCo_fast")
    with pytest.raises(ValueError, match=r"N=2500 is outside the range"):
        est.evaluate(X, maxDim=20)


def test_unguarded_estimator_still_extrapolates(data_500):
    """Opting out is possible, so the guard can be switched off to reproduce
    unguarded numbers rather than being a hard wall."""
    X = hyper_ball(np.random.RandomState(99), 2500, 12, center=[0] * 12)
    est = DANCo(N=2500, k=K, random_state=35, version="DANCo_fast",
                check_spline_range=False)
    de, _ = est.evaluate(X, maxDim=20)
    assert np.isfinite(de)


@pytest.mark.parametrize("version", ["DANCo_fast", "DANCo_gride_fast"])
def test_spline_k_mismatch_raises(data_500, version):
    """Both packaged surfaces were trained with k=10; another k must refuse."""
    est = DANCo(N=500, k=20, random_state=35, version=version)
    with pytest.raises(ValueError,
                       match=r"Spline trained with k=10, requested k=20"):
        est.evaluate(data_500, maxDim=12)
    assert est.calibration_statistic["maxdim"] == 0


def test_range_error_names_the_matching_versions():
    gride = load_splines(variant="gride")
    with pytest.raises(ValueError) as excinfo:
        spline_statistic(gride, 10, 2500)
    message = str(excinfo.value)
    assert message.startswith("DANCo_gride_fast:")
    assert "version='DANCo_gride'" in message


def test_default_spline_is_packaged():
    """The default asset resolves through the installed package, not the cwd."""
    from danco.spline import default_spline_path

    path = default_spline_path()
    assert path.endswith("fastdanco_splines_400.pkl")
    assert "danco/assets" in path.replace("\\", "/")
    loaded = load_splines()
    assert loaded["k"] == 10
    assert min(loaded["d_range"]) == 1 and max(loaded["d_range"]) == 400
