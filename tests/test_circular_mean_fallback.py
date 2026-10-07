"""Circular-mean validity and distance-only fallback regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest

import danco.base as base_module
from danco import DANCo
from danco._kernels.vonmises import circular_mean_resultant, ml_vm


class _StatisticBackend:
    """Minimal backend for exercising ``danco_statistic`` in isolation."""

    def __init__(self, n, k):
        self.n = n
        self.k = k

    def knn(self, X, k, n_jobs=1):
        return np.ones((self.n, self.k)), np.zeros(
            (self.n, self.k), dtype=int
        )

    def angle_stats(self, X, idx, combs, k):
        return np.zeros((self.n, 1))


def test_circular_mean_resultant_is_quadrant_aware():
    angles = np.array([3 * np.pi / 4, 3 * np.pi / 4])

    direction, resultant = circular_mean_resultant(angles)

    assert direction == pytest.approx(3 * np.pi / 4)
    assert resultant == pytest.approx(2.0)


def test_circular_mean_direction_uses_positive_pi_endpoint():
    direction, resultant = circular_mean_resultant(np.array([-np.pi]))

    assert direction == np.pi
    assert resultant == 1


def test_circular_mean_resultant_marks_exact_zero_as_undefined():
    # These four floating-point angles have exactly cancelling sine and cosine
    # sums under NumPy, so this exercises the exact-zero policy (not a tolerance).
    angles = np.array([-np.pi, -0.589048622548086, 0.0, 2.552544031041707])
    assert np.sum(np.sin(angles)) == 0
    assert np.sum(np.cos(angles)) == 0

    direction, resultant = circular_mean_resultant(angles)

    assert resultant == 0
    assert np.isnan(direction)


def test_ml_vm_can_report_per_center_resultants():
    angles = np.array([
        [-np.pi, -0.589048622548086, 0.0, 2.552544031041707],
        [0.2, 0.3, 0.4, 0.5],
    ])

    nu, tau, resultant = ml_vm(angles, return_resultant=True)

    assert resultant[0] == 0
    assert np.isnan(nu[0])
    assert resultant[1] > 0
    assert np.isfinite(nu[1])
    assert tau.shape == (2,)


@pytest.mark.parametrize("failure_level", ["per_center", "dataset"])
def test_danco_statistic_detects_zero_resultant(monkeypatch, failure_level):
    n, k = 4, 2
    est = DANCo(N=n, k=k, fractal=False)
    est.variant = SimpleNamespace(
        estimate_dimension=lambda distances, ambient_dim: 8.0
    )
    backend = _StatisticBackend(n, k)

    if failure_level == "per_center":
        nu = np.zeros(n)
        center_resultant = np.ones(n)
        center_resultant[1] = 0
    else:
        nu = np.array([-np.pi, -0.589048622548086, 0.0, 2.552544031041707])
        center_resultant = np.ones(n)

    def fake_ml_vm(thetas, return_resultant=False):
        assert return_resultant
        return nu, np.ones(n), center_resultant

    monkeypatch.setattr(base_module, "ml_vm", fake_ml_vm)
    stats = est.danco_statistic(np.zeros((n, 8)), backend=backend)

    assert stats["dhat"] == 8.0
    assert stats["angle_valid"] is False
    assert stats["angle_status"] == "zero_resultant"
    assert np.isnan(stats["mu_nu"])
    assert np.isnan(stats["mu_tau"])


def test_danco_statistic_reports_valid_angle(monkeypatch):
    n, k = 4, 2
    est = DANCo(N=n, k=k, fractal=False)
    est.variant = SimpleNamespace(
        estimate_dimension=lambda distances, ambient_dim: 8.0
    )

    def fake_ml_vm(thetas, return_resultant=False):
        assert return_resultant
        return np.zeros(n), np.ones(n), np.ones(n)

    monkeypatch.setattr(base_module, "ml_vm", fake_ml_vm)
    stats = est.danco_statistic(
        np.zeros((n, 8)), backend=_StatisticBackend(n, k)
    )

    assert stats["angle_valid"] is True
    assert stats["angle_status"] == "ok"
    assert stats["mu_nu"] == 0
    assert stats["mu_tau"] == 1


@pytest.mark.parametrize(
    "version", ["DANCo", "DANCo_fast", "DANCo_gride", "DANCo_nustrip",
     "DANCo_gride_nustrip", "DANCo_nustrip_fast", "DANCo_gride_nustrip_fast"]
)
def test_zero_resultant_returns_distance_only_without_calibration(version):
    n, k = 12, 3
    est = DANCo(N=n, k=k, version=version, random_state=23, fractal=False)
    rng_position = est.random_state_.get_state()[2]

    def zero_resultant_statistic(X, n_jobs=1, backend=None):
        return {
            "dhat": 8.5,
            "mu_nu": np.nan,
            "mu_tau": np.nan,
            "angle_valid": False,
            "angle_status": "zero_resultant",
        }

    est.danco_statistic = zero_resultant_statistic
    stats = est.evaluate_all(np.zeros((n, 10)), maxDim=10)
    dimension = stats["de_nustrip" if est.objective == "profiled" else "de_standard"]

    assert dimension == stats["dhat"] == 8.5
    assert stats["de_standard"] == stats["de_nustrip"] == 8.5
    assert stats["kl_divergence"] is None
    assert stats["kl_ang_strip"] is None
    assert stats["angle_valid"] is False
    assert stats["angle_status"] == "zero_resultant"
    assert est.calibration_statistic["maxdim"] == 0
    assert est.calibration_statistic["statistics"] == []
    assert est.random_state_.get_state()[2] == rng_position


def test_distance_shortcut_has_explicit_angle_status():
    n, k = 12, 3
    est = DANCo(N=n, k=k, random_state=23, fractal=False)

    def threshold_statistic(X, n_jobs=1, backend=None):
        return {
            "dhat": 5.0,
            "mu_nu": 0.4,
            "mu_tau": 2.0,
            "angle_valid": True,
            "angle_status": "ok",
        }

    est.danco_statistic = threshold_statistic
    stats = est.evaluate_all(np.zeros((n, 10)), maxDim=10)
    dimension = stats["de_nustrip" if est.objective == "profiled" else "de_standard"]

    assert dimension == 5.0
    assert stats["angle_valid"] is False
    assert stats["angle_status"] == "distance_below_threshold"
    assert stats["kl_divergence"] is None
    assert est.calibration_statistic["maxdim"] == 0


def test_three_field_calibration_cache_remains_accepted():
    cache = {
        "statistics": [{"dhat": 1.0, "mu_nu": 0.0, "mu_tau": 0.0}],
        "maxdim": 1,
    }

    est = DANCo(calibration_data=cache)

    assert est.calibration_statistic is cache


def test_generated_calibration_cache_keeps_three_field_schema():
    n, k = 12, 3
    est = DANCo(N=n, k=k, random_state=23, fractal=False)
    calls = 0

    def statistic(X, n_jobs=1, backend=None):
        nonlocal calls
        calls += 1
        return {
            "dhat": 8.0 if calls == 1 else float(calls - 1),
            "mu_nu": 0.4,
            "mu_tau": 2.0,
            "angle_valid": True,
            "angle_status": "ok",
        }

    est.danco_statistic = statistic
    est.evaluate(np.zeros((n, 2)), maxDim=2)

    assert len(est.calibration_statistic["statistics"]) == 2
    assert all(
        set(entry) == {"dhat", "mu_nu", "mu_tau"}
        for entry in est.calibration_statistic["statistics"]
    )
