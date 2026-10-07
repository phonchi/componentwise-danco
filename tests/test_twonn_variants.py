"""Guard and reference-variant behavior of the TWO-NN estimators."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from common import two_nn_reference  # noqa: E402
from danco import two_nn_legacy  # noqa: E402


def _gaussian(n=400, d=6, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


def test_two_nn_clean_data_matches_unguarded_formula():
    X = _gaussian()
    from sklearn.neighbors import NearestNeighbors

    dists, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors()
    expected = (X.shape[0] - 1) / np.log(dists[:, 1] / dists[:, 0]).sum()
    assert two_nn_legacy(X) == pytest.approx(expected, rel=1e-12)


def test_two_nn_guard_keeps_duplicates_finite():
    X = _gaussian()
    X_dup = np.vstack([X, X[:25]])
    value = two_nn_legacy(X_dup)
    assert np.isfinite(value) and value > 0
    # Duplicates are dropped, so the guarded value stays near the clean one.
    assert value == pytest.approx(two_nn_legacy(X), rel=0.25)


def test_two_nn_alias_is_gone():
    import danco

    assert "two_nn" not in danco.__all__
    assert not hasattr(danco, "two_nn")


def test_two_nn_reference_recovers_known_dimension():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((2000, 5))
    ball = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    ball *= rng.uniform(0, 1, size=(2000, 1)) ** (1 / 5)
    assert two_nn_reference(ball) == pytest.approx(5.0, abs=1.0)


def test_two_nn_reference_tolerates_duplicates():
    X = _gaussian()
    value = two_nn_reference(np.vstack([X, X[:25]]))
    assert np.isfinite(value) and value > 0
