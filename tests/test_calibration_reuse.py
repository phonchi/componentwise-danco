"""Passing a calibration cache in, and getting a useful error when it is unusable.

``calibration_data`` is public API, and the obvious thing to feed it is a cache
another estimator produced.  A cache whose statistics lack ``mu_nu`` cannot
work, because this package always computes the standard angle KL alongside the
strip one.  That has to fail at construction with an explanation, not with a
bare ``KeyError`` halfway through a long run.
"""

import copy

import numpy as np
import pytest

from danco import DANCo
from danco.calibration import hyper_ball

K = 10
N = 500


@pytest.fixture(scope="module")
def data():
    return hyper_ball(np.random.RandomState(808), N, 12, center=[0] * 12)


def test_prebuilt_cache_is_reused_and_reproduces(data):
    """A cache from a completed run gives the same answer with no RNG draws."""
    donor = DANCo(N=N, k=K, random_state=35)
    expected, _ = donor.evaluate(data, maxDim=20)

    receiver = DANCo(N=N, k=K, random_state=999,  # different seed on purpose
                     calibration_data=donor.calibration_statistic)
    rng_before = receiver.random_state_.get_state()[2]
    got, _ = receiver.evaluate(data, maxDim=20)

    assert got == expected
    assert receiver.random_state_.get_state()[2] == rng_before


def test_cache_without_nu_is_rejected_with_an_explanation(data):
    """A cache that never fitted ``nu`` cannot be reused."""
    donor = DANCo(N=N, k=K, random_state=35)
    donor.evaluate(data, maxDim=20)
    cache = copy.deepcopy(donor.calibration_statistic)
    for stats in cache["statistics"]:
        stats.pop("mu_nu", None)
    assert "mu_nu" not in cache["statistics"][-1]

    with pytest.raises(ValueError, match=r"missing \['mu_nu'\]"):
        DANCo(N=N, k=K, calibration_data=cache)


def test_inconsistent_cache_is_rejected():
    with pytest.raises(ValueError, match="claims maxdim=3 but carries 1"):
        DANCo(calibration_data={
            "statistics": [{"dhat": 1.0, "mu_nu": 0.0, "mu_tau": 0.0}],
            "maxdim": 3,
        })


def test_malformed_cache_is_rejected():
    with pytest.raises(ValueError, match="missing key"):
        DANCo(calibration_data={"statistics": []})
    with pytest.raises(TypeError, match="should be a dict"):
        DANCo(calibration_data=[{"dhat": 1.0}])
