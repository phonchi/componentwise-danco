"""The 'derived' calibration-seeding protocol.

Three contracts:

1. With ``calibration_seeding='derived'``, the estimate for a dataset does
   not depend on what was evaluated before it (order independence).  The
   default 'stream' mode is order independent too, because candidate m always
   consumes the m-th draw; 'derived' additionally decouples the dimensions.
2. Two derived-mode estimators with the same ``(random_state, N)`` produce
   identical calibration references, so results agree across instances.
3. The default remains 'stream' and rejects unknown values, so no existing
   protocol changes behavior.  'derived' requires an integer random_state.
"""

import numpy as np
import pytest

from danco import DANCo


def _make_data(seed, n=300, d=8, ambient=20):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    out = np.zeros((n, ambient))
    out[:, :d] = x
    return out


def _estimator(**kwargs):
    return DANCo(N=300, k=10, random_state=11, fractal=False, verbose=False,
                 calibration_seeding="derived", **kwargs)


def test_derived_mode_is_order_independent():
    a, b = _make_data(1), _make_data(2, d=12)

    est_ab = _estimator()
    dim_a_first, _ = est_ab.evaluate(a, maxDim=30)
    dim_b_second, _ = est_ab.evaluate(b, maxDim=30)

    est_ba = _estimator()
    dim_b_first, _ = est_ba.evaluate(b, maxDim=30)
    dim_a_second, _ = est_ba.evaluate(a, maxDim=30)

    assert dim_a_first == dim_a_second
    assert dim_b_first == dim_b_second


def test_derived_mode_agrees_across_instances():
    a = _make_data(3)
    stats1 = _estimator().evaluate_all(a, maxDim=30)
    stats2 = _estimator().evaluate_all(a, maxDim=30)
    assert stats1["de_standard"] == stats2["de_standard"]
    assert stats1["de_nustrip"] == stats2["de_nustrip"]
    np.testing.assert_array_equal(stats1["kl_divergence"],
                                  stats2["kl_divergence"])


def test_stream_mode_is_default_and_order_independent():
    est = DANCo(N=300, k=10, random_state=11)
    assert est.calibration_seeding == "stream"

    a, b = _make_data(1), _make_data(2, d=12)
    alone = DANCo(N=300, k=10, random_state=11, fractal=False)
    stats_alone = alone.evaluate_all(b, maxDim=30)
    after = DANCo(N=300, k=10, random_state=11, fractal=False)
    after.evaluate(a, maxDim=20)
    stats_after = after.evaluate_all(b, maxDim=30)
    np.testing.assert_array_equal(stats_alone["kl_divergence"],
                                  stats_after["kl_divergence"])


@pytest.mark.parametrize("seed", [None, np.random.RandomState(1), True])
def test_derived_mode_requires_integer_random_state(seed):
    with pytest.raises(ValueError, match="integer random_state"):
        DANCo(N=300, random_state=seed, calibration_seeding="derived")
    # stream mode keeps its original input contract
    DANCo(N=300, random_state=seed)


def test_unknown_seeding_rejected():
    with pytest.raises(ValueError):
        DANCo(N=300, calibration_seeding="typo")
