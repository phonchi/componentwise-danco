"""The eight versions form a 2 x 2 x 2 grid, and ``evaluate`` agrees bitwise
with the matching entry of ``evaluate_all``."""
import numpy as np
import pytest

from danco import DANCo
from danco.base import OBJECTIVES, VERSIONS
from danco.spline import default_spline_path
from danco.variants.gride import GrideVariant
from danco.variants.mind import MiNDVariant

K = 10
N = 500
D = 20
TRUE_D = 12
MAXDIM = 30

GRID = {
    ("mind", "full", "monte_carlo"): "DANCo",
    ("mind", "profiled", "monte_carlo"): "DANCo_nustrip",
    ("gride", "full", "monte_carlo"): "DANCo_gride",
    ("gride", "profiled", "monte_carlo"): "DANCo_gride_nustrip",
    ("mind", "full", "spline"): "DANCo_fast",
    ("mind", "profiled", "spline"): "DANCo_nustrip_fast",
    ("gride", "full", "spline"): "DANCo_gride_fast",
    ("gride", "profiled", "spline"): "DANCo_gride_nustrip_fast",
}
VARIANT_CLASSES = {"mind": MiNDVariant, "gride": GrideVariant}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    X = np.zeros((N, D))
    X[:, :TRUE_D] = rng.normal(size=(N, TRUE_D))
    return X


def test_versions_form_the_full_grid():
    assert set(VERSIONS) == set(GRID.values())
    for (variant, objective, reference), version in GRID.items():
        variant_cls, use_spline, report_profiled = VERSIONS[version]
        assert variant_cls is VARIANT_CLASSES[variant]
        assert use_spline is (reference == "spline")
        assert report_profiled is (objective == "profiled")
        assert DANCo(N=N, k=K, version=version).objective == objective


@pytest.mark.parametrize("key,version", sorted(GRID.items()))
def test_evaluate_matches_evaluate_all_bitwise(data, key, version):
    variant, objective, reference = key
    kwargs = dict(N=N, k=K, random_state=11, version=version)
    if reference == "monte_carlo":
        kwargs["calibration_seeding"] = "derived"
    spline = (default_spline_path(variant) if reference == "spline" else None)

    dimension, stats = DANCo(**kwargs).evaluate(
        data, maxDim=MAXDIM, spline_file=spline)
    both = DANCo(**kwargs).evaluate_all(data, maxDim=MAXDIM, spline_file=spline)

    expected_key = "de_nustrip" if objective == "profiled" else "de_standard"
    assert dimension == both[expected_key]
    assert stats["estimate"] == dimension
    assert stats["objective"] == objective
    assert abs(dimension - TRUE_D) <= 3

    np.testing.assert_array_equal(stats["kl_distance"], both["kl_divergence"][:, 0])
    angle_all = (both["kl_ang_strip"] if objective == "profiled"
                 else both["kl_divergence"][:, 1])
    np.testing.assert_array_equal(stats["kl_angle"], angle_all)

    # evaluate exposes only its own objective; evaluate_all exposes both.
    for other in ("de_standard", "de_nustrip", "kl_divergence", "kl_ang_strip"):
        assert other not in stats
    assert set(OBJECTIVES) == {"full", "profiled"}
    for needed in ("de_standard", "de_nustrip", "kl_divergence", "kl_ang_strip",
                   "dhat", "mu_nu", "mu_tau", "angle_valid", "angle_status",
                   "calibration_statistic"):
        assert needed in both


def test_shortcut_keeps_estimate_and_nulls_curves():
    rng = np.random.default_rng(3)
    X = np.zeros((N, D))
    X[:, :2] = rng.normal(size=(N, 2))          # a 2-dimensional cloud: dhat <= 5
    dimension, stats = DANCo(N=N, k=K, random_state=1,
                             version="DANCo_gride_nustrip").evaluate(X, maxDim=MAXDIM)
    assert stats["angle_status"] == "distance_below_threshold"
    assert dimension == stats["estimate"] == stats["dhat"]
    assert stats["kl_distance"] is None and stats["kl_angle"] is None
    both = DANCo(N=N, k=K, random_state=1, version="DANCo_gride").evaluate_all(
        X, maxDim=MAXDIM)
    assert both["de_standard"] == both["de_nustrip"] == both["dhat"]
    assert both["kl_divergence"] is None and both["kl_ang_strip"] is None
