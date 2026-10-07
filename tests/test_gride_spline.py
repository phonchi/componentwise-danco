"""The per-variant spline surfaces and the ``DANCo_gride_fast`` path.

A spline pickle stores the calibration ``dhat`` of *one* distance strategy
(recorded under its ``'variant'`` key; pickles without the key are MiND
surfaces).  Loading a surface into the other strategy's KL would compare two
different distance models, so ``load_splines`` refuses the mismatch.

The Gride surface is trained by ``experiments/train_splines.py --variant
gride`` on the *same* hyperball sequence as the shipped MiND surface
(``random_state = seed + replica``), so the two surfaces share their angular
statistics -- the angle computation never consults the distance strategy --
and differ only in ``dhat``.
"""

import os

import numpy as np
import pytest

from danco import DANCo
from danco.base import VERSIONS
from danco.calibration import hyper_ball
from danco.spline import (
    default_spline_path,
    load_splines,
    spline_statistic,
)

K = 10

GRIDE_PKL = default_spline_path("gride")
needs_gride_surface = pytest.mark.skipif(
    not os.path.exists(GRIDE_PKL),
    reason="gride spline surface not trained/packaged yet",
)


# ------------------------------------------------------------------- wiring

def test_gride_fast_version_registered():
    variant_cls, use_spline, report_strip = VERSIONS["DANCo_gride_fast"]
    assert variant_cls.name == "gride"
    assert use_spline is True
    assert report_strip is False


def test_mind_pickle_refused_for_gride():
    """The shipped (variant-less) pickle is a MiND surface."""
    with pytest.raises(ValueError, match=r"'mind'.*'gride'"):
        load_splines(default_spline_path("mind"), variant="gride")


def test_gride_fast_estimator_refuses_mind_pickle():
    X = hyper_ball(np.random.RandomState(4242), 500, 12, center=[0] * 12)
    est = DANCo(N=500, k=K, random_state=35, version="DANCo_gride_fast")
    with pytest.raises(ValueError, match=r"'mind'.*'gride'"):
        est.evaluate(X, maxDim=30, spline_file=default_spline_path("mind"))


def test_mind_default_still_loads_without_variant_argument():
    """Existing callers pass no variant; the shipped surface must keep working."""
    splines = load_splines()
    assert splines["k"] == K


# ------------------------------------------------------------- trained surface

@needs_gride_surface
def test_gride_pickle_carries_its_variant():
    splines = load_splines(GRIDE_PKL, variant="gride")
    assert splines["variant"] == "gride"
    assert min(splines["d_range"]) == 1 and max(splines["d_range"]) == 400


@needs_gride_surface
def test_gride_pickle_refused_for_mind():
    with pytest.raises(ValueError, match=r"'gride'.*'mind'"):
        load_splines(GRIDE_PKL, variant="mind")


@needs_gride_surface
def test_angular_surfaces_match_the_mind_surface():
    """Same balls, same angle kernel: the angular surfaces must agree.

    The only exception is the smallest dimensions, where the ``dhat < 5``
    shortcut may trip for one distance strategy and not the other, skipping
    the angle fit; away from that boundary the grids are the same data.
    """
    mind = load_splines(default_spline_path("mind"), variant="mind")
    gride = load_splines(GRIDE_PKL, variant="gride")
    for d in (20, 50, 100, 200, 400):
        for N in (450, 500, 700):
            np.testing.assert_allclose(
                gride["spline_mu_nu"](d, N)[0, 0],
                mind["spline_mu_nu"](d, N)[0, 0], rtol=1e-6)
            np.testing.assert_allclose(
                gride["spline_mu_tau"](d, N)[0, 0],
                mind["spline_mu_tau"](d, N)[0, 0], rtol=1e-6)


@needs_gride_surface
def test_gride_dhat_surface_tracks_dimension():
    """The Gride calibration ``dhat`` must be strictly increasing in d and
    show the same finite-sample high-dimensional underestimation as the MiND
    surface (that shared bias is what calibrated comparison corrects): both
    sit below the identity at large d, within a moderate band of each other."""
    gride = load_splines(GRIDE_PKL, variant="gride")
    mind = load_splines(default_spline_path("mind"), variant="mind")
    dims = (10, 20, 50, 100, 200, 400)
    g = [spline_statistic(gride, d, 500)["dhat"] for d in dims]
    m = [spline_statistic(mind, d, 500)["dhat"] for d in dims]
    assert all(np.diff(g) > 0)
    for d, gv, mv in zip(dims, g, m):
        assert 0 < gv <= d + 1
        assert abs(gv - mv) / mv < 0.15  # same statistic family, same balls


@needs_gride_surface
def test_gride_fast_recovers_hyperball_dimension():
    X = hyper_ball(np.random.RandomState(4242), 500, 12, center=[0] * 12)
    est = DANCo(N=500, k=K, random_state=35, version="DANCo_gride_fast",
                fractal=False)
    stats = est.evaluate_all(X, maxDim=40, spline_file=GRIDE_PKL)
    assert abs(stats["de_nustrip"] - 12) <= 3
    assert abs(stats["de_standard"] - 12) <= 3


@needs_gride_surface
def test_gride_fast_matches_monte_carlo_reference():
    """Spline and Monte-Carlo calibration must agree on the estimate itself
    (the surface averages 35 balls; the estimates are integers, so modest
    smoothing noise must not move the argmin by more than one)."""
    X = hyper_ball(np.random.RandomState(77), 500, 15, center=[0] * 15)
    fast = DANCo(N=500, k=K, random_state=35, version="DANCo_gride_fast",
                 fractal=False)
    stats_fast = fast.evaluate_all(X, maxDim=40, spline_file=GRIDE_PKL)
    mc = DANCo(N=500, k=K, random_state=35, version="DANCo_gride",
               fractal=False, calibration_seeding="derived")
    stats_mc = mc.evaluate_all(X, maxDim=40)
    assert abs(stats_fast["de_nustrip"] - stats_mc["de_nustrip"]) <= 1
    assert abs(stats_fast["de_standard"] - stats_mc["de_standard"]) <= 1
