"""Regression checks for the modern generic-order Gride likelihood."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from danco._kernels.distances import argmax_loglik, neg_dloglik_did


def _published_log_likelihood(dimension, ratios, n1, n2):
    """Denti et al. (2022), Eq. (15), up to its constant beta term."""
    log_ratios = np.log(ratios)
    return (
        ratios.size * np.log(dimension)
        + (n2 - n1 - 1) * np.sum(np.log(ratios ** dimension - 1.0))
        - ((n2 - 1) * dimension + 1.0) * np.sum(log_ratios)
    )


def _published_score(dimension, ratios, n1, n2):
    """Derivative of :func:`_published_log_likelihood`."""
    ratios_d = ratios ** dimension
    log_ratios = np.log(ratios)
    return (
        ratios.size / dimension
        + (n2 - n1 - 1)
        * np.sum(ratios_d * log_ratios / (ratios_d - 1.0))
        - (n2 - 1) * np.sum(log_ratios)
    )


def test_gride_root_matches_independent_likelihood_maximum():
    """The bisection root agrees with direct maximization of published Eq. (15)."""
    rng = np.random.RandomState(202208)
    n1, n2 = 5, 10
    generating_dimension = 12.0
    beta_draws = rng.beta(n2 - n1, n1, size=80)
    ratios = (1.0 - beta_draws) ** (-1.0 / generating_dimension)

    optimum = minimize_scalar(
        lambda d: -_published_log_likelihood(d, ratios, n1, n2),
        bounds=(0.1, 100.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert optimum.success

    estimate = argmax_loglik(
        0.001,
        1000.0,
        ratios.copy(),
        1e-10,
        ratios.size,
        n1,
        n2,
    )

    assert estimate == pytest.approx(optimum.x, rel=1e-7)
    assert _published_score(estimate, ratios, n1, n2) == pytest.approx(
        0.0, abs=1e-8
    )


def test_modern_score_is_negative_published_score():
    """Protect the ``N / d`` score term independently of the root finder."""
    ratios = np.array([1.01, 1.03, 1.07, 1.12, 1.20, 1.35], dtype=float)
    n1, n2 = 5, 10
    dimension = 7.5

    actual = neg_dloglik_did(
        dimension,
        ratios.copy(),
        1e-12,
        ratios.size,
        n1,
        n2,
    )

    assert actual == pytest.approx(
        -_published_score(dimension, ratios, n1, n2)
    )
