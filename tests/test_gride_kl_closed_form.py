"""The finite digamma sum for the Gride KL reference term matches quadrature.

The closed form (Supplementary Eq. S1.9) and the one-dimensional quadrature
used by the archived results agree within 1e-6 over the candidate ranges used
(m <= 100 for k in {5, 15, 20}; m <= 400 at k = 10).  Neighbor orders:
k = 5 -> (3, 6), k = 10 -> (5, 10), k = 15 -> (8, 16), k = 20 -> (10, 20).
Only data estimates above 5 reach the KL, so the grids start above 5.
"""

import numpy as np
import pytest
from scipy.special import digamma

from danco._kernels import distances as D


@pytest.mark.parametrize("n1,n2", [(5, 10), (3, 6), (8, 16), (10, 20)])
def test_reference_log_moment_reduces_to_lemma_at_gamma_one(n1, n2):
    expected = digamma(n2 - n1) - digamma(n1)
    assert abs(D.gride_reference_log_moment(1.0, n1, n2) - expected) < 1e-8


def test_closed_form_matches_quadrature_over_candidate_range():
    """k = 10: candidates m <= 400."""
    worst = 0.0
    for dhat in (7.0, 15.0, 30.0, 70.0, 150.0, 300.0, 400.0):
        for dcal in np.linspace(1.0, 420.0, 43):
            q = D.kl_gride(dhat, dcal, 5, 10)
            c = D.kl_gride_closed_form(dhat, dcal, 5, 10)
            worst = max(worst, abs(q - c))
    assert worst < 1e-6, worst


@pytest.mark.parametrize("n1,n2", [(3, 6), (8, 16), (10, 20)])
def test_closed_form_matches_quadrature_other_orders(n1, n2):
    """k in {5, 15, 20}: candidates m <= 100 (reference estimates up to 120)."""
    worst = 0.0
    for dhat in (7.0, 15.0, 30.0, 70.0, 100.0, 120.0):
        for dcal in np.linspace(1.0, 120.0, 25):
            q = D.kl_gride(dhat, dcal, n1, n2)
            c = D.kl_gride_closed_form(dhat, dcal, n1, n2)
            worst = max(worst, abs(q - c))
    assert worst < 1e-6, worst


def test_closed_form_kl_vanishes_at_match():
    for d in (5.0, 40.0, 200.0):
        assert abs(D.kl_gride_closed_form(d, d, 5, 10)) < 1e-9
