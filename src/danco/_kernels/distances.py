"""Pure numerical cores of the two distance-based dimension estimators.

Two families live here:

MiND_ML (used by the ``DANCo*`` versions without ``gride``)
    Works on ``rhos = r_1 / r_(k+1)``.  ``mind_mli`` gives an integer starting
    point by grid search, ``mind_mlk`` refines it with L-BFGS-B.  ``kl_mind``
    is the closed-form KL between two MiND_ML densities.

Gride (used by the ``DANCo_gride*`` versions)
    Works on ``rho = r_n2 / r_n1``.  ``argmax_loglik`` bisects the derivative of
    the published generic-order log-likelihood.  ``kl_gride``, used by the
    estimator, evaluates the cross term by quadrature; ``kl_gride_closed_form``
    gives the same divergence through the finite digamma sum of Supplementary
    Eq. (S1.9).

Every function is free-standing and takes ``k`` / ``N`` / ``n1`` / ``n2``
explicitly, so the variant classes stay thin.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import beta, comb, digamma

__all__ = [
    "lld",
    "nlld",
    "nlld_gr",
    "mind_mli",
    "mind_mlk",
    "harmonic_number",
    "kl_mind",
    "filter_mus",
    "neg_dloglik_did",
    "argmax_loglik",
    "kl_gride",
]


# --------------------------------------------------------------------------
# MiND_ML
# --------------------------------------------------------------------------

def lld(d, rhos, N, k):
    """Log-likelihood of the MiND_ML density at dimension ``d``."""
    if d == 0:
        return np.array([-1e30])
    return (
        N * np.log(k * d)
        + (d - 1) * np.sum(np.log(rhos))
        + (k - 1) * np.sum(np.log(1 - rhos ** d))
    )


def nlld(d, rhos, N, k):
    return -lld(d, rhos, N, k)


def nlld_gr(d, rhos, N, k):
    if d == 0:
        return np.array([-1e30])
    return -(
        N / d
        + np.sum(
            np.log(rhos) - (k - 1) * (rhos ** d) * np.log(rhos) / (1 - rhos ** d)
        )
    )


def mind_mli(rhos, D, N, k):
    """Integer grid search over ``d in 1..D``; returns the argmax."""
    d_lik = np.array([np.nan] * D)
    for d in range(D):
        d_lik[d] = lld(d + 1, rhos, N, k)
    return np.argmax(d_lik) + 1


def mind_mlk(rhos, dinit, D, k):
    """Continuous refinement of ``dinit`` with L-BFGS-B.

    NOTE: ``len(rhos)`` -- not the configured ``N`` -- is passed as the
    sample size to the objective, while :func:`mind_mli` uses the configured
    ``N``.  That asymmetry is inherited from the earlier implementation and
    kept deliberately: it shifts the optimum, so "fixing" it would change the
    archived results.
    """
    res = minimize(
        fun=nlld,
        x0=np.array([dinit]),
        jac=nlld_gr,
        args=(rhos, len(rhos), k),
        method="L-BFGS-B",
        bounds=[(0, D)],
    )
    return res["x"][0]


def harmonic_number(n):
    return sum(1.0 / i for i in range(1, n + 1))


def kl_mind(dhat, dcal, k):
    """Closed-form KL between MiND_ML densities of dimension ``dhat`` / ``dcal``."""
    ratio = dcal / dhat
    h_k = harmonic_number(k)
    h_k_minus_1 = harmonic_number(k - 1)
    part1 = h_k * ratio - 1 - h_k_minus_1 - np.log(ratio)
    series_sum = 0
    for i in range(k + 1):
        term = ((-1) ** i) * comb(k, i) * digamma(1 + i / ratio)
        series_sum += term
    part2 = (k - 1) * series_sum
    return part1 - part2


# --------------------------------------------------------------------------
# Gride
# --------------------------------------------------------------------------

def filter_mus(mus):
    """Legacy no-op except the ``mu = 1`` nudge.

    The percentile filter below is computed but its result is discarded by
    :func:`argmax_loglik`; only the in-place nudge affects the estimate.
    Two behaviours here are load-bearing and intentionally kept:

    1. ``mus`` is nudged **in place** where it equals exactly 1, so the caller's
       array is mutated even though a *new* filtered array is returned.
    2. The earlier implementation discards the returned array and keeps
       optimising over the unfiltered (but nudged) input; ``argmax_loglik``
       below reproduces that.
    """
    indx = np.nonzero(mus == 1)
    mus[indx] += 10 * np.finfo(mus.dtype).eps
    q3, q2 = np.percentile(mus, [95, 50])
    mu_max = 20 * (q3 - q2) + q2
    select = mus < mu_max
    return mus[select]


def neg_dloglik_did(d, mus, eps, N, n1, n2):
    """Negative score of the generic-order Gride log-likelihood.

    ``N`` is the number of observed ratios.  The likelihood contains
    ``N * log(d)``, so its score contributes ``N / d`` (the published score,
    not an ``(N - 1) / d`` finite-sample adjustment).
    """
    one_m_mus_d = 1.0 - mus ** (-d)
    one_m_mus_d[one_m_mus_d < 2 * eps] = 2 * eps
    summation = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return summation - N / d


def argmax_loglik(d0, d1, rho, eps, N, n1, n2):
    """Bisect the published Gride score between ``d0`` and ``d1``."""
    # Called for the in-place nudge only; the filtered copy is discarded, which
    # retains the earlier ratio handling.  See filter_mus().
    filter_mus(rho)
    l1 = neg_dloglik_did(d1, rho, eps, N, n1, n2)
    while abs(d0 - d1) > eps:
        d2 = (d0 + d1) / 2.0
        l2 = neg_dloglik_did(d2, rho, eps, N, n1, n2)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    return (d0 + d1) / 2.0


def gride_reference_log_moment(gamma, n1, n2):
    """Closed form of the reference-side expectation in the Gride KL.

    Returns ``E[log((1+z)**gamma - 1)]`` for ``z ~ beta'(n2-n1, n1)`` as the
    finite digamma sum (Supplementary Eq. (S1.9)); it equals
    ``digamma(n2-n1) - digamma(n1)`` at ``gamma == 1``.
    """
    a, b = n2 - n1, n1
    total = 0.0
    for j in range(a):
        total += ((-1) ** j) * comb(a - 1, j, exact=True) / (b + j) * (
            digamma(1.0) - digamma((b + j) / gamma)
        )
    return total / beta(a, b)


def kl_gride_closed_form(dhat, dcal, n1, n2):
    """Closed-form Gride KL; agrees with :func:`kl_gride` to about 1e-6."""
    term2 = (digamma(n2) - digamma(n1)) / dhat
    term31 = digamma(n2 - n1) - digamma(n1)
    term32 = gride_reference_log_moment(dcal / dhat, n1, n2)
    return (
        np.log(dhat / dcal)
        + (n2 - 1) * (dcal - dhat) * term2
        + (n2 - n1 - 1) * (term31 - term32)
    )


def kl_gride(dhat, dcal, n1, n2):
    """KL between two Gride densities; the cross term is evaluated by quadrature.

    :func:`kl_gride_closed_form` gives the same value through the finite
    digamma sum; the archived results use this quadrature form.
    """

    def log_g(r, n_1, n_2, d):
        log_r = np.log(r)
        log_r_d = d * log_r  # log(r^d - 1) = log1p(-r^-d) + d*log(r)
        log_r_d_minus1 = np.log1p(-np.exp(-log_r_d)) + log_r_d
        return (
            np.log(d)
            + (n_2 - n_1 - 1) * log_r_d_minus1
            + (d * (1 - n_2) - 1) * log_r
            - np.log(beta(n_2 - n_1, n_1))
        )

    def integrand(r):
        log_r = np.log(r)
        log_r_d2 = dcal * log_r
        log_term = np.log1p(-np.exp(-log_r_d2)) + log_r_d2  # log(r^d2 - 1)
        log_pdf = log_g(r, n1, n2, dhat)
        return log_term * np.exp(log_pdf)

    term32, _ = quad(integrand, 1 + 1e-10, np.inf, limit=50, epsrel=1e-6)

    term2 = (digamma(n2) - digamma(n1)) / dhat
    term31 = digamma(n2 - n1) - digamma(n1)
    return (
        np.log(dhat / dcal)
        + (n2 - 1) * (dcal - dhat) * term2
        + (n2 - n1 - 1) * (term31 - term32)
    )
