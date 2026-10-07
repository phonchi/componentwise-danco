"""Scaled versus unscaled Bessel evaluation of the angular KL: same answers
where the unscaled path works, finite answers where it does not.

The earlier implementation wrote the angle KL with unscaled ``i0``/``i1`` and guarded the
overflow with ``min(sys.float_info.max, .)``.  That guard does not work.  Once
the concentration ``tau`` passes the point where ``i0``/``i1`` overflow to
``inf`` there are two distinct failure modes:

* **data-side (tau1) overflow -> NaN.**  The denominator ``2 * min(FMAX, i0(tau1))``
  is ``2 * FMAX``, which itself overflows to ``inf``, while ``i1(-tau1)`` clamps
  to ``-inf`` making the numerator ``inf``.  ``inf / inf`` is NaN, and the NaN
  propagates through the whole KL curve.
* **calibration-side (tau2) overflow -> silently wrong, and sign-broken.**  No
  NaN appears; ``log(FMAX / i0(tau1))`` is simply far too small.  The result
  goes *negative*, which a KL divergence cannot be, so the argmin is pulled
  toward the highest calibration dimensions -- a systematic overestimate rather
  than a visible crash.

The rewritten path uses ``i0e``/``i1e`` and the identities

    log(i0(t2)/i0(t1)) = (t2 - t1) + log(i0e(t2)/i0e(t1))
    i1(t)/i0(t)        = i1e(t)/i0e(t)

so no intermediate ever overflows.
"""

import numpy as np
import pytest
import mpmath as mp
from scipy.special import i0, i1

from danco._kernels.vonmises import kl_vonmises, kl_vonmises_strip
from danco.spline import load_splines

# Every value is below the overflow point, so this whole grid must agree.
TAUS = [0.1, 1, 10, 50, 100, 200, 280, 283, 300, 400]
NUS = [0.0, 0.5, 1.2568, np.pi / 2, np.pi, -1.0]

ABS_TOL = 1e-10


def _overflow_threshold():
    """Bisect the exact tau where i0/i1 stop being representable."""
    lo, hi = 1.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.isinf(i0(mid)) or np.isinf(i1(mid)):
            hi = mid
        else:
            lo = mid
    return lo


THRESHOLD = _overflow_threshold()


def test_threshold_is_where_double_range_ends():
    """Sanity-check the bisection: ~709.78, i.e. log(DBL_MAX) plus the 1/sqrt(2 pi t)
    prefactor of the asymptotic i0."""
    assert 709.0 < THRESHOLD < 710.0
    assert np.isfinite(i0(THRESHOLD)) and np.isfinite(i1(THRESHOLD))
    assert np.isinf(i0(np.nextafter(THRESHOLD, np.inf)))


@pytest.mark.parametrize("tau1", TAUS)
def test_standard_kl_agrees_below_threshold(tau1):
    worst = 0.0
    for tau2 in TAUS:
        for nu1 in NUS:
            for nu2 in NUS:
                old = kl_vonmises(nu1, tau1, nu2, tau2, use_legacy_bessel=True)
                new = kl_vonmises(nu1, tau1, nu2, tau2, use_legacy_bessel=False)
                assert np.isfinite(old) and np.isfinite(new)
                worst = max(worst, abs(old - new))
    assert worst < ABS_TOL, f"tau1={tau1}: worst |delta| = {worst:.3e}"


@pytest.mark.parametrize("tau1", TAUS)
def test_strip_kl_agrees_below_threshold(tau1):
    worst = 0.0
    for tau2 in TAUS:
        old = kl_vonmises_strip(tau1, tau2, use_legacy_bessel=True)
        new = kl_vonmises_strip(tau1, tau2, use_legacy_bessel=False)
        assert np.isfinite(old) and np.isfinite(new)
        worst = max(worst, abs(old - new))
    assert worst < ABS_TOL, f"tau1={tau1}: worst |delta| = {worst:.3e}"


NEGATIVE_TAUS = [-0.5, -2.0, -10.0, -100.0, -400.0]


@pytest.mark.parametrize("tau1", NEGATIVE_TAUS)
def test_identity_holds_for_negative_tau(tau1):
    """The scaling exponent is ``|x|``, not ``x``.

    ``tau`` is a concentration and should never be negative -- but ``ml_vm``'s
    third Fisher branch, ``1 / (eta^3 - 4 eta^2 + 3 eta)``, is singular at
    ``eta = 1``, and a perfectly concentrated row can round just past it.  If
    the scaled form dropped the absolute values it would silently disagree with
    the unscaled one by exactly ``2 * tau1`` on those rows.
    """
    for tau2 in [0.5, 3.0, 50.0, 400.0] + NEGATIVE_TAUS:
        for nu1 in NUS[:3]:
            for nu2 in NUS[:3]:
                old = kl_vonmises(nu1, tau1, nu2, tau2, use_legacy_bessel=True)
                new = kl_vonmises(nu1, tau1, nu2, tau2, use_legacy_bessel=False)
                assert abs(old - new) < ABS_TOL, (
                    f"tau1={tau1}, tau2={tau2}: {old} vs {new}"
                )
        old = kl_vonmises_strip(tau1, tau2, use_legacy_bessel=True)
        new = kl_vonmises_strip(tau1, tau2, use_legacy_bessel=False)
        assert abs(old - new) < ABS_TOL, f"strip tau1={tau1}, tau2={tau2}"


def test_ml_vm_third_branch_is_singular_at_eta_one():
    """Documents the upstream source of a negative/infinite tau."""
    from danco._kernels.vonmises import ml_vm

    thetas = np.full((3, 45), 0.7)  # every angle identical -> eta == 1
    with np.errstate(divide="ignore"):
        _, tau = ml_vm(thetas)
    assert np.all(np.isinf(tau))

    # eta is sqrt(mean(cos)^2 + mean(sin)^2), mathematically <= 1; rounding can
    # place it a few ulps above, and the branch then flips sign.
    for eta in (1 + 1e-15, 1 + 1e-12, 1 + 1e-9):
        assert 1 / (eta ** 3 - 4 * eta ** 2 + 3 * eta) < 0


def test_legacy_goes_nan_when_data_tau_overflows():
    """tau1 above the threshold: legacy NaN, new finite and non-negative."""
    below = np.nextafter(THRESHOLD, -np.inf)
    above = np.nextafter(THRESHOLD, np.inf)
    with np.errstate(invalid="ignore", over="ignore"):
        assert np.isfinite(kl_vonmises(0.0, below, 0.0, 500.0, use_legacy_bessel=True))
        for tau1 in (above, 710.0, 800.0, 1500.0, 5000.0):
            old = kl_vonmises(0.0, tau1, 0.0, 500.0, use_legacy_bessel=True)
            new = kl_vonmises(0.0, tau1, 0.0, 500.0, use_legacy_bessel=False)
            assert np.isnan(old), f"expected legacy NaN at tau1={tau1}, got {old}"
            assert np.isfinite(new) and new >= 0.0


def test_legacy_goes_nan_when_data_tau_overflows_strip():
    below = np.nextafter(THRESHOLD, -np.inf)
    above = np.nextafter(THRESHOLD, np.inf)
    with np.errstate(invalid="ignore", over="ignore"):
        assert np.isfinite(kl_vonmises_strip(below, 500.0, use_legacy_bessel=True))
        for tau1 in (above, 710.0, 800.0, 1500.0, 5000.0):
            old = kl_vonmises_strip(tau1, 500.0, use_legacy_bessel=True)
            new = kl_vonmises_strip(tau1, 500.0, use_legacy_bessel=False)
            assert np.isnan(old), f"expected legacy NaN at tau1={tau1}, got {old}"
            assert np.isfinite(new) and new >= 0.0


def test_legacy_goes_negative_when_calibration_tau_overflows():
    """tau2 above the threshold: no NaN, just a wrong -- and eventually negative
    -- divergence.  This is the dangerous mode: it biases the argmin instead of
    announcing itself."""
    with np.errstate(invalid="ignore", over="ignore"):
        for tau2 in (800.0, 1000.0, 2000.0):
            old = kl_vonmises(0.0, 500.0, 0.0, tau2, use_legacy_bessel=True)
            new = kl_vonmises(0.0, 500.0, 0.0, tau2, use_legacy_bessel=False)
            assert np.isfinite(old)
            assert old < 0.0, f"expected a negative (invalid) KL at tau2={tau2}"
            assert new > 0.0 and np.isfinite(new)


@pytest.mark.parametrize("tau1,tau2", [(300.0, 800.0), (800.0, 1500.0),
                                        (1500.0, 5000.0)])
def test_scaled_kl_matches_high_precision_oracle_above_overflow(tau1, tau2):
    """The stable identity agrees with an 80-digit direct Bessel evaluation."""
    mp.mp.dps = 80
    nu1, nu2 = mp.mpf("0.7"), mp.mpf("1.1")
    t1, t2 = mp.mpf(str(tau1)), mp.mpf(str(tau2))
    oracle = (
        mp.log(mp.besseli(0, t2) / mp.besseli(0, t1))
        + mp.besseli(1, t1) / mp.besseli(0, t1)
        * (t1 - t2 * mp.cos(nu1 - nu2))
    )
    stable = kl_vonmises(float(nu1), tau1, float(nu2), tau2)

    assert np.isclose(stable, float(oracle), rtol=2e-13, atol=2e-13)


def test_first_affected_calibration_dimension():
    """Locate where the unscaled evaluation first fails in practice.

    The calibration concentration grows roughly linearly with dimension, so the
    threshold maps to a dimension.  Below it the two evaluations agree; at and
    above it the unscaled angle KL is NaN or sign-broken.
    """
    splines = load_splines()
    report = {}
    for N in splines["N_range"]:
        taus = np.array(
            [float(splines["spline_mu_tau"](d, N)[0, 0]) for d in range(1, 401)]
        )
        crossing = int(np.argmax(taus > THRESHOLD)) + 1
        report[N] = crossing
        # Nothing at or below maxDim=150 can be hit.
        assert crossing > 150

    print(f"\nBessel overflow threshold: tau > {THRESHOLD:.10f}")
    print("first calibration dimension whose mu_tau exceeds it, per N:")
    for N, d in sorted(report.items()):
        print(f"  N={N:>5}: d={d}")
