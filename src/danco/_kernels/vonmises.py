"""von Mises fitting and KL divergences.

``use_legacy_bessel=True`` reproduces the earlier implementation's expression built on
:func:`scipy.special.i0` / :func:`scipy.special.i1` with a
``min(sys.float_info.max, .)`` clamp.  That expression overflows once the
concentration ``tau`` is large enough for ``i0``/``i1`` to exceed the double
range (around ``tau ~ 7.1e2``), at which point the clamp silently returns a
wrong -- and eventually non-finite -- value.

The default path is algebraically identical but written with the exponentially
scaled variants ``i0e(x) = exp(-|x|) i0(x)`` and ``i1e``, using

    log(i0(t2) / i0(t1)) = (|t2| - |t1|) + log(i0e(t2) / i0e(t1))
    i1(t) / i0(t)        = i1e(t) / i0e(t)

so it never forms an overflowing intermediate.  The absolute values matter: the
scaling exponent is ``|x|``, not ``x``, so a negative concentration would
otherwise shift the result by ``2 t1``.  ``tau`` is a concentration and should
never be negative, but :func:`ml_vm`'s third Fisher branch is singular at
``eta = 1`` and rounding can push a perfectly concentrated row just past it,
producing ``+-inf`` or a large negative ``tau``.  Both identities hold for all
real arguments as written.
"""

import sys

import numpy as np
from scipy.special import i0, i0e, i1, i1e

__all__ = [
    "circular_mean_resultant",
    "ml_vm",
    "kl_vonmises",
    "kl_vonmises_strip",
]

_FMAX = sys.float_info.max


def circular_mean_resultant(angles, axis=None):
    """Return the circular mean direction and resultant magnitude.

    For angles ``theta_r``, the resultant vector is

    ``(C, S) = (sum_r cos(theta_r), sum_r sin(theta_r))``.

    Its magnitude is ``sqrt(C**2 + S**2)`` and its quadrant-aware direction is
    ``arctan2(S, C)``.  The direction is undefined when the magnitude is
    exactly zero; this function reports ``nan`` in that case so callers cannot
    silently treat the arbitrary numerical value ``arctan2(0, 0) == 0`` as an
    observed direction.

    Parameters
    ----------
    angles : array-like
        Angles in radians.
    axis : int or tuple of int, optional
        Axis or axes over which to form the resultant.  The default aggregates
        all angles.

    Returns
    -------
    direction, resultant : ndarray or scalar
        Mean direction in ``(-pi, pi]`` and the unnormalised resultant
        magnitude.  ``direction`` is ``nan`` exactly where ``resultant == 0``.
    """
    angles = np.asarray(angles)
    sine_sum = np.sum(np.sin(angles), axis=axis)
    cosine_sum = np.sum(np.cos(angles), axis=axis)
    resultant = np.hypot(cosine_sum, sine_sum)
    direction = np.arctan2(sine_sum, cosine_sum)
    direction = np.where(direction <= -np.pi, np.pi, direction)
    if np.ndim(direction) == 0:
        if resultant == 0:
            direction = np.float64(np.nan)
        return np.float64(direction), resultant
    direction = np.where(resultant == 0, np.nan, direction)
    return direction, resultant


def ml_vm(thetas, return_resultant=False):
    """Per-point maximum-likelihood von Mises fit.

    ``thetas`` has shape ``(n_points, n_angles)``.  Returns ``(nu, tau)``, both
    of shape ``(n_points,)``.  ``tau`` uses Fisher's three-branch approximation
    to the inverse of ``A(tau) = i1(tau)/i0(tau)``.  With
    ``return_resultant=True``, the per-point resultant magnitude is returned as
    a third array so callers can detect an undefined mean direction.
    """
    sinth = np.sin(thetas)
    costh = np.cos(thetas)
    nu, resultant = circular_mean_resultant(thetas, axis=1)
    eta = np.sqrt(np.mean(costh, axis=1) ** 2 + np.mean(sinth, axis=1) ** 2)
    # NOTE: the earlier implementation aliases `tau = eta` and mutates in
    # place.  Keeping the alias is what makes the branch conditions read the
    # *original* eta values, so it is reproduced verbatim rather than "cleaned
    # up" into a copy.
    tau = eta
    for i in range(len(eta)):
        if eta[i] < 0.53:
            tau[i] = 2 * eta[i] + eta[i] ** 3 + 5 * (eta[i] ** 5) / 6
        elif eta[i] < 0.85:
            tau[i] = -0.4 + 1.39 * eta[i] + 0.43 / (1 - eta[i])
        else:
            tau[i] = 1 / ((eta[i] ** 3) - 4 * (eta[i] ** 2) + 3 * eta[i])
    if return_resultant:
        return nu, tau, resultant
    return nu, tau


def kl_vonmises(nu1, tau1, nu2, tau2, use_legacy_bessel=False):
    """KL( VM(nu1, tau1) || VM(nu2, tau2) )."""
    if use_legacy_bessel:
        return np.log(
            min(_FMAX, i0(tau2)) / min(_FMAX, i0(tau1))
        ) + (min(_FMAX, i1(tau1)) - min(_FMAX, i1(-tau1))) / (
            2 * min(_FMAX, i0(tau1))
        ) * (tau1 - tau2 * np.cos(nu1 - nu2))
    return (
        (abs(tau2) - abs(tau1))
        + np.log(i0e(tau2) / i0e(tau1))
        + i1e(tau1) / i0e(tau1) * (tau1 - tau2 * np.cos(nu1 - nu2))
    )


def kl_vonmises_strip(tau1, tau2, use_legacy_bessel=False):
    """nu-restricted angle KL: :func:`kl_vonmises` with ``cos(nu1 - nu2) = 1``.

    nu is still fitted and reported, but its mismatch is not treated as
    dimension evidence.  ``use_legacy_bessel=True`` evaluates the same
    expression with unscaled Bessel functions; it exists only for tests of
    their overflow behaviour.
    """
    if use_legacy_bessel:
        return np.log(
            min(_FMAX, i0(tau2)) / min(_FMAX, i0(tau1))
        ) + (min(_FMAX, i1(tau1)) - min(_FMAX, i1(-tau1))) / (
            2 * min(_FMAX, i0(tau1))
        ) * (tau1 - tau2)
    return (
        (abs(tau2) - abs(tau1))
        + np.log(i0e(tau2) / i0e(tau1))
        + i1e(tau1) / i0e(tau1) * (tau1 - tau2)
    )
