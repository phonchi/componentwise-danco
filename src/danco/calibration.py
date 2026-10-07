"""Calibration-sample generation.

DANCo calibrates against uniformly-sampled hyperballs of known dimension.  The
growth loop itself lives in :meth:`danco.base.DANCoBase.evaluate` because it
needs the estimator's own statistic function; only the sampler is here.
"""

import numpy as np
from scipy.special import gammainc

__all__ = ["empty_calibration", "hyper_ball"]


def empty_calibration():
    """A fresh, empty calibration cache."""
    return dict(statistics=list(), maxdim=0)


def hyper_ball(random_state, n, d, radius=1.0, center=[]):
    """``n`` points drawn uniformly from the ``d``-ball of the given radius.

    Consumes **exactly one** draw from ``random_state`` --
    ``normal(size=(n, d))`` -- and the RNG stream position is part of the
    reproduction contract: adding, removing or reordering a draw anywhere in the
    calibration loop changes every downstream number.
    """
    if center == []:
        center = np.array([0] * d)
    x = random_state.normal(size=(n, d))
    ssq = np.sum(x ** 2, axis=1)
    fr = radius * gammainc(d / 2, ssq / 2) ** (1 / d) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, d))
    return center + np.multiply(x, frtiled)
