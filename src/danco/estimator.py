"""Public estimator API."""

from .base import VERSIONS, DANCoBase

__all__ = ["DANCo", "VERSIONS"]


class DANCo(DANCoBase):
    """DANCo intrinsic-dimension estimator.

    ``version`` names the objective that :meth:`evaluate` returns.  Four
    basic versions calibrate against Monte Carlo hyperballs drawn at the
    observed sample size:

    ==========  =======================  ==============================
    distance    Full                     Profiled
    ==========  =======================  ==============================
    MiND        ``'DANCo'``              ``'DANCo_nustrip'``
    Gride       ``'DANCo_gride'``        ``'DANCo_gride_nustrip'``
    ==========  =======================  ==============================

    Appending ``_fast`` to any of them (``'DANCo_fast'``,
    ``'DANCo_nustrip_fast'``, ``'DANCo_gride_fast'``,
    ``'DANCo_gride_nustrip_fast'``) replaces the Monte Carlo references by the
    packaged spline surface of that distance statistic.

    :meth:`evaluate` computes only the selected objective and returns
    ``(dimension, stats)``; :meth:`evaluate_all` computes Full and Profiled of
    the same distance statistic in one pass (``stats['de_standard']`` and
    ``stats['de_nustrip']``), which is what the paper's experiments use.

    Each spline surface stores the calibration statistics of *one* distance
    strategy, recorded in its ``'variant'`` metadata, so the Gride ``_fast``
    versions require a Gride-trained surface
    (``experiments/train_splines.py --variant gride``).  Loading a surface whose
    variant does not match the estimator's raises ``ValueError``: feeding a
    MiND ``dhat`` into the Gride KL would compare two different distance models.
    The surface's trained ``k`` must also equal the estimator's ``k`` (both
    packaged surfaces use ``k = 10``); a mismatch raises ``ValueError``.

    Examples
    --------
    >>> import numpy as np
    >>> from danco import DANCo
    >>> X = np.random.default_rng(0).normal(size=(500, 10))[:, :5] @ np.eye(5, 10)
    >>> est = DANCo(N=500, k=10, random_state=35)
    >>> dim, stats = est.evaluate(X, maxDim=20)      # doctest: +SKIP

    Notes
    -----
    The estimator is **stateful by design**: its calibration cache grows across
    ``evaluate`` calls and is never recomputed for a dimension it already holds.
    The cache only extends.  Candidate ``m`` always consumes the ``m``-th draw
    of the estimator's RNG, so a reference replicate is a deterministic
    function of ``(random_state, N)`` and does not depend on which datasets
    were evaluated first.  ``calibration_seeding='derived'`` additionally makes
    each dimension's draw independent of the others.
    """
