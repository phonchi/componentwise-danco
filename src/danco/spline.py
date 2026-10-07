"""Pre-trained calibration splines for the ``_fast`` versions.

Two surfaces ship inside the package, one per distance strategy, and each
serves both objectives (Full and Profiled) of that strategy:

* ``assets/fastdanco_splines_400.pkl`` (MiND), used by ``'DANCo_fast'`` and
  ``'DANCo_nustrip_fast'``;
* ``assets/fastdanco_splines_gride_400.pkl`` (Gride), used by
  ``'DANCo_gride_fast'`` and ``'DANCo_gride_nustrip_fast'``.

Both were fitted with ``k = 10`` on a ``RectBivariateSpline`` grid of
``d in 1..400`` by ``N in {450, 500, 580, 640, 700}``.  Because
:class:`scipy.interpolate.RectBivariateSpline` extrapolates silently outside
its grid, :func:`spline_statistic` range-checks by default and raises
``ValueError`` instead.  The check is a pure gate: inside the fitted grid the
returned numbers are bit-identical to the unchecked path.  :func:`load_splines`
also refuses a surface trained with a different distance strategy or a
different neighbourhood size ``k``.
"""

import pickle
from functools import lru_cache
from importlib import resources

__all__ = [
    "DEFAULT_SPLINE_NAME",
    "DEFAULT_SPLINE_NAMES",
    "default_spline_path",
    "load_splines",
    "check_spline_range",
    "spline_statistic",
    "grow_spline_calibration",
]

DEFAULT_SPLINE_NAME = "fastdanco_splines_400.pkl"

# One surface per distance strategy.  A surface stores that strategy's
# calibration ``dhat``, so mixing them would compare two different distance
# models; ``load_splines`` refuses the mismatch.
DEFAULT_SPLINE_NAMES = {
    "mind": "fastdanco_splines_400.pkl",
    "gride": "fastdanco_splines_gride_400.pkl",
}


def default_spline_path(variant="mind"):
    """Filesystem path of the spline pickle shipped inside the package."""
    return str(resources.files("danco.assets") / DEFAULT_SPLINE_NAMES[variant])


@lru_cache(maxsize=4)
def _load_cached(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_splines(spline_file=None, variant="mind", k=None):
    """Load a spline pickle; ``None`` selects the packaged default for ``variant``.

    The result is cached per path.  ``RectBivariateSpline`` objects are stateless
    under evaluation, so sharing one dict between estimators is safe.

    A pickle records the distance strategy it was trained with under its
    ``'variant'`` key; pickles without the key predate it and are MiND surfaces.
    A mismatch with ``variant`` raises ``ValueError`` -- the surface's ``dhat``
    would be fed into the other model's KL.

    When ``k`` is given, the pickle's ``'k'`` entry (the neighbourhood size the
    surface was trained with) must equal it; otherwise ``ValueError`` is raised,
    because a reference built with a different ``k`` is not comparable.
    """
    if spline_file is None:
        spline_file = default_spline_path(variant)
    splines = _load_cached(str(spline_file))
    trained_variant = splines.get("variant", "mind")
    if trained_variant != variant:
        raise ValueError(
            f"spline file {spline_file} stores {trained_variant!r} calibration "
            f"statistics, but the estimator's distance strategy is {variant!r}. "
            "Feeding one model's dhat into the other's KL compares two "
            "different distance models; retrain the surface with "
            "experiments/train_splines.py --variant " + variant + "."
        )
    if k is not None:
        trained_k = splines.get("k")
        if trained_k != k:
            raise ValueError(
                f"Spline trained with k={trained_k}, requested k={k}; "
                "retrain for the requested k."
            )
    return splines


_FAST_VERSIONS = {
    "mind": ("DANCo_fast", "DANCo", "DANCo_nustrip"),
    "gride": ("DANCo_gride_fast", "DANCo_gride", "DANCo_gride_nustrip"),
}


def check_spline_range(splines, d, N):
    """Raise ``ValueError`` if ``(d, N)`` falls outside the fitted grid."""
    fast, slow, slow_profiled = _FAST_VERSIONS.get(
        splines.get("variant", "mind"), _FAST_VERSIONS["mind"]
    )
    d_range = splines["d_range"]
    N_range = splines["N_range"]
    d_lo, d_hi = min(d_range), max(d_range)
    N_lo, N_hi = min(N_range), max(N_range)
    if not (N_lo <= N <= N_hi):
        raise ValueError(
            f"{fast}: N={N} is outside the range the splines were fitted on "
            f"([{N_lo}, {N_hi}]). RectBivariateSpline would extrapolate silently. "
            f"Use version='{slow}' or '{slow_profiled}' (Monte Carlo "
            f"calibration) for this N, or retrain the splines with an N grid "
            f"that covers it."
        )
    if not (d_lo <= d <= d_hi):
        raise ValueError(
            f"{fast}: dimension d={d} is outside the range the splines were "
            f"fitted on ([{d_lo}, {d_hi}]). Lower maxDim, or retrain the splines "
            f"over a wider d grid."
        )


def spline_statistic(splines, d, N, check_range=True):
    """Calibration statistic ``dict(dhat, mu_nu, mu_tau)`` at ``(d, N)``."""
    if check_range:
        check_spline_range(splines, d, N)
    return {
        "dhat": float(splines["spline_dhat"](d, N)[0, 0]),
        "mu_nu": float(splines["spline_mu_nu"](d, N)[0, 0]),
        "mu_tau": float(splines["spline_mu_tau"](d, N)[0, 0]),
    }


def grow_spline_calibration(calibration_statistic, maxDim, N,
                            spline_file=None, check_range=True,
                            variant="mind", k=None):
    """Extend ``calibration_statistic`` up to ``maxDim`` from the splines.

    Mirrors the Monte-Carlo growth loop in :meth:`danco.base.DANCoBase.evaluate`:
    only the missing dimensions are appended, and the cache keeps growing across
    calls on the same estimator.  ``k`` is forwarded to :func:`load_splines`.
    """
    splines = load_splines(spline_file, variant=variant, k=k)
    while calibration_statistic["maxdim"] < maxDim:
        new_dim = calibration_statistic["maxdim"] + 1
        calibration_statistic["statistics"].append(
            spline_statistic(splines, new_dim, N, check_range=check_range)
        )
        calibration_statistic["maxdim"] = new_dim
    return calibration_statistic
