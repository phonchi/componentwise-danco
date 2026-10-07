"""Legacy TwoNN intrinsic-dimension estimator (Facco et al., 2017)."""


import numpy as np

from .backends import get_backend

__all__ = ["two_nn_legacy"]


def two_nn_legacy(X, n_jobs=1, backend="cpu"):
    """Bias-corrected full-sample TWO-NN estimator ``(N - 1) / sum_i log mu_i``.

    ``mu_i = r_2(i) / r_1(i)``.  Not used in the paper; the paper's TWO-NN is
    ``experiments/paper_methods.paper_twonn`` (scikit-dimension ``TwoNN`` with
    ``discard_fraction=0.1``).  Retained as an explicit compatibility estimator.

    Pairs with ``r_1 = 0`` (duplicate observations) or a non-finite log
    ratio are dropped before summing; on data without duplicates the result
    is unchanged.
    """
    dists, _ = get_backend(backend).knn(X, 1, n_jobs=n_jobs)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mus = np.log(dists[:, 1] / dists[:, 0])
    log_mus = log_mus[np.isfinite(log_mus)]
    if log_mus.size < 2:
        return float("nan")
    return (log_mus.size - 1) / np.sum(log_mus)
