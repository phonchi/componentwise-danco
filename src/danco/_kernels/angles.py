"""Pairwise-angle kernel shared by every DANCo variant.

The numba kernel is a single copy of what the earlier implementation
duplicated across its estimator classes, in two flavours: with and without a
NaN guard for zero-length displacement vectors.

The guarded version is canonical here.  The guard only fires when a neighbour
coincides with the query point (``cos_th`` becomes ``0/0 = NaN``); on any input
without duplicate points the two flavours are bit-identical.
"""

import numpy as np
from numba import njit, prange

__all__ = ["danco_angles", "indnComb"]


@njit(parallel=True)
def danco_angles(X, nbs, combs, k=10):
    N = len(X)
    thetas = np.zeros((N, int(k * (k - 1) / 2)))
    for i in prange(N):
        vec = X[nbs[i, :], :] - X[i, :]
        vec_len = np.sqrt(np.sum(vec ** 2, axis=1))
        sc_prod = np.sum(vec[combs[0, :]] * vec[combs[1, :]], axis=1)
        cos_th = sc_prod / (vec_len[combs[0, :]] * vec_len[combs[1, :]])
        # NaN guard: zero-length displacement (e.g. raw image with
        # uniform patches) -> cos_th NaN -> poisons ml_vm.
        # Replace with 0 (= theta=pi/2, neutral angle).
        for j in range(cos_th.shape[0]):
            if np.isnan(cos_th[j]):
                cos_th[j] = 0.0
        if np.any(np.abs(cos_th) > 1):
            np.clip(cos_th, a_min=-1, a_max=1, out=cos_th)
        thetas[i, ] = np.arccos(cos_th)
    return thetas


def indnComb(k):
    """Index pairs (i, j), i < j, of the k nearest neighbours.

    Returns an array of shape ``(2, k*(k-1)/2)``.  Enumeration order is
    load-bearing: ``danco_angles`` indexes ``combs[0]``/``combs[1]`` directly.
    """
    a = []
    for i in range(k - 1):
        for j in range(i + 1, k):
            a.append([i, j])
    return np.array(a).T
