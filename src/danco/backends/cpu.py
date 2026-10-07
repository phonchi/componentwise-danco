"""CPU backend.

The backend surface is deliberately two functions.  Everything that differs
between a CPU and a GPU implementation (neighbour search and the pairwise-angle
reduction) lives behind them; ``danco.base`` never touches sklearn or numba
directly.  The GPU backend provides the same interface.
"""

from sklearn.neighbors import NearestNeighbors

from .._kernels.angles import danco_angles

__all__ = ["knn", "angle_stats"]


def knn(X, k, n_jobs=1):
    """Distances and indices of the ``k+1`` nearest neighbours of each point.

    Self is excluded (sklearn's ``kneighbors()`` with no argument queries the
    training set and drops the trivial self match), so column ``0`` is the
    first neighbour and column ``-1`` the ``(k+1)``-th.  Both arrays have shape
    ``(n_samples, k+1)``.
    """
    neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs)
    neigh.fit(X)
    dists, idx = neigh.kneighbors(return_distance=True)
    return dists, idx


def angle_stats(X, idx, combs, k):
    """Pairwise angles between the displacement vectors to the ``k`` neighbours.

    ``idx`` may carry more than ``k`` columns; only the first ``k`` are used.
    Returns shape ``(n_samples, k*(k-1)/2)``.
    """
    return danco_angles(X, idx[:, :k], combs, k)
