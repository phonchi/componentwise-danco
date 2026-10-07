"""MiND_ML distance strategy -- the original DANCo distance model.

Binds the two choices that always travel together:
  (a) the dimension estimate from ``rho_i = r_1(i) / r_(k+1)(i)``, and
  (b) the closed-form KL between two MiND_ML densities.
"""

from .._kernels import distances

__all__ = ["MiNDVariant"]


class MiNDVariant:
    name = "mind"

    def __init__(self, k, N):
        self.k = k
        self.N = N

    def ratios(self, dists):
        """``r_1 / r_(k+1)`` per point."""
        return dists[:, 0] / dists[:, -1]

    def estimate_dimension(self, dists, D):
        rhos = self.ratios(dists)
        dhat_0 = distances.mind_mli(rhos, D, self.N, self.k)
        return distances.mind_mlk(rhos, dhat_0, D, self.k)

    def kl_distance(self, dhat, dcal):
        return distances.kl_mind(dhat, dcal, self.k)
