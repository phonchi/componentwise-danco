"""Gride distance strategy.

Binds the pair:
  (a) the dimension estimate from ``rho_i = r_n2(i) / r_n1(i)`` by bisecting the
      published generic-order log-likelihood score, and
  (b) the Gride KL.  The archived estimator evaluates the cross term by
      quadrature; ``gride_reference_log_moment`` implements Supplementary
      Eq. (S1.9), and ``kl_gride_closed_form`` implements Eq. (S1.10).

``n2`` is ``k`` rounded up to the next even number and ``n1 = n2 // 2``, matching
the original neighbor-order convention: for the default ``k = 10`` that is
``n1 = 5, n2 = 10``.  The estimator uses the published ``N / d`` score term
rather than ``(N - 1) / d``.
"""

from .._kernels import distances

__all__ = ["GrideVariant"]


class GrideVariant:
    name = "gride"

    def __init__(self, k, N, d0=0.001, d1=1000.0, eps=1e-7):
        self.k = k
        self.N = N
        self.n2 = int(k + k % 2)  # force even
        self.n1 = self.n2 // 2
        self.d0 = d0
        self.d1 = d1
        self.eps = eps

    def ratios(self, dists):
        """``r_n2 / r_n1`` per point."""
        return dists[:, self.n2 - 1] / dists[:, self.n1 - 1]

    def estimate_dimension(self, dists, D):
        rho = self.ratios(dists)
        # NOTE: argmax_loglik mutates `rho` in place (see distances.filter_mus);
        # `rho` is a fresh array here, so nothing outside is affected.
        return distances.argmax_loglik(
            self.d0, self.d1, rho, self.eps, self.N, self.n1, self.n2
        )

    def kl_distance(self, dhat, dcal):
        return distances.kl_gride(dhat, dcal, self.n1, self.n2)
