"""CUDA backend for chunked neighbour search and angle reduction.

Same two functions as :mod:`danco.backends.cpu`, so ``danco.base`` does not know
which one it is talking to.  Both are chunked, and both retry with a halved
batch on CUDA OOM, so large ambient dimensions (for example ``D = 150528`` for
a raw ``224x224x3`` image) fit on a modest GPU.

Behaviours inherited from the earlier GPU implementation
---------------------------------------------------------
The earlier GPU path was not a drop-in replacement for the CPU class; it
re-implemented ``evaluate`` and picked up three behaviours along the way.  Only
the first is a default here; the other two are opt-in:

* **float32 throughout.**  Distances come from the expanded form
  ``|q|^2 + |r|^2 - 2 q.r`` in single precision.  That form is what makes a
  single ``matmul`` possible, and it is also the whole reason the backend is
  fast; ``dtype='float64'`` is available when accuracy matters more.
* **A duplicate filter.**  With ``dup_threshold`` set (the original
  implementation used 1e-4), any candidate closer than it is pushed to ``inf`` before the
  ``topk``, so coincident points never become neighbours.  This is the GPU path's answer to the CPU
  class's ``_clean_zero_nb``, and it is not the same answer: ``_clean_zero_nb``
  only repairs points whose *nearest* neighbour is below 1e-5, whereas this
  drops near-duplicates for every point.  On duplicate-free data (hyperballs,
  synthetic manifolds) neither fires and the two backends agree.  The default
  ``dup_threshold=None`` switches it off, matching the CPU backend and the
  paper runs, which remove exact duplicate rows upstream.
* **An off-by-one in the neighbour count** -- see ``extra_neighbors``.  This one
  is *not* the default, because it would silently change the estimator.

The CPU/GPU split for calibration is decided in :mod:`danco.base`: Monte-Carlo
hyperballs stay on the CPU regardless of this backend, because they are small
and the transfer dominates.  The spline-calibrated versions never reach that
path at all, because their calibration is a table lookup.
"""

import gc

import numpy as np

__all__ = ["make_backend", "GpuBackend"]


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "backend='gpu' needs PyTorch; install it (see pytorch.org) or use "
            "backend='cpu'"
        ) from exc
    return torch


def make_backend(**options):
    """Factory used by :func:`danco.backends.get_backend`."""
    return GpuBackend(**options)


class GpuBackend:
    """Neighbour search and angle reduction on CUDA.

    Parameters
    ----------
    query_batch : int
        Query rows per distance chunk (default 128).  Halved automatically on
        OOM.
    angle_batch : int
        Points per ``bmm`` chunk in the angle reduction (default 4).  Small
        because the Gram batch is ``angle_batch x k x k`` on top of
        an ``angle_batch x k x D`` gather, and ``D`` is the large axis.
    dup_threshold : float or None
        Candidates closer than this are excluded from the neighbour list.
        ``None`` (default) disables the filter and makes the backend a pure
        kNN, matching ``backends.cpu`` on data with duplicates; ``1e-4`` was
        the earlier implementation's setting.
    extra_neighbors : int
        How many neighbours beyond ``k`` to return.  The backend contract -- and
        every CPU code path -- is ``1``: MiND_ML's ratio is
        ``r_1 / r_(k+1)``, so the estimator needs ``k + 1`` columns while the
        angle kernel uses only the first ``k``.

        The earlier GPU path returned only ``k`` columns and formed the ratio
        ``r_1 / r_k`` -- an off-by-one against the CPU class it was meant to
        accelerate, and one that biases ``dhat`` upward because a larger
        nearest-to-outer-neighbor ratio corresponds to a higher fitted
        dimension.  ``extra_neighbors=0`` reproduces that behaviour; do not use
        it for new results.
    device : str or None
        Defaults to ``'cuda'``.  There is no silent CPU fallback: if CUDA is
        unavailable this raises, because a run that quietly took the slow path
        is worse than one that stopped.  ``device='cpu'`` is accepted and does
        run (torch on CPU), which is useful only for debugging.
    dtype : str
        ``'float32'`` (the fast path) or ``'float64'``.
    """

    name = "gpu"

    def __init__(self, query_batch=128, angle_batch=4, dup_threshold=None,
                 extra_neighbors=1, device=None, dtype="float32"):
        torch = _import_torch()
        self._torch = torch
        if query_batch < 1 or angle_batch < 1:
            raise ValueError("query_batch and angle_batch must be >= 1")
        if extra_neighbors < 0:
            raise ValueError("extra_neighbors must be >= 0")
        if dtype not in ("float32", "float64"):
            raise ValueError(
                f"dtype should be 'float32' or 'float64', got {dtype!r}"
            )
        self.query_batch = int(query_batch)
        self.angle_batch = int(angle_batch)
        self.dup_threshold = dup_threshold
        self.extra_neighbors = int(extra_neighbors)
        self.dtype = dtype
        if device is None:
            device = "cuda"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "backend='gpu' requires CUDA, but torch.cuda.is_available() is "
                "False. Check the driver/toolkit (torch was built for "
                f"CUDA {getattr(torch.version, 'cuda', None)}), or use "
                "backend='cpu'. Passing backend_options={'device': 'cpu'} runs "
                "these kernels through torch on the CPU, which is slower than "
                "the 'cpu' backend and is meant for debugging only."
            )
        self.device = torch.device(device)
        self._np_dtype = np.float32 if dtype == "float32" else np.float64
        self._torch_dtype = (
            torch.float32 if dtype == "float32" else torch.float64
        )

    # ------------------------------------------------------------------
    def _free(self):
        gc.collect()
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()

    def _as_tensor(self, X):
        arr = np.asarray(X, dtype=self._np_dtype, order="C")
        return self._torch.as_tensor(arr, device=self.device)

    # ------------------------------------------------------------------
    def knn(self, X, k, n_jobs=1):
        """Distances and indices of the ``k + extra_neighbors`` neighbours.

        Self is excluded (the diagonal is forced to ``inf``), so column ``0`` is
        the first neighbour.  Returns ``float64`` distances and ``int64``
        indices, exactly like the CPU backend -- everything downstream is numpy.

        ``n_jobs`` is accepted and ignored; the parallelism here is the device.
        """
        torch = self._torch
        n_out = k + self.extra_neighbors
        X_t = self._as_tensor(X)
        n = X_t.shape[0]
        if n <= n_out:
            raise ValueError(
                f"need more than k+{self.extra_neighbors}={n_out} points for "
                f"kNN, got n={n}"
            )
        try:
            ref2 = (X_t * X_t).sum(dim=1, keepdim=True).T
            batch = self.query_batch
            while True:
                try:
                    out_d, out_i = [], []
                    for start in range(0, n, batch):
                        stop = min(start + batch, n)
                        q = X_t[start:stop]
                        q2 = (q * q).sum(dim=1, keepdim=True)
                        dist2 = torch.clamp(
                            q2 + ref2 - 2.0 * (q @ X_t.T), min=0.0
                        )
                        rows = torch.arange(stop - start, device=self.device)
                        cols = torch.arange(start, stop, device=self.device)
                        dist2[rows, cols] = float("inf")
                        if self.dup_threshold is not None:
                            dist2 = torch.where(
                                dist2 < self.dup_threshold ** 2,
                                float("inf"), dist2,
                            )
                        vals, inds = torch.topk(
                            dist2, k=n_out, dim=1, largest=False, sorted=True
                        )
                        out_d.append(torch.sqrt(vals).cpu())
                        out_i.append(inds.cpu())
                    break
                except RuntimeError as exc:
                    if batch <= 1 or "out of memory" not in str(exc).lower():
                        raise
                    self._free()
                    batch = max(1, batch // 2)
            dists = np.ascontiguousarray(
                torch.cat(out_d, dim=0).numpy(), dtype=np.float64
            )
            idx = np.ascontiguousarray(
                torch.cat(out_i, dim=0).numpy(), dtype=np.int64
            )
        finally:
            del X_t
            self._free()
        if not np.all(np.isfinite(dists)):
            raise RuntimeError(
                "kNN produced non-finite distances: fewer than "
                f"{n_out} candidates survived the duplicate filter "
                f"(dup_threshold={self.dup_threshold}). Lower or disable it "
                "with backend_options={'dup_threshold': None}."
            )
        return dists, idx

    # ------------------------------------------------------------------
    def angle_stats(self, X, idx, combs, k):
        """Pairwise angles between displacements to the ``k`` neighbours.

        Matches :func:`danco.backends.cpu.angle_stats`: shape
        ``(n_samples, k*(k-1)/2)``, column order given by ``combs``, and a
        zero-length displacement yields ``pi/2``.  The CPU kernel gets there by
        rewriting a NaN cosine to 0; here the denominator is clamped away from
        zero while the numerator stays 0, which lands on the same value without
        a branch.
        """
        torch = self._torch
        nbs = np.ascontiguousarray(np.asarray(idx)[:, :k], dtype=np.int64)
        combs = np.asarray(combs)
        X_t = self._as_tensor(X)
        n = nbs.shape[0]
        try:
            idx_t = torch.as_tensor(nbs, device=self.device)
            left = torch.as_tensor(
                np.ascontiguousarray(combs[0]), device=self.device
            )
            right = torch.as_tensor(
                np.ascontiguousarray(combs[1]), device=self.device
            )
            batch = self.angle_batch
            while True:
                try:
                    pieces = []
                    for start in range(0, n, batch):
                        stop = min(start + batch, n)
                        base = X_t[start:stop]
                        vec = X_t[idx_t[start:stop]] - base[:, None, :]
                        gram = torch.bmm(vec, vec.transpose(1, 2))
                        norms = torch.sqrt(
                            torch.clamp(
                                torch.diagonal(gram, dim1=1, dim2=2), min=1e-24
                            )
                        )
                        denom = torch.clamp(
                            norms[:, :, None] * norms[:, None, :], min=1e-24
                        )
                        cos = torch.clamp(gram / denom, -1.0, 1.0)
                        pieces.append(
                            torch.arccos(cos[:, left, right]).cpu()
                        )
                    break
                except RuntimeError as exc:
                    if batch <= 1 or "out of memory" not in str(exc).lower():
                        raise
                    self._free()
                    batch = max(1, batch // 2)
            thetas = torch.cat(pieces, dim=0).numpy().astype(np.float64)
        finally:
            del X_t
            self._free()
        return thetas
