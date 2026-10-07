#!/usr/bin/env python
"""Toy sweeps behind the nu-mechanism figure.

    python experiments/run_nu_toy_sweep.py

Three sweeps, all with the estimator's own angle machinery (k=10):

1. ``ball``: nu-hat of the uniform d-ball across d -- the finite-sample
   calibration curve moving from near pi/2 toward the equal-shell pi/3
   reference line as dimension grows at fixed N.
2. ``gsm``: nu-hat of the Gaussian scale mixture x = s*g,
   log s ~ N(0, sigma_s^2), across the sigma_s grid of the Section 4.4.1 GSM
   sweep (the range anchored by the image-class centered-norm CVs) -- scale
   heterogeneity driving nu-hat through and below pi/3 in this controlled
   simulation.
3. ``gsm_profile``: per-point nu_i binned by centred norm for one mixture
   (sigma_s = 0.25, the focal level of Section 4.4.1), testing the conditional mechanism that larger-amplitude
   centers can have smaller selected-neighbor angles. This is an empirical
   profile, not a theorem about the kNN-selected amplitude distribution.

Output: results/diagnostics/nu_toy_{ball,gsm,gsm_profile}.csv
"""

import argparse

import numpy as np
import pandas as pd

from common import REPO_ROOT
from danco._kernels.angles import indnComb
from danco._kernels.vonmises import ml_vm
from danco.backends import get_backend

K = 10
N = 2500
OUT = REPO_ROOT / "results" / "diagnostics"


def nu_stats(X, backend, combs):
    _, idx = backend.knn(X, K)
    thetas = backend.angle_stats(X, idx, combs, K)
    nu_i, _ = ml_vm(thetas)
    nu = float(np.arctan2(np.sin(nu_i).sum(), np.cos(nu_i).sum()))
    return nu, nu_i


def main(argv=None):
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]).parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    backend = get_backend("cpu")
    combs = indnComb(K)
    rng = np.random.default_rng(0)

    rows = []
    for d in [2, 3, 5, 8, 12, 20, 35, 50, 70, 100, 150, 250, 400]:
        g = rng.standard_normal((N, d))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        X = g * (rng.random(N) ** (1 / d))[:, None]
        nu, _ = nu_stats(X, backend, combs)
        rows.append(dict(d=d, nu=nu))
        print(f"ball d={d}: nu={nu:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nu_toy_ball.csv", index=False)

    rows = []
    for sigma in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        g = rng.standard_normal((N, 70))
        s = np.exp(sigma * rng.standard_normal(N))
        nu, _ = nu_stats(g * s[:, None], backend, combs)
        rows.append(dict(sigma_s=sigma, nu=nu))
        print(f"gsm sigma_s={sigma}: nu={nu:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nu_toy_gsm.csv", index=False)

    g = rng.standard_normal((N, 70))
    s = np.exp(0.25 * rng.standard_normal(N))
    X = g * s[:, None]
    _, nu_i = nu_stats(X, backend, combs)
    norms = np.linalg.norm(X - X.mean(axis=0, keepdims=True), axis=1)
    q = np.quantile(norms, np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        sel = (norms >= q[i]) & (norms <= q[i + 1])
        rows.append(dict(decile=i, norm_mid=float(norms[sel].mean()),
                         nu_mean=float(nu_i[sel].mean()),
                         nu_sd=float(nu_i[sel].std())))
    pd.DataFrame(rows).to_csv(OUT / "nu_toy_gsm_profile.csv", index=False)
    print("wrote nu_toy_{ball,gsm,gsm_profile}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
