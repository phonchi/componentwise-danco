"""Candidate-wise angular-location penalty on the focal GSM level.

Main Section 3.1 separates the von Mises divergence into a concentration term
and the nonnegative angular-location penalty
``A(tau_hat) tau_ref_m {1 - cos(nu_ref_m - nu_hat)}``.  This script measures
that penalty across the candidate range on the focal Gaussian scale-mixture
level (d = 70 embedded in D = 100, sigma_S = .25), using the same Monte Carlo
reference construction as the experiment, so the manuscript can state how the
penalty behaves instead of assuming monotonicity.

Writes results/diagnostics/gsm_penalty_curve.csv.
"""
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from scipy.special import i0e, i1e

from common import REPO_ROOT
from danco import DANCo

OUT = REPO_ROOT / "results" / "diagnostics" / "gsm_penalty_curve.csv"


def main(argv=None):
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]).parse_args(argv)
    n, d, ambient, k, max_dim, reps = 2500, 70, 100, 10, 100, 5
    sigma_s, first = 0.25, 6
    rows = []
    for rep in range(reps):
        generator = np.random.default_rng(rep)
        latent = generator.standard_normal((n, d))
        scales = np.exp(sigma_s * generator.standard_normal(n))
        observed = np.zeros((n, ambient))
        observed[:, :d] = scales[:, None] * latent

        estimator = DANCo(N=n, k=k, version="DANCo", random_state=700 + rep,
                          calibration_seeding="stream")
        stats = estimator.evaluate_all(observed, maxDim=max_dim)
        reference = stats["calibration_statistic"]["statistics"]
        nu_ref = np.array([r["mu_nu"] for r in reference])
        tau_ref = np.array([r["mu_tau"] for r in reference])

        amplitude = i1e(stats["mu_tau"]) / i0e(stats["mu_tau"])
        penalty = amplitude * tau_ref * (1.0 - np.cos(nu_ref - stats["mu_nu"]))
        candidates = np.arange(1, max_dim + 1)
        valid = candidates >= first
        steps = np.diff(penalty[valid])
        rows.append(dict(
            rep=rep, sigma_s=sigma_s, d=d, ambient_d=ambient, n=n, k=k,
            first_candidate=first, max_dim=max_dim,
            nu_hat=float(stats["mu_nu"]), tau_hat=float(stats["mu_tau"]),
            full=float(stats["de_standard"]), profiled=float(stats["de_nustrip"]),
            penalty_first=float(penalty[valid][0]),
            penalty_last=float(penalty[valid][-1]),
            increasing_steps=int((steps > 0).sum()), total_steps=int(steps.size),
            references_below_observed=int((nu_ref[valid] < stats["mu_nu"]).sum()),
            valid_candidates=int(valid.sum()),
        ))
        print(f"rep {rep}: penalty {rows[-1]['penalty_first']:.2f} -> "
              f"{rows[-1]['penalty_last']:.2f} nats, "
              f"{rows[-1]['increasing_steps']}/{rows[-1]['total_steps']} steps up")

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame[["penalty_first", "penalty_last", "increasing_steps",
                 "references_below_observed"]].describe().loc[
        ["mean", "min", "max"]].round(2).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
