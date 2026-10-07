"""Single-sample mechanism check for the Gaussian scale mixture.

For one GSM sample (d = 70, N = 2500, seed 0) at each log-amplitude spread,
record how neighbor selection couples the two DANCo channels: the mean
amplitude of the ten selected neighbors relative to the center amplitude, the
mean first-to-eleventh neighbor distance ratio rho = r_1 / r_11, and the
Pearson correlation between the per-center ratio and the center log-amplitude.
Writes results/diagnostics/gsm_selection_check.csv.

Supporting check, not cited in the paper.
"""
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import REPO_ROOT

OUT = REPO_ROOT / "results" / "diagnostics" / "gsm_selection_check.csv"


def main(argv=None):
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]).parse_args(argv)
    n, d, k = 2500, 70, 10
    rows = []
    for sigma_s in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75):
        generator = np.random.default_rng(0)
        latent = generator.standard_normal((n, d))
        scales = np.ones(n)
        if sigma_s != 0:
            scales = np.exp(sigma_s * generator.standard_normal(n))
        X = scales[:, None] * latent
        dist, idx = cKDTree(X).query(X, k=k + 2)
        rho = dist[:, 1] / dist[:, k + 1]
        neighbor_ratio = np.mean(scales[idx[:, 1:k + 1]] / scales[:, None])
        corr = (np.corrcoef(np.log(scales), rho)[0, 1]
                if sigma_s != 0 else float("nan"))
        rows.append(dict(sigma_s=sigma_s, n=n, d=d, k=k, seed=0,
                         mean_rho=float(rho.mean()),
                         neighbor_to_center_amplitude=float(neighbor_ratio),
                         corr_log_amplitude_rho=float(corr)))
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame.round(3).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
