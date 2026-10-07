#!/usr/bin/env python
"""Data behind the Bessel-overflow figure.

    python experiments/run_overflow_curve.py
    python experiments/run_overflow_curve.py --calibration-n 450,500,580,640,700

Evaluates the von Mises KL term with the legacy (unscaled ``i0``/``i1``,
clamped) expression and with the exponentially scaled default, as the
calibration-side concentration sweeps through the double-precision overflow
threshold at tau = 709.78. The data-side statistic is held fixed at values
typical of an image-scale run. Also records the self-KL and a mid-range
agreement check so the figure can quote both. ``--calibration-n`` sweeps the
reference sample size of the packaged calibration surface, so the
N-dependence of the first overflowing candidate dimension is reproducible.

Output: results/diagnostics/overflow_curve.csv
        results/diagnostics/overflow_candidate_curve.csv
"""

import argparse

import numpy as np
import pandas as pd

from common import REPO_ROOT
from danco._kernels.vonmises import kl_vonmises, kl_vonmises_strip
from danco.spline import load_splines

OUT = REPO_ROOT / "results" / "diagnostics"

NU1, TAU1 = 1.0, 300.0    # data-side statistics, safely below overflow
NU2 = 1.2                 # calibration-side mean direction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibration-n", default="500",
                        help="comma-separated reference sample sizes for the "
                             "candidate sweep (default: 500)")
    args = parser.parse_args(argv)
    calibration_ns = [int(v) for v in args.calibration_n.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    taus = np.concatenate([
        np.linspace(1, 700, 200),
        np.linspace(700, 720, 120),          # dense around the threshold
        np.linspace(720, 1200, 200),
    ])
    rows = []
    for tau2 in taus:
        rows.append(dict(
            tau2=float(tau2),
            kl_scaled=float(kl_vonmises(NU1, TAU1, NU2, tau2)),
            kl_legacy=float(kl_vonmises(NU1, TAU1, NU2, tau2,
                                        use_legacy_bessel=True)),
            kl_strip_scaled=float(kl_vonmises_strip(TAU1, tau2)),
            kl_strip_legacy=float(kl_vonmises_strip(TAU1, tau2,
                                                    use_legacy_bessel=True)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "overflow_curve.csv", index=False)

    splines = load_splines()
    candidate_rows = []
    for calibration_n in calibration_ns:
        for candidate_dim in range(1, 401):
            tau_ref = float(
                splines["spline_mu_tau"](candidate_dim, calibration_n)[0, 0]
            )
            candidate_rows.append(dict(
                candidate_dim=candidate_dim,
                calibration_n=calibration_n,
                tau_ref=tau_ref,
                kl_scaled=float(kl_vonmises(NU1, TAU1, NU2, tau_ref)),
                kl_legacy=float(kl_vonmises(
                    NU1, TAU1, NU2, tau_ref, use_legacy_bessel=True,
                )),
            ))
    candidate = pd.DataFrame(candidate_rows)
    out_candidate = OUT / "overflow_candidate_curve.csv"
    if out_candidate.exists():
        old = pd.read_csv(out_candidate)
        key = ["calibration_n", "candidate_dim"]
        old = old[~old.set_index(key).index.isin(
            candidate.set_index(key).index)]
        candidate = pd.concat([old, candidate], ignore_index=True, sort=False)
        candidate = candidate.sort_values(key).reset_index(drop=True)
    candidate.to_csv(out_candidate, index=False)

    below = df[df.tau2 < 700]
    agreement = float(np.max(np.abs(below.kl_scaled - below.kl_legacy)))
    self_kl = float(kl_vonmises(NU1, TAU1, NU1, TAU1))
    print(f"max |scaled - legacy| below threshold: {agreement:.3e}")
    print(f"self-KL (scaled): {self_kl:.3e}")
    neg = df[(df.kl_legacy < 0)]
    if len(neg):
        first = neg.iloc[0]
        print(f"legacy turns negative at tau2={first.tau2:.1f} "
              f"(KL={first.kl_legacy:.1f})")
    for calibration_n, grp in candidate.groupby("calibration_n"):
        affected = grp[grp["tau_ref"] > 709.78]
        if len(affected):
            first = affected.iloc[0]
            print(
                f"first overflowing candidate at N={calibration_n}: "
                f"m={int(first.candidate_dim)}, tau_ref={first.tau_ref:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
