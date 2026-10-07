#!/usr/bin/env python
"""Angle-channel diagnostics: why is nu-hat < 1 on natural images?

    python experiments/run_angle_diagnostics.py --dataset all
    python experiments/run_angle_diagnostics.py --dataset cifar --no-evaluate

Tests the norm-heterogeneity mechanism behind the low von Mises mean
direction observed on CIFAR-10 / ImageNet (nu-hat in [0.80, 0.95], below the
synthetic hyperball range [1.1, 1.5]).  Two predictions:

(a) per-point nu_i correlates *negatively* with the centred norm
    ||x_i - mean(x)||: points far from the data core see neighbour
    displacement vectors that all point inward, hence more collinear
    angles, hence smaller nu_i;
(b) removing per-sample norm heterogeneity (projecting to the unit sphere
    around the class mean, or per-image contrast normalisation) raises
    nu-hat and brings the standard angular calibration closer to the
    profiled one (angle-KL curve shape, standard estimate vs strip estimate).

Per-point statistics reuse the exact estimator machinery (`backend.knn`,
`backend.angle_stats`, `ml_vm`), so nu_i here is *the* quantity DANCo
aggregates.  Synthetic controls: a uniform 70-ball (isotropic, prediction:
no correlation, high nu-hat) and a Gaussian scale mixture (norm-heterogeneous
by construction, prediction: strong negative correlation, nu-hat < 1).
The synthetic rows use sigma_S = 0.5, outside the paper's GSM grid
(sigma_S in [0, 0.35]); they are supporting controls and no paper table or
figure reads them.

Outputs ``angle_diagnostics.csv`` (one row per dataset/class/variant) plus
one ``.npz`` per row with the per-point arrays for figure generation.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import IMAGE_SEED_BASE, REPO_ROOT
from run_images import load_cifar, load_imagenet, load_mnist
from danco import DANCo
from danco._kernels.angles import indnComb
from danco._kernels.vonmises import ml_vm
from danco.backends import get_backend

K = 10
MAXDIM = 200


# ---------------------------------------------------------------------------
# variants: how a class of images is preprocessed before the estimator
# ---------------------------------------------------------------------------
def variant_raw(X):
    return X


def variant_sphere(X):
    """Centre on the class mean and project every sample to the unit sphere.

    Kills per-sample norm heterogeneity while leaving directions untouched;
    this is the intervention the mechanism says should restore nu-hat.
    """
    C = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return C / norms


def variant_contrast(X):
    """Classic per-image contrast normalisation: (x - mean) / std per row.

    The Gaussian-scale-mixture view of natural images says contrast is the
    per-sample scale variable, so standardising each image individually
    should remove most of the norm heterogeneity without centring on the
    class mean.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


VARIANTS = {
    "raw": variant_raw,
    "sphere": variant_sphere,
    "contrast": variant_contrast,
}


# ---------------------------------------------------------------------------
# synthetic controls
# ---------------------------------------------------------------------------
def load_synthetic(gsm_sigma=0.5):
    """Uniform 70-ball and a d = 70 Gaussian scale mixture, both N = 2500.

    One ``default_rng(0)`` stream draws the ball first and then the GSM, so
    ``gsm_sigma`` changes only the log-amplitude scale of the same draws.
    """
    rng = np.random.default_rng(0)
    n = 2500

    g = rng.standard_normal((n, 70))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    ball70 = g * (rng.random(n) ** (1 / 70))[:, None]

    # Gaussian scale mixture x = s * g, log s ~ N(0, gsm_sigma^2); at the
    # default 0.5 the norm heterogeneity pushes nu-hat below pi/3.
    g = rng.standard_normal((n, 70))
    s = np.exp(gsm_sigma * rng.standard_normal(n))
    gsm = g * s[:, None]

    return {"ball70": ball70, f"gsm_sigma{gsm_sigma:g}": gsm}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def per_point_stats(X, backend, combs):
    """(nu_i, tau_i, centred norms, dataset-level circular-mean nu, mean tau)."""
    _, idx = backend.knn(X, K)
    thetas = backend.angle_stats(X, idx, combs, K)
    nu_i, tau_i = ml_vm(thetas)
    mu_nu = np.arctan2(np.sin(nu_i).sum(), np.cos(nu_i).sum())
    norms = np.linalg.norm(X - X.mean(axis=0, keepdims=True), axis=1)
    return nu_i, tau_i, norms, mu_nu, float(np.mean(tau_i))


def calibration_diagnosis(X, N, n_jobs, random_state):
    """Full DANCo run; report both estimates and the shape of the KL curves.

    A second, Gride-distance estimator runs on the same seed.  Both fresh
    estimators draw identical calibration balls (same RNG stream), so their
    angular references agree entry for entry and only the distance channel
    differs -- the Gride columns isolate the distance model.
    """
    est = DANCo(N=N, k=K, version="DANCo", random_state=random_state,
                fractal=True, verbose=False)
    stats = est.evaluate_all(X, maxDim=MAXDIM, n_jobs=n_jobs, spline_file=None)
    est_g = DANCo(N=N, k=K, version="DANCo_gride", random_state=random_state,
                  fractal=True, verbose=False)
    stats_g = est_g.evaluate_all(X, maxDim=MAXDIM, n_jobs=n_jobs,
                                 spline_file=None)
    out = dict(de_standard=stats["de_standard"], de_nustrip=stats["de_nustrip"],
               de_standard_gride=stats_g["de_standard"],
               de_nustrip_gride=stats_g["de_nustrip"])
    kl = stats.get("kl_divergence")
    if kl is not None:
        from scipy.stats import spearmanr as _sp

        ang = kl[:, 1]
        total = kl.sum(axis=1)
        out["angle_kl_monotone"] = bool(np.all(np.diff(ang) > 0))
        # The failure signature on image data is not global monotonicity but
        # an angular argmin at very small m followed by an essentially
        # increasing curve, which drags the joint argmin far below the
        # distance channel's; record the pieces of that signature.
        out["ang_argmin"] = int(np.argmin(ang)) + 1
        out["ang_spearman"] = float(
            _sp(np.arange(len(ang)), ang).statistic)
        out["dist_argmin"] = int(np.argmin(kl[:, 0])) + 1
        out["total_argmin"] = int(np.argmin(total)) + 1
        out["argmin_at_boundary"] = bool(np.argmin(total) in (0, len(total) - 1))
        out["_kl_curves"] = kl
        out["_kl_strip"] = stats.get("kl_ang_strip")
    if stats_g.get("kl_divergence") is not None:
        out["_kl_curves_gride"] = stats_g["kl_divergence"]
    return out


def run_class(name, label, X, args, backend, combs, rows, npz_dir):
    for vname, fn in VARIANTS.items():
        Xv = fn(np.asarray(X, dtype=np.float64))
        t0 = time.time()
        nu_i, tau_i, norms, mu_nu, mu_tau = per_point_stats(Xv, backend, combs)
        r_p = pearsonr(norms, nu_i)
        r_s = spearmanr(norms, nu_i)
        # Data-side statistics do not depend on the calibration seed; they are
        # computed once and repeated verbatim on every per-seed row.
        base = dict(
            dataset=name, label=label, variant=vname,
            N=Xv.shape[0], D=Xv.shape[1],
            mu_nu=mu_nu, mu_tau=mu_tau,
            norm_cv=float(norms.std() / norms.mean()),
            pearson_r=float(r_p.statistic), pearson_p=float(r_p.pvalue),
            spearman_r=float(r_s.statistic), spearman_p=float(r_s.pvalue),
        )
        perpoint_seconds = time.time() - t0
        extra_arrays = {}
        n_seeds = 1 if args.no_evaluate else args.seeds
        for i in range(n_seeds):
            seed_val = IMAGE_SEED_BASE + i
            row = dict(base, seed=seed_val)
            t1 = time.time()
            if not args.no_evaluate:
                diag = calibration_diagnosis(Xv, Xv.shape[0], args.n_jobs,
                                             seed_val)
                for key in ("_kl_curves", "_kl_strip", "_kl_curves_gride"):
                    arr = diag.pop(key, None)
                    # The per-point npz keeps the KL curves of the base seed
                    # only, so the figure inputs keep their historical meaning.
                    if arr is not None and i == 0:
                        extra_arrays[key.lstrip("_")] = arr
                row.update(diag)
            row["seconds"] = round(
                time.time() - t1 + (perpoint_seconds if i == 0 else 0.0), 1)
            rows.append(row)
            print(f"  {name}/{label}/{vname}[seed {seed_val}]: nu={mu_nu:.3f} "
                  f"pearson={row['pearson_r']:+.3f} cv={row['norm_cv']:.3f} "
                  + (f"std={row.get('de_standard', float('nan')):.1f} "
                     f"strip={row.get('de_nustrip', float('nan')):.1f} "
                     f"ang_monotone={row.get('angle_kl_monotone', '-')}"
                     if not args.no_evaluate else "")
                  + f" ({row['seconds']}s)", flush=True)
        np.savez_compressed(
            npz_dir / f"{name}_{label}_{vname}.npz",
            nu_i=nu_i, tau_i=tau_i, norms=norms, **extra_arrays)


LOADERS = {
    "synthetic": (load_synthetic, None),
    "mnist": (load_mnist, None),
    "cifar": (load_cifar, 5000),
    "imagenet": (load_imagenet, None),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=list(LOADERS) + ["all"],
                        default="all")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="skip the full calibration runs (fast pass)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of calibration seeds; estimator "
                             "random_state = IMAGE_SEED_BASE + i, matching "
                             "the run_images.py convention")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results" / "diagnostics")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    npz_dir = args.output / "perpoint"
    npz_dir.mkdir(exist_ok=True)

    backend = get_backend("cpu")
    combs = indnComb(K)
    names = list(LOADERS) if args.dataset == "all" else [args.dataset]

    rows = []
    for name in names:
        loader, n_override = LOADERS[name]
        print(f"\n{name}: loading...", flush=True)
        for label, X in loader().items():
            if n_override is not None:
                X = X[:n_override]
            run_class(name, label, X, args, backend, combs, rows, npz_dir)

    df = pd.DataFrame(rows)
    out = args.output / "angle_diagnostics.csv"
    if out.exists():
        old = pd.read_csv(out)
        # Legacy single-seed files predate the seed column; those rows were
        # produced with random_state = IMAGE_SEED_BASE.
        if "seed" not in old.columns:
            old["seed"] = IMAGE_SEED_BASE
        # Union the columns (runs with --no-evaluate carry fewer) and replace
        # rows re-run for the same (dataset, label, variant, seed).
        key = ["dataset", "label", "variant", "seed"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df], ignore_index=True, sort=False)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(rows)} new rows ({len(df)} total) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
