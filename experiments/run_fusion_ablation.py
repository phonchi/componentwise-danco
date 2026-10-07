#!/usr/bin/env python
"""Channel ablations for DANCo distance and angular calibration objectives.

The command-line defaults (suites, scale sigmas, output path) are exploratory
and do not reproduce the paper designs.  The exact commands behind
``results/diagnostics/fusion_ablation_clean.csv`` and
``fusion_ablation_gsm.csv`` are listed in ``experiments/README.md``.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from danco import DANCo


OUT = Path(__file__).resolve().parents[1] / "results" / "diagnostics" / \
    "fusion_ablation_run.csv"   # the paper commands always pass --output
COMPONENTS = (
    ("MiND", "none", "distance_only"),
    ("Gride", "none", "distance_only"),
    ("none", "standard", "angle_only"),
    ("none", "profiled", "angle_only"),
    ("MiND", "standard", "joint"),
    ("MiND", "profiled", "joint"),
    ("Gride", "standard", "joint"),
    ("Gride", "profiled", "joint"),
)
COLUMNS = (
    "suite", "dataset", "true_d", "ambient_d", "n", "rep", "data_seed",
    "noise_sigma", "noise_seed", "scale_sigma", "intervention", "max_dim",
    "calibration_seed", "distance_model", "angle_mode", "objective",
    "estimate", "integer_argmin", "mu_nu", "mu_tau", "base_dhat",
    "seconds",
)
CELL_FIELDS = (
    "suite", "dataset", "rep", "noise_sigma", "scale_sigma", "intervention",
    "n", "max_dim",
)


def estimate_objective(curve, ambient_dim, fractal=True):
    """Return the integer or interpolated minimizer of one calibration curve."""
    curve = np.asarray(curve, dtype=float)
    estimate = int(np.argmin(curve)) + 1
    if not fractal:
        return float(estimate)

    # Same refinement as ``DANCo.evaluate``: the interpolation order is limited
    # by the number of candidates, and refinement is restricted to
    # [1, maxDim] (``ambient_dim`` is kept for call compatibility).
    n_candidates = len(curve)
    if n_candidates < 2:
        return float(estimate)
    kind = "cubic" if n_candidates > 3 else (
        "quadratic" if n_candidates == 3 else "linear"
    )
    objective = interp1d(
        np.arange(1, n_candidates + 1),
        curve,
        kind=kind,
        bounds_error=False,
    )
    return float(minimize(
        objective,
        estimate,
        bounds=[(1, n_candidates)],
        tol=1e-3,
    )["x"][0])


def component_estimates(stats, ambient_dim, fractal=True):
    """Rebuild all single-channel and joint estimates from evaluator stats."""
    if stats["kl_divergence"] is None:
        return {
            "distance_only": float(stats["dhat"]),
            "angle_standard": float("nan"),
            "angle_profiled": float("nan"),
            "joint_standard": float(stats["de_standard"]),
            "joint_profiled": float(stats["de_nustrip"]),
        }
    kl = np.asarray(stats["kl_divergence"], dtype=float)
    angular_profiled = np.asarray(stats["kl_ang_strip"], dtype=float)
    curves = {
        "distance_only": kl[:, 0],
        "angle_standard": kl[:, 1],
        "angle_profiled": angular_profiled,
        "joint_standard": kl[:, 0] + kl[:, 1],
        "joint_profiled": kl[:, 0] + angular_profiled,
    }
    return {
        name: estimate_objective(curve, ambient_dim, fractal)
        for name, curve in curves.items()
    }


def _normal(value):
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), 8)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _cell_key(row):
    return tuple(_normal(row[field]) for field in CELL_FIELDS)


def _component_key(row):
    return row["distance_model"], row["angle_mode"], row["objective"]


def load_checkpoint(path):
    """Load complete cells and discard any partial cell left by interruption."""
    if not path.exists():
        return [], set()
    frame = pd.read_csv(path)
    if "intervention" not in frame.columns:
        frame["intervention"] = "raw"
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}; use --no-resume")

    expected = set(COMPONENTS)
    keep = []
    complete = set()
    for _, group in frame.groupby(list(CELL_FIELDS), dropna=False, sort=False):
        signatures = {_component_key(row) for row in group.to_dict("records")}
        if signatures == expected and len(group) == len(expected):
            keep.extend(group.to_dict("records"))
            complete.add(_cell_key(group.iloc[0]))
    dropped = len(frame) - len(keep)
    if dropped:
        print(f"resume: dropped {dropped} row(s) from partial cells")
    print(f"resume: {len(complete)} complete cell(s) in {path}")
    return keep, complete


def write_checkpoint(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _integer_argmins(stats):
    if stats["kl_divergence"] is None:
        return {
            "distance_only": float(stats["dhat"]),
            "angle_standard": float("nan"),
            "angle_profiled": float("nan"),
            "joint_standard": float(stats["de_standard"]),
            "joint_profiled": float(stats["de_nustrip"]),
        }
    kl = np.asarray(stats["kl_divergence"])
    profiled = np.asarray(stats["kl_ang_strip"])
    curves = {
        "distance_only": kl[:, 0],
        "angle_standard": kl[:, 1],
        "angle_profiled": profiled,
        "joint_standard": kl[:, 0] + kl[:, 1],
        "joint_profiled": kl[:, 0] + profiled,
    }
    return {name: int(np.argmin(curve)) + 1 for name, curve in curves.items()}


def make_estimators(n, calibration_seed):
    common = dict(
        N=n,
        k=10,
        random_state=calibration_seed,
        fractal=True,
        verbose=False,
        calibration_seeding="stream",
    )
    return {
        "MiND": DANCo(version="DANCo", **common),
        "Gride": DANCo(version="DANCo_gride", **common),
    }


def evaluate_components(X, estimators, max_dim, n_jobs):
    stats_by_model = {}
    seconds = {}
    for model, estimator in estimators.items():
        started = time.perf_counter()
        stats_by_model[model] = estimator.evaluate_all(
            X, maxDim=max_dim, n_jobs=n_jobs,
        )
        seconds[model] = time.perf_counter() - started

    mind = stats_by_model["MiND"]
    gride = stats_by_model["Gride"]
    if (mind["kl_divergence"] is not None
            and gride["kl_divergence"] is not None):
        if not np.allclose(
            mind["kl_divergence"][:, 1],
            gride["kl_divergence"][:, 1],
            equal_nan=True,
        ) or not np.allclose(
            mind["kl_ang_strip"], gride["kl_ang_strip"], equal_nan=True,
        ):
            raise RuntimeError("MiND and Gride did not share angular calibration")
    return stats_by_model, seconds


def component_rows(metadata, stats_by_model, seconds):
    estimates = {
        model: component_estimates(stats, metadata["ambient_d"], fractal=True)
        for model, stats in stats_by_model.items()
    }
    integers = {
        model: _integer_argmins(stats)
        for model, stats in stats_by_model.items()
    }
    specs = (
        ("MiND", "none", "distance_only", "MiND", "distance_only"),
        ("Gride", "none", "distance_only", "Gride", "distance_only"),
        ("none", "standard", "angle_only", "MiND", "angle_standard"),
        ("none", "profiled", "angle_only", "MiND", "angle_profiled"),
        ("MiND", "standard", "joint", "MiND", "joint_standard"),
        ("MiND", "profiled", "joint", "MiND", "joint_profiled"),
        ("Gride", "standard", "joint", "Gride", "joint_standard"),
        ("Gride", "profiled", "joint", "Gride", "joint_profiled"),
    )
    rows = []
    for distance_model, angle_mode, objective, source, key in specs:
        stats = stats_by_model[source]
        rows.append({
            **metadata,
            "distance_model": distance_model,
            "angle_mode": angle_mode,
            "objective": objective,
            "estimate": estimates[source][key],
            "integer_argmin": integers[source][key],
            "mu_nu": float(stats["mu_nu"]),
            "mu_tau": float(stats["mu_tau"]),
            "base_dhat": float(stats["dhat"]),
            "seconds": seconds[source],
        })
    return rows


def run_cell(X, metadata, estimators, args, rows, complete):
    key = _cell_key(metadata)
    if key in complete:
        return 0
    stats, seconds = evaluate_components(
        X, estimators, max_dim=metadata["max_dim"], n_jobs=args.n_jobs,
    )
    rows.extend(component_rows(metadata, stats, seconds))
    complete.add(key)
    write_checkpoint(rows, args.output)
    print(
        f"{metadata['suite']}/{metadata['dataset']} rep={metadata['rep']} "
        f"scale={metadata['scale_sigma']:.2f} "
        f"intervention={metadata['intervention']} "
        f"noise={metadata['noise_sigma']:.2f} "
        f"MiND={seconds['MiND']:.2f}s Gride={seconds['Gride']:.2f}s",
        flush=True,
    )
    return 1


def benchmark_data(seed, n):
    from skdim.datasets import BenchmarkManifolds

    benchmark = BenchmarkManifolds(random_state=seed)
    datasets = benchmark.generate(n=n)
    truth = {name: float(benchmark._dict_truth[name][0]) for name in datasets}
    return datasets, truth


def select_manifolds(datasets, requested):
    if not requested:
        return datasets
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in datasets]
    if unknown:
        raise SystemExit(f"unknown manifold(s) {unknown}; available: {list(datasets)}")
    return {name: datasets[name] for name in datasets if name in names}


def metadata_for(
    suite, dataset, X, true_d, rep, data_seed, noise_sigma, noise_seed,
    scale_sigma, max_dim, calibration_seed, intervention="raw",
):
    return dict(
        suite=suite,
        dataset=dataset,
        true_d=float(true_d),
        ambient_d=int(X.shape[1]),
        n=int(X.shape[0]),
        rep=int(rep),
        data_seed=int(data_seed),
        noise_sigma=float(noise_sigma),
        noise_seed=None if noise_seed is None else int(noise_seed),
        scale_sigma=float(scale_sigma),
        intervention=intervention,
        max_dim=int(max_dim),
        calibration_seed=int(calibration_seed),
    )


def run_clean(args, rows, complete):
    datasets, truth = benchmark_data(71, args.n_points)
    datasets = select_manifolds(datasets, args.manifolds)
    new = 0
    for rep in range(args.reps):
        calibration_seed = 35 + rep
        estimators = make_estimators(args.n_points, calibration_seed)
        for name, X in datasets.items():
            metadata = metadata_for(
                "clean", name, X, truth[name], rep, 71, 0.0, None, 0.0,
                args.max_dim, calibration_seed,
            )
            new += run_cell(X, metadata, estimators, args, rows, complete)
    return new


def run_noise(args, rows, complete):
    new = 0
    for rep in range(args.reps):
        datasets, truth = benchmark_data(2000 + rep, args.n_points)
        selected = set(select_manifolds(datasets, args.manifolds))
        calibration_seed = 500 + rep
        estimators = make_estimators(args.n_points, calibration_seed)
        for sigma_index, sigma in enumerate(args.sigmas):
            noise_seed = 91000 + 100 * rep + sigma_index
            generator = np.random.default_rng(noise_seed)
            for name, X in datasets.items():
                noisy = X if sigma == 0 else (
                    X + sigma * generator.standard_normal(X.shape)
                )
                if name not in selected:
                    continue
                metadata = metadata_for(
                    "noise", name, noisy, truth[name], rep, 2000 + rep,
                    sigma, noise_seed, 0.0, args.max_dim, calibration_seed,
                )
                new += run_cell(
                    noisy, metadata, estimators, args, rows, complete,
                )
    return new


def gaussian_scale_mixture_components(seed, sigma_s, n, d=70, ambient_d=None):
    """Return the latent Gaussian, amplitudes, and observed GSM sample.

    Exposing the generating components permits the paired latent-amplitude
    intervention
    ``X_i / S_i = Z_i``.  Raw and de-scaled observations therefore share the
    same latent draw, rather than comparing independently generated samples.
    """
    generator = np.random.default_rng(seed)
    latent = generator.standard_normal((n, d))
    scales = np.ones(n)
    if sigma_s != 0:
        scales = np.exp(sigma_s * generator.standard_normal(n))
    observed = latent * scales[:, None]
    if ambient_d is None or ambient_d == d:
        return latent, scales, observed
    latent_embedded = np.zeros((n, ambient_d), dtype=latent.dtype)
    observed_embedded = np.zeros((n, ambient_d), dtype=observed.dtype)
    latent_embedded[:, :d] = latent
    observed_embedded[:, :d] = observed
    return latent_embedded, scales, observed_embedded


def gaussian_scale_mixture(seed, sigma_s, n, d=70, ambient_d=None):
    """Return the observed sample from ``X_i = S_i Z_i``."""
    return gaussian_scale_mixture_components(
        seed, sigma_s, n, d=d, ambient_d=ambient_d,
    )[2]


def run_gsm(args, rows, complete):
    new = 0
    for rep in range(args.reps):
        calibration_seed = 700 + rep
        estimators = make_estimators(args.n_points, calibration_seed)
        for sigma_s in args.scale_sigmas:
            latent, _, X = gaussian_scale_mixture_components(
                rep, sigma_s, args.n_points, ambient_d=args.gsm_ambient_d,
            )
            metadata = metadata_for(
                "gsm", f"gsm_scale_{sigma_s:g}", X, 70, rep, rep,
                0.0, None, sigma_s, args.max_dim, calibration_seed,
            )
            new += run_cell(X, metadata, estimators, args, rows, complete)
            if sigma_s in args.latent_descale_sigmas:
                descaled_metadata = metadata_for(
                    "gsm", f"gsm_scale_{sigma_s:g}", latent, 70, rep, rep,
                    0.0, None, sigma_s, args.max_dim, calibration_seed,
                    intervention="latent_descaled",
                )
                new += run_cell(
                    latent, descaled_metadata, estimators, args, rows, complete,
                )
    return new


def run_crossed(args, rows, complete):
    new = 0
    for rep in range(args.reps):
        calibration_seed = 800 + rep
        estimators = make_estimators(args.n_points, calibration_seed)
        for scale_index, sigma_s in enumerate(args.scale_sigmas):
            clean = gaussian_scale_mixture(
                10000 + rep, sigma_s, args.n_points, d=70, ambient_d=100,
            )
            for noise_index, sigma in enumerate(args.crossed_sigmas):
                noise_seed = (
                    93000 + 1000 * rep + 100 * scale_index + noise_index
                )
                generator = np.random.default_rng(noise_seed)
                X = clean if sigma == 0 else (
                    clean + sigma * generator.standard_normal(clean.shape)
                )
                metadata = metadata_for(
                    "crossed", f"gsm70_in_100_scale_{sigma_s:g}", X, 70,
                    rep, 10000 + rep, sigma, noise_seed, sigma_s,
                    args.max_dim, calibration_seed,
                )
                new += run_cell(X, metadata, estimators, args, rows, complete)
    return new


def parse_floats(value):
    return tuple(float(item) for item in value.split(",") if item.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites",
        default="clean,noise,gsm,crossed",
        help="comma-separated subset of clean,noise,gsm,crossed",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n-points", type=int, default=2500)
    parser.add_argument("--max-dim", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--manifolds", default=None)
    parser.add_argument("--sigmas", default="0,0.01,0.02,0.03,0.04,0.05")
    parser.add_argument("--scale-sigmas", default="0,0.25,0.5")
    parser.add_argument(
        "--latent-descale-sigmas", default="",
        help="GSM log-amplitude spreads for paired X_i/S_i = Z_i controls",
    )
    parser.add_argument(
        "--gsm-ambient-d", type=int, default=None,
        help="embed the d=70 GSM in this ambient dimension by zero padding "
             "(default: d=D=70); the candidate search is capped at the "
             "ambient dimension, so D>d keeps the full --max-dim range",
    )
    parser.add_argument("--crossed-sigmas", default="0,0.01,0.03,0.05")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    runners = {
        "clean": run_clean,
        "noise": run_noise,
        "gsm": run_gsm,
        "crossed": run_crossed,
    }
    unknown = [suite for suite in suites if suite not in runners]
    if unknown:
        parser.error(f"unknown suite(s) {unknown}; choose from {list(runners)}")
    if args.gsm_ambient_d is not None and args.gsm_ambient_d < 70:
        parser.error("--gsm-ambient-d must be at least the generating d=70")
    if args.reps < 1 or args.n_points < 12 or args.max_dim < 1:
        parser.error(
            "--reps >= 1, --n-points >= 12, and --max-dim >= 1 required"
        )
    args.sigmas = parse_floats(args.sigmas)
    args.scale_sigmas = parse_floats(args.scale_sigmas)
    args.latent_descale_sigmas = parse_floats(args.latent_descale_sigmas)
    missing_descaled = set(args.latent_descale_sigmas) - set(args.scale_sigmas)
    if missing_descaled:
        parser.error(
            "--latent-descale-sigmas must be a subset of --scale-sigmas; "
            f"missing {sorted(missing_descaled)}"
        )
    args.crossed_sigmas = parse_floats(args.crossed_sigmas)

    rows, complete = ([], set()) if args.no_resume else load_checkpoint(args.output)
    started = time.perf_counter()
    new = 0
    for suite in suites:
        new += runners[suite](args, rows, complete)
    elapsed = time.perf_counter() - started
    print(
        f"wrote {args.output}: {new} new cell(s), {len(complete)} total, "
        f"{elapsed:.1f}s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
