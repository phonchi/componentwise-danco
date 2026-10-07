#!/usr/bin/env python
"""Export the image-experiment reference curves and their verified envelope."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import REPO_ROOT
except ImportError:  # pragma: no cover - direct CLI execution
    from common import REPO_ROOT

from danco import DANCo


DEFAULT_CURVES = REPO_ROOT / "results" / "diagnostics" / \
    "image_reference_curves.csv"
DEFAULT_ENVELOPE = REPO_ROOT / "results" / "diagnostics" / \
    "image_reference_envelope.csv"
ANGLE_DIAGNOSTICS = REPO_ROOT / "results" / "diagnostics" / \
    "angle_diagnostics.csv"


def image_sample_sizes(path=ANGLE_DIAGNOSTICS):
    """Sample sizes actually used by the raw image diagnostic runs."""
    frame = pd.read_csv(path)
    selected = frame[
        frame["dataset"].isin(("mnist", "cifar", "imagenet"))
        & (frame["variant"] == "raw")
    ]
    sizes = tuple(sorted(int(value) for value in selected["N"].unique()))
    if not sizes:
        raise ValueError(f"no raw image sample sizes found in {path}")
    return sizes


def build_reference_curves(sample_sizes, seeds, max_dim=200, k=10, n_jobs=-1):
    """Build stream-seeded references matching the image experiment."""
    rows = []
    for n in sample_sizes:
        for seed in seeds:
            started = time.perf_counter()
            estimator = DANCo(
                N=int(n), k=int(k), version="DANCo", random_state=int(seed),
                fractal=True, verbose=False, calibration_seeding="stream",
            )
            for candidate_dim in range(1, int(max_dim) + 1):
                ball = estimator.hyperBall(
                    int(n), candidate_dim, center=[0] * candidate_dim,
                )
                stats = estimator.danco_statistic(ball, n_jobs=n_jobs)
                rows.append({
                    "N": int(n),
                    "k": int(k),
                    "seed": int(seed),
                    "candidate_dim": candidate_dim,
                    "dhat_ref": float(stats["dhat"]),
                    "nu_ref": float(stats["mu_nu"]),
                    "tau_ref": float(stats["mu_tau"]),
                    "calibration_seeding": "stream",
                })
            print(f"N={n} seed={seed}: {max_dim} references in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    return pd.DataFrame(rows)


def summarize_reference_envelope(frame, expected_ns, expected_seeds,
                                 max_dim=200):
    """Validate the grid and return the identifiable angular envelope.

    Candidate references whose distance pre-estimate is at most five carry
    ``mu_nu=mu_tau=0`` by the estimator shortcut. Their location parameter is
    undefined rather than a zero-radian reference, so they remain in the curve
    export but are excluded from the angular envelope.
    """
    expected = {
        (int(n), int(seed), candidate_dim)
        for n in expected_ns
        for seed in expected_seeds
        for candidate_dim in range(1, int(max_dim) + 1)
    }
    actual = {
        (int(row.N), int(row.seed), int(row.candidate_dim))
        for row in frame.itertuples(index=False)
    }
    if actual != expected or len(frame) != len(expected):
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(
            "complete reference grid required: "
            f"expected {len(expected)} rows, got {len(frame)} "
            f"(missing={missing}, extra={extra})"
        )
    valid = frame[
        (frame["dhat_ref"] > 5)
        & (frame["tau_ref"] > 0)
        & np.isfinite(frame["nu_ref"])
    ]
    if valid.empty:
        raise ValueError("reference grid contains no identifiable angular rows")
    return {
        "n_rows": len(frame),
        "n_valid_angular_rows": len(valid),
        "n_sample_sizes": len(expected_ns),
        "n_seeds": len(expected_seeds),
        "max_dim": int(max_dim),
        "valid_candidate_min": int(valid["candidate_dim"].min()),
        "valid_candidate_max": int(valid["candidate_dim"].max()),
        "nu_min": float(valid["nu_ref"].min()),
        "nu_max": float(valid["nu_ref"].max()),
    }


def _atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", default=None,
                        help="comma-separated N values; default: raw image Ns")
    parser.add_argument("--seeds", default="43,44,45,46,47")
    parser.add_argument("--max-dim", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--curves-output", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--envelope-output", type=Path,
                        default=DEFAULT_ENVELOPE)
    parser.add_argument("--reuse-curves", action="store_true",
                        help="validate and summarize --curves-output without "
                             "regenerating reference samples")
    args = parser.parse_args(argv)

    sample_sizes = (
        tuple(int(item) for item in args.sample_sizes.split(",") if item.strip())
        if args.sample_sizes else image_sample_sizes()
    )
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    frame = (
        pd.read_csv(args.curves_output)
        if args.reuse_curves else build_reference_curves(
            sample_sizes, seeds, args.max_dim, args.k, args.n_jobs,
        )
    )
    summary = summarize_reference_envelope(
        frame, sample_sizes, seeds, args.max_dim,
    )
    envelope = pd.DataFrame([{
        **summary,
        "sample_sizes": ",".join(str(value) for value in sample_sizes),
        "seeds": ",".join(str(value) for value in seeds),
        "k": args.k,
        "calibration_seeding": "stream",
    }])
    if not args.reuse_curves:
        _atomic_csv(frame, args.curves_output)
    _atomic_csv(envelope, args.envelope_output)
    action = "validated" if args.reuse_curves else "wrote"
    print(f"{action} {args.curves_output}: {len(frame)} rows")
    print(f"wrote {args.envelope_output}: "
          f"nu in [{summary['nu_min']:.6f}, {summary['nu_max']:.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
