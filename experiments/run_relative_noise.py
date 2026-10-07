#!/usr/bin/env python
"""Component ablation under dimensionless, neighborhood-relative noise.

For each clean dataset, let ``r10`` be the median tenth-neighbor distance.  At
level ``eta`` the coordinate-noise standard deviation is

    sigma_epsilon = eta * r10 / sqrt(2 D).

The expected Euclidean separation of two independent noise vectors is then
approximately ``eta * r10``.  This makes the perturbation comparable across
datasets with different ambient dimensions and coordinate units.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from .common import REPO_ROOT
    from .fusion_ablation_summary import METHODS
    from .run_fusion_ablation import (
        benchmark_data,
        component_rows,
        evaluate_components,
        make_estimators,
        metadata_for,
        parse_floats,
        select_manifolds,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from common import REPO_ROOT
    from fusion_ablation_summary import METHODS
    from run_fusion_ablation import (
        benchmark_data,
        component_rows,
        evaluate_components,
        make_estimators,
        metadata_for,
        parse_floats,
        select_manifolds,
    )


OUT = REPO_ROOT / "results" / "diagnostics" / "relative_noise_ablation.csv"
SUMMARY = OUT.with_name("relative_noise_ablation_summary.csv")


def relative_noise_sd(data, eta, neighbor_order=10):
    """Coordinate SD whose pairwise noise norm is ``eta`` times local distance."""
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if eta == 0:
        return 0.0
    if len(data) <= neighbor_order:
        raise ValueError("data must contain more points than neighbor_order")
    distances = NearestNeighbors(
        n_neighbors=neighbor_order, algorithm="auto",
    ).fit(data).kneighbors(return_distance=True)[0]
    local_distance = float(np.median(distances[:, neighbor_order - 1]))
    return float(eta * local_distance / np.sqrt(2.0 * data.shape[1]))


def summarize(frame):
    frame = frame.copy()
    keys = list(zip(
        frame["distance_model"], frame["angle_mode"], frame["objective"],
    ))
    frame["method"] = [METHODS[key] for key in keys]
    frame["ape"] = 100.0 * (frame["estimate"] - frame["true_d"]).abs() \
        / frame["true_d"]
    per_rep = frame.groupby(
        ["noise_eta", "method", "rep"], as_index=False,
    ).agg(mpe=("ape", "mean"), n_datasets=("dataset", "nunique"))
    return per_rep.groupby(["noise_eta", "method"], as_index=False).agg(
        n_reps=("rep", "nunique"),
        n_datasets=("n_datasets", "min"),
        mpe_mean=("mpe", "mean"),
        mpe_sd=("mpe", "std"),
    )


def _write(frame, output, summary_output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output)
    summarize(frame).to_csv(summary_output, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--etas", default="0,0.05,0.1,0.2,0.4")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--n-points", type=int, default=2500)
    parser.add_argument("--max-dim", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--manifolds", default=None)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY)
    args = parser.parse_args(argv)
    if args.reps < 1 or args.n_points < 12 or args.max_dim < 1:
        parser.error("--reps >= 1, --n-points >= 12, and --max-dim >= 1 required")
    etas = parse_floats(args.etas)

    rows = []
    for rep in range(args.reps):
        data_seed = 12_000 + rep
        datasets, truth = benchmark_data(data_seed, args.n_points)
        datasets = select_manifolds(datasets, args.manifolds)
        calibration_seed = 2_000 + rep
        estimators = make_estimators(args.n_points, calibration_seed)
        for eta_index, eta in enumerate(etas):
            for dataset_index, (name, clean) in enumerate(datasets.items()):
                sigma_epsilon = relative_noise_sd(clean, eta, neighbor_order=10)
                noise_seed = 150_000 + 10_000 * rep + 100 * eta_index + dataset_index
                generator = np.random.default_rng(noise_seed)
                observed = clean if eta == 0 else (
                    clean + sigma_epsilon * generator.standard_normal(clean.shape)
                )
                metadata = metadata_for(
                    "relative_noise", name, observed, truth[name], rep,
                    data_seed, sigma_epsilon, noise_seed, 0.0, args.max_dim,
                    calibration_seed,
                )
                metadata["noise_eta"] = float(eta)
                metadata["noise_sd"] = sigma_epsilon
                stats, seconds = evaluate_components(
                    observed, estimators, args.max_dim, args.n_jobs,
                )
                rows.extend(component_rows(metadata, stats, seconds))
            frame = pd.DataFrame(rows)
            _write(frame, args.output, args.summary_output)
            print(
                f"rep={rep} eta={eta:g}: {len(frame)} rows; "
                f"wrote {args.output}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
