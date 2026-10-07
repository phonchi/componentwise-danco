"""Signed relative bias of each ablation objective, per regime.

Companion to fusion_ablation_summary.py: same complete-case filter and
dataset-first aggregation, but the statistic is the signed percentage
error 100*(estimate - true_d)/true_d instead of its absolute value, so
the direction of each channel's bias is preserved.
"""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd

from fusion_ablation_summary import (
    METHOD_ORDER, REGIME, complete_case_frame, prepare,
)


def signed_bias_summary(frame, source_label):
    frame = complete_case_frame(frame)
    frame["signed_pe"] = 100 * (frame["estimate"] - frame["true_d"]) \
        / frame["true_d"]
    keys = REGIME + ["method"]
    per_dataset = frame.groupby(keys + ["dataset"], as_index=False).agg(
        signed_pe=("signed_pe", "mean"),
        n_reps=("rep", "nunique"),
    )
    cells = frame.groupby(keys, as_index=False).agg(
        n_cells=("signed_pe", "size"),
        cells_over=("signed_pe", lambda s: int((s > 0).sum())),
        cells_under=("signed_pe", lambda s: int((s < 0).sum())),
    )
    summary = per_dataset.groupby(keys, as_index=False).agg(
        n_datasets=("dataset", "nunique"),
        n_reps=("n_reps", "max"),
        signed_pe_mean=("signed_pe", "mean"),
        signed_pe_sd=("signed_pe", "std"),
    )
    summary = summary.merge(cells, on=keys, how="left", validate="one_to_one")
    summary.insert(0, "source", source_label)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True,
    )
    return summary.sort_values(REGIME + ["method"]).reset_index(drop=True)


def complete_case_frame_for(frame, regime):
    """Complete-case filter with a caller-chosen regime key."""
    frame = prepare(frame)
    cell = regime + ["dataset", "rep"]
    status = frame.assign(finite=np.isfinite(frame["estimate"])).groupby(
        cell, as_index=False,
    ).agg(
        n_rows=("method", "size"),
        n_methods=("method", "nunique"),
        all_finite=("finite", "all"),
    )
    keep = status[
        (status["n_rows"] == len(METHOD_ORDER))
        & (status["n_methods"] == len(METHOD_ORDER))
        & status["all_finite"]
    ][cell]
    return frame.merge(keep, on=cell, how="inner", validate="many_to_one")


def signed_bias_by_replicate(frame, regime, source_label):
    """Signed bias aggregated replicate-first, matching the MPE tables."""
    frame = complete_case_frame_for(frame, regime)
    frame["signed_pe"] = 100 * (frame["estimate"] - frame["true_d"]) \
        / frame["true_d"]
    keys = regime + ["method"]
    per_rep = frame.groupby(keys + ["rep"], as_index=False).agg(
        signed_pe=("signed_pe", "mean"),
        n_datasets=("dataset", "nunique"),
    )
    summary = per_rep.groupby(keys, as_index=False).agg(
        n_reps=("rep", "nunique"),
        n_datasets=("n_datasets", "max"),
        signed_pe_mean=("signed_pe", "mean"),
        signed_pe_sd=("signed_pe", "std"),
    )
    summary.insert(0, "source", source_label)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True,
    )
    return summary.sort_values(regime + ["method"]).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-source", type=Path,
        default=Path("results/diagnostics/fusion_ablation_clean.csv"),
    )
    parser.add_argument(
        "--ablation-source", type=Path, default=None,
        help="optional additional run_fusion_ablation.py CSV whose rows are "
             "appended with source=ablation (not used for the paper)",
    )
    parser.add_argument(
        "--relative-noise-source", type=Path,
        default=Path("results/diagnostics/relative_noise_ablation.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/diagnostics/fusion_ablation_signed_bias.csv"),
    )
    args = parser.parse_args(argv)
    clean = pd.read_csv(args.clean_source)
    parts = [signed_bias_summary(clean, "clean_confirmatory")]
    # An earlier, superseded ablation run is not distributed with this
    # repository; its "ablation" rows are appended only when that source file
    # is present.  The manuscript tables do not depend on it.
    if args.ablation_source is not None:
        parts.append(signed_bias_summary(
            pd.read_csv(args.ablation_source), "ablation",
        ))
    summary = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print("wrote", args.output)
    by_replicate = pd.concat([
        signed_bias_by_replicate(clean, REGIME, "clean_confirmatory"),
        signed_bias_by_replicate(
            pd.read_csv(args.relative_noise_source),
            ["suite", "noise_eta"], "relative_noise",
        ),
    ], ignore_index=True)
    by_replicate_output = args.output.with_name(
        "fusion_ablation_signed_bias_by_replicate.csv",
    )
    by_replicate.to_csv(by_replicate_output, index=False)
    print("wrote", by_replicate_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
