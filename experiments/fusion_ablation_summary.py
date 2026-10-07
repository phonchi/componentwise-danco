"""Aggregate and paired-bootstrap the DANCo channel-ablation CSV."""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    ("MiND", "none", "distance_only"): "mind_distance_only",
    ("Gride", "none", "distance_only"): "gride_distance_only",
    ("none", "standard", "angle_only"): "standard_angle_only",
    ("none", "profiled", "angle_only"): "profiled_angle_only",
    ("MiND", "standard", "joint"): "danco",
    ("MiND", "profiled", "joint"): "profiled_danco",
    ("Gride", "standard", "joint"): "danco_gride",
    ("Gride", "profiled", "joint"): "profiled_danco_gride",
}
METHOD_ORDER = tuple(METHODS.values())
CLAIM_COMPARISONS = (
    ("profiled_danco", "danco"),
    ("profiled_danco_gride", "danco_gride"),
    ("danco", "mind_distance_only"),
    ("danco", "standard_angle_only"),
    ("profiled_danco", "mind_distance_only"),
    ("profiled_danco", "profiled_angle_only"),
    ("danco_gride", "gride_distance_only"),
    ("danco_gride", "standard_angle_only"),
    ("profiled_danco_gride", "gride_distance_only"),
    ("profiled_danco_gride", "profiled_angle_only"),
)
REGIME = ["suite", "noise_sigma", "scale_sigma", "intervention"]
CELL = REGIME + ["dataset", "rep"]


def prepare(frame):
    frame = frame.copy()
    if "intervention" not in frame.columns:
        frame["intervention"] = "raw"
    keys = list(zip(
        frame["distance_model"], frame["angle_mode"], frame["objective"],
    ))
    frame["method"] = [METHODS[key] for key in keys]
    frame["ape"] = 100 * (frame["estimate"] - frame["true_d"]).abs() \
        / frame["true_d"]
    return frame


def complete_case_frame(frame):
    """Retain only paired cells containing eight finite method estimates."""
    frame = prepare(frame)
    status = frame.assign(finite=np.isfinite(frame["estimate"])).groupby(
        CELL, as_index=False,
    ).agg(
        n_rows=("method", "size"),
        n_methods=("method", "nunique"),
        all_finite=("finite", "all"),
    )
    keep = status[
        (status["n_rows"] == len(METHOD_ORDER))
        & (status["n_methods"] == len(METHOD_ORDER))
        & status["all_finite"]
    ][CELL]
    return frame.merge(keep, on=CELL, how="inner", validate="many_to_one")


def aggregate(frame):
    frame = complete_case_frame(frame)
    cells = frame[CELL].drop_duplicates()
    counts = cells.groupby(REGIME, as_index=False).agg(
        n_pairs=("rep", "size"),
        n_datasets=("dataset", "nunique"),
    )
    per_rep = frame.groupby(REGIME + ["method", "rep"], as_index=False).agg(
        mpe=("ape", "mean"),
        estimate_mean=("estimate", "mean"),
    )
    summary = per_rep.groupby(REGIME + ["method"], as_index=False).agg(
        n_reps=("rep", "nunique"),
        mpe_mean=("mpe", "mean"),
        mpe_sd=("mpe", "std"),
        estimate_mean=("estimate_mean", "mean"),
        estimate_sd=("estimate_mean", "std"),
    )
    summary = summary.merge(
        counts, on=REGIME, how="left", validate="many_to_one",
    )
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True,
    )
    return summary.sort_values(REGIME + ["method"]).reset_index(drop=True)


def paired_bootstrap(frame, n_boot=10000, seed=20260816):
    """Bootstrap paired differences by dataset, or by rep for one dataset."""
    frame = complete_case_frame(frame)
    generator = np.random.default_rng(seed)
    records = []
    grouped = frame.groupby(REGIME, sort=True)
    for regime, group in grouped:
        for reference, comparator in CLAIM_COMPARISONS:
            ref = group[group["method"] == reference][
                ["dataset", "rep", "ape"]
            ].rename(columns={"ape": "ape_reference"})
            other = group[group["method"] == comparator][
                ["dataset", "rep", "ape"]
            ].rename(columns={"ape": "ape_comparator"})
            paired = ref.merge(other, on=["dataset", "rep"], how="inner")
            paired["difference"] = (
                paired["ape_reference"] - paired["ape_comparator"]
            )
            n_datasets = paired["dataset"].nunique()
            if n_datasets > 1:
                differences = (
                    paired.groupby("dataset")["difference"].mean().to_numpy()
                )
                resampling_unit = "dataset"
            else:
                differences = paired["difference"].to_numpy()
                resampling_unit = "rep"
            if len(differences) == 0:
                continue
            indices = generator.integers(
                0, len(differences), size=(n_boot, len(differences)),
            )
            boot = differences[indices].mean(axis=1)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            records.append(dict(
                suite=regime[0],
                noise_sigma=regime[1],
                scale_sigma=regime[2],
                intervention=regime[3],
                reference=reference,
                comparator=comparator,
                n_pairs=len(paired),
                n_datasets=n_datasets,
                resampling_unit=resampling_unit,
                n_resampling_units=len(differences),
                delta_mpe=float(differences.mean()),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                ci_excludes_zero=bool(ci_high < 0 or ci_low > 0),
                supports_joint_improvement=bool(ci_high < 0),
                probability_reference_better=float(np.mean(boot < 0)),
                bootstrap_reps=n_boot,
                bootstrap_seed=seed,
            ))
    return pd.DataFrame(records)


def paired_intervention_bootstrap(frame, n_boot=10000, seed=20260826):
    """Compare latent-amplitude de-scaling with its paired raw GSM sample."""
    frame = complete_case_frame(frame)
    generator = np.random.default_rng(seed)
    records = []
    group_fields = ["suite", "noise_sigma", "scale_sigma", "dataset", "method"]
    for keys, group in frame.groupby(group_fields, sort=True):
        raw = group[group["intervention"] == "raw"][
            ["rep", "estimate", "ape"]
        ].rename(columns={"estimate": "estimate_raw", "ape": "ape_raw"})
        descaled = group[group["intervention"] == "latent_descaled"][
            ["rep", "estimate", "ape"]
        ].rename(columns={
            "estimate": "estimate_descaled", "ape": "ape_descaled",
        })
        paired = descaled.merge(raw, on="rep", how="inner", validate="one_to_one")
        if paired.empty:
            continue
        estimate_difference = (
            paired["estimate_descaled"] - paired["estimate_raw"]
        ).to_numpy()
        mpe_difference = (
            paired["ape_descaled"] - paired["ape_raw"]
        ).to_numpy()
        indices = generator.integers(
            0, len(paired), size=(n_boot, len(paired)),
        )
        estimate_boot = estimate_difference[indices].mean(axis=1)
        mpe_boot = mpe_difference[indices].mean(axis=1)
        estimate_low, estimate_high = np.percentile(estimate_boot, [2.5, 97.5])
        mpe_low, mpe_high = np.percentile(mpe_boot, [2.5, 97.5])
        records.append(dict(
            suite=keys[0], noise_sigma=keys[1], scale_sigma=keys[2],
            dataset=keys[3], method=keys[4], n_pairs=len(paired),
            reference_intervention="latent_descaled",
            comparator_intervention="raw",
            delta_estimate=float(estimate_difference.mean()),
            estimate_ci_low=float(estimate_low),
            estimate_ci_high=float(estimate_high),
            delta_mpe=float(mpe_difference.mean()),
            mpe_ci_low=float(mpe_low), mpe_ci_high=float(mpe_high),
            bootstrap_reps=n_boot, bootstrap_seed=seed,
        ))
    return pd.DataFrame(records)


def write_summaries(
    source, summary_out, bootstrap_out, intervention_bootstrap_out=None,
):
    frame = pd.read_csv(source)
    summary = aggregate(frame)
    bootstrap = paired_bootstrap(frame)
    for output, data in ((summary_out, summary), (bootstrap_out, bootstrap)):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        print("wrote", output)
    if intervention_bootstrap_out is not None:
        intervention = paired_intervention_bootstrap(frame)
        output = Path(intervention_bootstrap_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        intervention.to_csv(output, index=False)
        print("wrote", output)
    return summary, bootstrap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("bootstrap", type=Path)
    parser.add_argument("--intervention-bootstrap", type=Path)
    args = parser.parse_args(argv)
    write_summaries(
        args.source, args.summary, args.bootstrap,
        intervention_bootstrap_out=args.intervention_bootstrap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
