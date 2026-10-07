#!/usr/bin/env python
"""Regenerate the manuscript tables from the CSVs in ``results/``.

    python experiments/make_tables.py                      # -> results/generated/tables/
    python experiments/make_tables.py --paper-dir paper    # -> paper/tables/ (+ extras)

The eleven tables that the manuscript ``\\input``s go to ``--paper-dir``/tables
when that option is given, otherwise to ``results/generated/tables/``.
``table_impl.tex`` (implementation comparison, not included in the manuscript)
always goes to ``results/generated/tables/``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import add_paper_dir_argument, artifact_dirs
from paper_methods import load_canonical_cnn_results

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results" / "tables"
# Set by main(); defaults are the no-``--paper-dir`` locations.
OUT, INTERNAL_OUT = artifact_dirs(None, "tables")
PRIMARY_CORRECTED = ROOT / "results" / "impl_comparison" / \
    "impl_comparison.csv"

# Manuscript column names (code column -> displayed name)
COLS = {
    "DANCo": r"MiND--Full",
    "accDANCo": r"MiND--Full",     # same values as DANCo (one implementation); printed once
    "DANCo_Gride": r"Gride--Full",
    "DANCo_nustrip": r"MiND--Profiled",
    "DANCo_Gride_nustrip": r"Gride--Profiled",
    "2NN": r"TWO-NN",
    "2NN_ref": r"TWO-NN",
}

IMAGE_EST_COLS = [
    "aDANCo", "DANCo_G", "DANCo_nus", "DANCo_G_nus", "2NN_ref",
]
SENSITIVITY_EST_COLS = [
    "accDANCo", "DANCo_Gride", "DANCo_nustrip",
    "DANCo_Gride_nustrip", "2NN_ref",
]


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print("wrote", shown)


def esc(name):
    return name.replace("_", r"\_")


def image_tables():
    # CIFAR rows are keyed by class index (2=bird, 3=cat); MNIST rows by digit
    label = {("mnist", "3"): "MNIST digit 3", ("mnist", "7"): "MNIST digit 7",
             ("cifar", "2"): "CIFAR-10 bird", ("cifar", "3"): "CIFAR-10 cat",
             ("imagenet", "koala"): "ImageNet koala",
             ("imagenet", "butterfly"): "ImageNet butterfly"}
    est = IMAGE_EST_COLS
    # Two-line header keeps the eight columns inside the 10pt text width.
    lines = [r"\begin{tabular}{l rrrrr rr}",
             r"\toprule",
             r" & \multicolumn{2}{c}{Full} & \multicolumn{2}{c}{Profiled}"
             r" & & & \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Dataset & MiND & Gride & MiND & Gride & TWO-NN"
             r" & $\hat\nu$ & $\hat\tau$ \\",
             r"\midrule"]
    for grp in ["mnist", "cifar", "imagenet"]:
        ests = pd.read_csv(IN / f"image_estimates_{grp}.csv", index_col=0)
        stats = pd.read_csv(IN / f"image_statistics_{grp}.csv", index_col=0)
        ests.index = ests.index.map(str)
        stats.index = stats.index.map(str)
        for n in ests.index:
            vals = " & ".join(fmt(ests.loc[n, c]) for c in est)
            nu, tau = stats.loc[n, "nu"], stats.loc[n, "tau"]
            lines.append(rf"{label[(grp, n)]} & {vals} & {fmt(nu, 3)} & {fmt(tau, 1)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_images.tex", lines)


def diagnostics_table():
    """Image normalization diagnostics from ``angle_diagnostics.csv``.

    Each row is one image class.  The three variants report the data-side mean
    direction and Full/Profiled estimates averaged across calibration draws.
    Synthetic amplitude removal is reported separately because it uses the
    known generating amplitudes rather than an image normalization.
    """
    src = ROOT / "results" / "diagnostics" / "angle_diagnostics.csv"
    if not src.exists():
        print("skip diagnostics_table:", src, "missing")
        return
    df = pd.read_csv(src)
    if "seed" in df.columns:
        df = df.groupby(["dataset", "label", "variant"], as_index=False).mean(
            numeric_only=True)
    label = {("mnist", "3"): "MNIST 3", ("mnist", "7"): "MNIST 7",
             ("cifar", "2"): "CIFAR-10 bird", ("cifar", "3"): "CIFAR-10 cat",
             ("imagenet", "koala"): "ImageNet koala",
             ("imagenet", "butterfly"): "ImageNet butterfly"}
    order = [("mnist", "3"), ("mnist", "7"),
             ("cifar", "2"), ("cifar", "3"),
             ("imagenet", "koala"), ("imagenet", "butterfly")]
    df = df.set_index(["dataset", "label", "variant"])
    lines = [r"\begin{tabular}{l rrr rrr rrr}",
             r"\toprule",
             r" & \multicolumn{3}{c}{raw} & \multicolumn{3}{c}{centered radial}"
             r" & \multicolumn{3}{c}{contrast-norm.} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Dataset & $\hat\nu$ & Full & Profiled & $\hat\nu$ & Full & Profiled"
             r" & $\hat\nu$ & Full & Profiled \\",
             r"\midrule"]
    for ds, lab in order:
        cells = []
        for variant in ("raw", "sphere", "contrast"):
            try:
                row = df.loc[(ds, lab, variant)]
            except KeyError:
                row = df.loc[(ds, str(lab), variant)]
            cells.append(fmt(row["mu_nu"], 3))
            for col in ("de_standard", "de_nustrip"):
                v = row.get(col)
                cells.append("--" if pd.isna(v) else fmt(v, 1))
        lines.append(rf"{label[(ds, lab)]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_diagnostics.tex", lines)


def gsm_sweep_table():
    """Table S7: MiND estimates and the observed mean direction across the GSM
    amplitude sweep (30 replicates; raw samples only).  The known-amplitude
    control equals the ``scale_sigma == 0`` row by construction, so it needs no
    separate row."""
    src = ROOT / "results" / "diagnostics" / "fusion_ablation_gsm.csv"
    if not src.exists():
        print("skip gsm_sweep_table:", src, "missing")
        return
    df = pd.read_csv(src)
    df = df[(df["suite"] == "gsm") & (df["intervention"] == "raw")]
    per_rep = (df.drop_duplicates(["scale_sigma", "rep"])
                 .groupby("scale_sigma")["mu_nu"].agg(["mean", "std"]))
    mind = df[df["distance_model"] == "MiND"]
    cols = (
        ("Distance-only", mind[mind["objective"] == "distance_only"]),
        ("Full", mind[(mind["objective"] == "joint") & (mind["angle_mode"] == "standard")]),
        ("Profiled", mind[(mind["objective"] == "joint") & (mind["angle_mode"] == "profiled")]),
    )
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$\sigma_S$ & $\hat\nu$ & Distance-only & Full & Profiled \\",
             r"\midrule"]
    for sigma in sorted(df["scale_sigma"].unique()):
        label = f"{sigma:g}"
        cells = [rf"${per_rep.loc[sigma, 'mean']:.3f}\pm{per_rep.loc[sigma, 'std']:.3f}$"]
        for _, frame in cols:
            est = frame[frame["scale_sigma"] == sigma]["estimate"]
            if len(est) != 30:
                raise ValueError(f"gsm sweep: expected 30 replicates, got {len(est)}")
            cells.append(rf"${est.mean():.2f}\pm{est.std():.2f}$")
        lines.append(rf"${label}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_gsm_sweep.tex", lines)


def cnn_peak_summary(frame):
    """Median-curve peak and Full--Profiled gap for each CNN architecture."""
    required = {
        "arch", "layer_idx", "layer_name", "depth", "id_nustrip",
        "id_orig_diagnostic", "mu_nu", "delta_orig_minus_nustrip",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"CNN peak summary missing columns: {sorted(missing)}")

    rows = []
    for arch, arch_frame in frame.groupby("arch", sort=False):
        layers = arch_frame.groupby("layer_idx", as_index=False).agg(
            layer_name=("layer_name", "first"),
            depth=("depth", "first"),
            profiled=("id_nustrip", "median"),
            full=("id_orig_diagnostic", "median"),
            mu_nu=("mu_nu", "median"),
            full_minus_profiled=("delta_orig_minus_nustrip", "median"),
        )
        depth_max = max(float(layers["depth"].max()), 1.0)
        peak_idx = layers["profiled"].idxmax()
        gap_idx = layers["full_minus_profiled"].abs().idxmax()
        if peak_idx != gap_idx:
            raise ValueError(
                f"{arch}: largest Full--Profiled gap does not coincide "
                "with the Profiled peak"
            )
        peak = layers.loc[peak_idx]
        rows.append({
            "arch": arch,
            "layer_idx": int(peak["layer_idx"]),
            "layer_name": peak["layer_name"],
            "relative_depth": float(peak["depth"]) / depth_max,
            "mu_nu": float(peak["mu_nu"]),
            "full": float(peak["full"]),
            "profiled": float(peak["profiled"]),
            "full_minus_profiled": float(peak["full_minus_profiled"]),
        })
    return pd.DataFrame(rows)


def cnn_peak_table():
    """Peak-layer evidence behind the CNN calibration-sensitivity statement."""
    summary = cnn_peak_summary(load_canonical_cnn_results(
        ROOT / "results" / "cnn"
    ))
    arch_labels = {
        "alexnet": "AlexNet", "vgg16": "VGG-16",
        "resnet18": "ResNet-18", "resnet34": "ResNet-34",
    }
    summary["_order"] = summary["arch"].map(
        {name: i for i, name in enumerate(arch_labels)}
    )
    summary = summary.sort_values("_order")
    lines = [
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Architecture & Peak layer & depth (\%) & $\hat\nu$ & Full & "
        r"Profiled & Median paired Full$-$Profiled \\", r"\midrule",
    ]
    for _, row in summary.iterrows():
        layer_label = row["layer_name"]
        if layer_label == "MaxPool2d":
            ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth",
                       5: "fifth"}.get(int(row["layer_idx"]), str(row["layer_idx"]))
            layer_label = f"{ordinal} max-pooling checkpoint"
        elif layer_label in {"layer1", "layer2", "layer3", "layer4"}:
            ordinal = {"layer1": "first", "layer2": "second",
                       "layer3": "third", "layer4": "fourth"}[layer_label]
            layer_label = f"{ordinal} residual stage"
        lines.append(
            f"{arch_labels.get(row['arch'], row['arch'])} & "
            f"{layer_label} & {100 * row['relative_depth']:.1f} & "
            f"{row['mu_nu']:.3f} & {row['full']:.0f} & "
            f"{row['profiled']:.0f} & {row['full_minus_profiled']:.0f} "
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_cnn_peak_summary.tex", lines)


def cnn_endpoint_table():
    """Lower panel of Table S10: mean direction at the first pooling
    checkpoint and at the output, with the output Gride--Full / Gride--Profiled
    medians and the median paired Full$-$Profiled difference, all medians over
    category--subsample cells (same estimand as the peak panel)."""
    frame = load_canonical_cnn_results(ROOT / "results" / "cnn")
    arch_labels = {"alexnet": "AlexNet", "vgg16": "VGG-16",
                   "resnet18": "ResNet-18", "resnet34": "ResNet-34"}
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Architecture & $\hat\nu$ first pooling & $\hat\nu$ output & "
             r"Full output & Profiled output & Median paired Full$-$Profiled \\",
             r"\midrule"]
    for arch in arch_labels:
        f = frame[frame["arch"] == arch]
        first = f[f["layer_idx"] == 1]
        last = f[f["layer_idx"] == f["layer_idx"].max()]
        if first["layer_name"].iloc[0].lower() not in {"maxpool2d", "maxpool"}:
            raise ValueError(f"{arch}: layer 1 is not a pooling checkpoint")
        lines.append(
            f"{arch_labels[arch]} & {first['mu_nu'].median():.3f} & "
            f"{last['mu_nu'].median():.3f} & {last['id_orig_diagnostic'].median():.0f} & "
            f"{last['id_nustrip'].median():.0f} & "
            f"{last['delta_orig_minus_nustrip'].median():.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_cnn_endpoints.tex", lines)


def impl_table():
    """results/impl_comparison/impl_comparison.csv -> implementation table.

    Each implementation reports MPE and error rate, plus mean +- sd on the five
    manifolds where the implementations disagree most.
    """
    src = ROOT / "results" / "impl_comparison" / "impl_comparison.csv"
    if not src.exists():
        print("skip impl_table:", src, "missing")
        return
    df = pd.read_csv(src)
    df = df[df["status"] != "cal"].dropna(subset=["dhat"])
    # The primary corrected arm now has 20 replicates, whereas the defect
    # comparison remains a five-replicate paired experiment. Keep this table on
    # the intersection so stock and corrected rows see identical data draws.
    paired_impls = ("skdim_stock", "ours")
    shared_reps = set.intersection(*(
        set(df.loc[df["impl"] == impl, "rep"].astype(int))
        for impl in paired_impls
    ))
    df = df[df["rep"].astype(int).isin(shared_reps)]
    label = {"skdim_stock": r"\texttt{scikit-dimension} 0.3.7 (stock)",
             "ours": "corrected implementation",
             "ours_legacy_bessel": "corrected implementation, unscaled Bessel"}

    per = df.groupby(["impl", "manifold"]).agg(
        true_d=("true_d", "first"), mean=("dhat", "mean"),
        sd=("dhat", "std")).reset_index()
    per["rel_err"] = (per["mean"] - per["true_d"]).abs() / per["true_d"]

    summary = {}
    for impl, grp in per.groupby("impl"):
        raw = df[df["impl"] == impl]
        rel = (raw["dhat"] - raw["true_d"]).abs() / raw["true_d"]
        # MPE follows the paper's definition (and _impl_stats): average the
        # estimate over replicates within each manifold, then the relative
        # error, then the mean over manifolds. Error rate stays per-cell.
        summary[impl] = (100 * grp["rel_err"].mean(), (rel > 0.1).mean())

    wide = per.pivot(index="manifold", columns="impl", values="mean")
    gap = (wide["skdim_stock"] - wide["ours"]).abs().sort_values(
        ascending=False)
    top = gap.head(5).index

    lines = [r"\begin{tabular}{l r rr}",
             r"\toprule",
             r"Manifold & true $d$ & stock & corrected \\",
             r"\midrule"]
    truth = per.set_index("manifold")["true_d"].to_dict()
    sd = per.set_index(["impl", "manifold"])["sd"].to_dict()
    for name in top:
        lines.append(
            rf"{esc(name)} & {truth[name]:.0f} & "
            rf"{wide.loc[name, 'skdim_stock']:.1f} "
            rf"$\pm$ {sd[('skdim_stock', name)]:.1f} & "
            rf"{wide.loc[name, 'ours']:.1f} "
            rf"$\pm$ {sd[('ours', name)]:.1f} \\")
    lines += [r"\midrule"]
    for impl in ("skdim_stock", "ours"):
        mpe_v, err = summary[impl]
        lines.append(rf"\multicolumn{{2}}{{l}}{{{label[impl]}}} & "
                     rf"\multicolumn{{2}}{{l}}{{MPE {mpe_v:.2f}\%, "
                     rf"error rate {err:.3f}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(INTERNAL_OUT / "table_impl.tex", lines)


def sensitivity_table():
    """Combine the neighborhood- and sample-size-sensitivity MPE sweeps."""
    src = ROOT / "results" / "baselines" / "sensitivity.csv"
    if not src.exists():
        print("skip sensitivity_table:", src, "missing")
        return
    df = pd.read_csv(src)
    df["rel"] = (df["dhat"] - df["true_d"]).abs() / df["true_d"]
    cols = SENSITIVITY_EST_COLS
    head = " & ".join(COLS_SENS[c] for c in cols)
    lines = [r"\begin{tabular}{r rrrrr}", r"\toprule",
             rf"Setting & {head} \\", r"\midrule"]
    for mode, xcol, block in (
        ("k", "k", r"Neighborhood sweep ($N=2500$)"),
        ("n", "n", r"Sample-size sweep ($k=10$)"),
    ):
        sub = df[df["mode"] == mode]
        t = sub.groupby([xcol, "estimator"])["rel"].mean().unstack() * 100
        if mode == "n":
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{l}}{{{block}}} \\")
        for x, row in t.iterrows():
            lines.append(rf"{x:.0f} & " + " & ".join(
                fmt(row[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_sensitivity.tex", lines)


COLS_SENS = {
    "accDANCo": "MiND--Full",
    "DANCo_Gride": "Gride--Full",
    "DANCo_nustrip": r"MiND--Profiled",
    "DANCo_Gride_nustrip": r"Gride--Profiled",
    "2NN": "TWO-NN",
    "2NN_ref": "TWO-NN",
}


def primary_corrected_rows(frame, expected_reps=20, expected_manifolds=24):
    """Select and validate the corrected primary-benchmark result rows."""
    selected = frame[
        (frame["impl"] == "ours")
        & (frame["status"] != "cal")
        & (frame["manifold"] != "__calibration__")
    ].copy()
    expected = set(range(expected_reps))
    actual = set(int(rep) for rep in selected["rep"].unique())
    counts = selected.groupby("rep")["manifold"].nunique().to_dict()
    unique_cells = not selected.duplicated(["rep", "manifold"]).any()
    if (actual != expected or not unique_cells
            or any(counts.get(rep) != expected_manifolds for rep in expected)):
        raise ValueError(
            f"primary corrected benchmark requires {expected_reps} complete "
            f"replicates x {expected_manifolds} manifolds; got reps={sorted(actual)} "
            f"and per-rep counts={counts}"
        )
    return selected


def estimator_summary(frame):
    """Paper MPE, failure-inclusive error rate, failures, and median time."""
    relative = (frame["dhat"] - frame["true_d"]).abs() / frame["true_d"]
    failed = frame["dhat"].isna() | ~np.isfinite(frame["dhat"])
    error_rate = float(((relative > 0.1) | failed).mean())
    n_fail = int(failed.sum())
    per_manifold = frame.groupby("manifold").apply(
        lambda group: (
            abs(group["dhat"].dropna().mean() - group["true_d"].iloc[0])
            / group["true_d"].iloc[0]
            if group["dhat"].notna().any() else np.nan
        ),
        include_groups=False,
    )
    mpe = float(100.0 / len(per_manifold) * np.nansum(per_manifold))
    return {
        "mpe": mpe,
        "error_rate": error_rate,
        "n_fail": n_fail,
        "median_seconds": float(frame["seconds"].median()),
    }


def combined_baseline_cells(baseline_frame, corrected_frame):
    """Per-manifold means for nine baselines plus corrected DANCo."""
    baseline = baseline_frame.dropna(subset=["dhat"])
    per_baseline = baseline.groupby(["estimator", "manifold"]).agg(
        true_d=("true_d", "first"), mean=("dhat", "mean"),
        sd=("dhat", "std"),
    ).reset_index()
    corrected = corrected_frame.dropna(subset=["dhat"]).groupby(
        "manifold",
    ).agg(
        true_d=("true_d", "first"), mean=("dhat", "mean"),
        sd=("dhat", "std"),
    ).reset_index()
    corrected.insert(0, "estimator", "DANCo_corrected")
    return pd.concat([corrected, per_baseline], ignore_index=True)


def baselines_tables():
    """results/baselines/baselines_benchmark.csv -> baseline tables.

    Main summary table: MPE, error rate and median time per cell for each
    baseline, together with MiND--Full (the ``ours`` rows of impl_comparison,
    same data-generation protocol).  Supplementary table: mean +- sd for all
    24 manifolds x 9 baselines.
    """
    src = ROOT / "results" / "baselines" / "baselines_benchmark.csv"
    if not src.exists():
        print("skip baselines_tables:", src, "missing")
        return
    df = pd.read_csv(src)

    display = {"TwoNN": "TWO-NN", "MiND_ML": "MiND--ML"}
    rows_summary = []
    for est, grp in df.groupby("estimator"):
        stats = estimator_summary(grp)
        rows_summary.append((display.get(est, est), stats["mpe"],
                             stats["error_rate"], stats["median_seconds"],
                             stats["n_fail"]))
    # MiND--Full: the same data seeds 1000--1019 as the nine baselines.
    impl = PRIMARY_CORRECTED
    if not impl.exists():
        print("warn: impl_comparison.csv missing; the DANCo (corrected impl.)"
              " row of table_baselines_main will be omitted")
    io = pd.DataFrame()
    if impl.exists():
        io = primary_corrected_rows(pd.read_csv(impl))
        stats = estimator_summary(io)
        rows_summary.append((r"MiND--Full (DANCo)", stats["mpe"],
                             stats["error_rate"], stats["median_seconds"],
                             stats["n_fail"]))
    rows_summary.sort(key=lambda r: r[1])

    lines = [r"\begin{tabular}{l rrrr}", r"\toprule",
             r"Estimator & MPE (\%) & $>10\%$ error & failed & median time (s) \\",
             r"\midrule"]
    for est, mpe_v, err, med_s, n_fail in rows_summary:
        name = est.replace("_", r"\_")
        time_text = r"$<0.01$" if med_s < 0.005 else fmt(med_s)
        lines.append(" & ".join([name, fmt(mpe_v), fmt(err, 3), str(n_fail),
                                 time_text])
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_baselines_main.tex", lines)

    per = combined_baseline_cells(df, io)
    ests = sorted(per["estimator"].unique())
    full_labels = {
        "DANCo_corrected": "MiND--Full (DANCo)",
        "MiND_ML": "MiND--ML",
        "TwoNN": "TWO-NN",
    }
    lines = [r"\begin{tabular}{l r " + "r" * len(ests) + "}", r"\toprule",
             r"Manifold & $d$ & " + " & ".join(
                 full_labels.get(e, e).replace("_", r"\_") for e in ests)
             + r" \\",
             r"\midrule"]
    piv_m = per.pivot(index="manifold", columns="estimator", values="mean")
    truth = per.groupby("manifold")["true_d"].first()
    for name in sorted(piv_m.index):
        cells = [fmt(piv_m.loc[name, e], 1) for e in ests]
        lines.append(rf"{esc(name)} & {truth[name]:.0f} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_baselines_full.tex", lines)


def class_survey_table():
    """Build the Supplementary table for additional image classes."""
    src = ROOT / "results" / "std_datasets" / "class_survey.csv"
    if not src.exists():
        print("skip class_survey_table:", src, "missing")
        return
    df = pd.read_csv(src).sort_values(["source", "class"])
    source_labels = {"cifar10": "CIFAR-10", "imagenet": "ImageNet"}
    lines = [r"\begin{tabular}{ll rr rrr}", r"\toprule",
             r"Source & Class & $N$ & $D$ & $\hat\nu$ & $\operatorname{CV}(a)$ & "
             r"$r(\hat\nu_i, \|x_i-\bar x\|)$ \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cls = str(r["class"]).split("_", 1)[-1].replace("_", r"\_")
        lines.append(
            rf"{source_labels.get(r['source'], r['source'])} & {cls} & "
            rf"{int(r['N'])} & {int(r['D'])} & "
            rf"{fmt(r['mu_nu'], 3)} & {fmt(r['norm_cv'], 3)} & "
            rf"{fmt(r['pearson_r'], 2)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_class_survey.tex", lines)


def relative_noise_table():
    """Dimensionless noise sweep used for the primary robustness claim."""
    source = ROOT / "results" / "diagnostics" / \
        "relative_noise_ablation_summary.csv"
    if not source.exists():
        print("skip relative_noise_table:", source, "missing")
        return
    frame = pd.read_csv(source).set_index(["noise_eta", "method"])
    # Column order matches Tables 2, S5, and S6.
    methods = ("danco", "danco_gride", "profiled_danco",
               "profiled_danco_gride")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"$\eta$ & MiND--Full & Gride--Full & MiND--Profiled & Gride--Profiled \\",
        r"\midrule",
    ]
    for eta in sorted(frame.index.get_level_values("noise_eta").unique()):
        cells = []
        for method in methods:
            row = frame.loc[(eta, method)]
            cells.append(rf"{row['mpe_mean']:.2f} $\pm$ {row['mpe_sd']:.2f}")
        lines.append(rf"{eta:g} & " + " & ".join(cells) + r" \\")
    # Restricted subset at the largest noise level: the seven manifolds with
    # d > 5 whose ambient dimension caps the candidate search at D <= d + 1
    # are excluded, leaving 17 manifolds.
    raw_src = ROOT / "results" / "diagnostics" / "relative_noise_ablation.csv"
    raw = pd.read_csv(raw_src)
    capped = raw.groupby("dataset")[["true_d", "ambient_d"]].first()
    capped = capped[(capped["true_d"] > 5) & (capped["ambient_d"] <= capped["true_d"] + 1)]
    if len(capped) != 7:
        raise ValueError(f"expected 7 capped manifolds, found {len(capped)}")
    eta_max = raw["noise_eta"].max()
    sub = raw[(raw["noise_eta"] == eta_max) & (raw["objective"] == "joint")
              & ~raw["dataset"].isin(capped.index)]
    sub = sub.assign(pe=(sub["estimate"] - sub["true_d"]).abs() / sub["true_d"] * 100)
    method_key = {("MiND", "standard"): "danco", ("Gride", "standard"): "danco_gride",
                  ("MiND", "profiled"): "profiled_danco",
                  ("Gride", "profiled"): "profiled_danco_gride"}
    per_rep = (sub.groupby(["distance_model", "angle_mode", "rep"])["pe"].mean()
                  .groupby(level=[0, 1]).agg(["mean", "std", "count"]))
    n_left = sub["dataset"].nunique()
    cells = []
    for method in methods:
        key = next(k for k, v in method_key.items() if v == method)
        row = per_rep.loc[key]
        if row["count"] != 10:
            raise ValueError("restricted subset: expected 10 replicates")
        cells.append(rf"{row['mean']:.2f} $\pm$ {row['std']:.2f}")
    lines += [
        r"\midrule",
        rf"\multicolumn{{5}}{{l}}{{Restricted subset: {n_left} manifolds (excluding $d>5$ with $D\leq d+1$)}} \\",
        rf"{eta_max:g} & " + " & ".join(cells) + r" \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    write(OUT / "table_relative_noise.tex", lines)

    # Lower panel of Table S5: signed error for all eight objectives on the
    # 14-manifold complete-case set, so that every number quoted in the
    # Supplementary noise paragraph can be read from the table.
    signed_src = ROOT / "results" / "diagnostics" / \
        "fusion_ablation_signed_bias_by_replicate.csv"
    if not signed_src.exists():
        print("skip relative_noise signed panel:", signed_src, "missing")
        return
    signed = pd.read_csv(signed_src)
    signed = signed[signed["source"] == "relative_noise"]
    signed = signed.set_index(["method", "noise_eta"])
    etas = sorted(signed.index.get_level_values("noise_eta").unique())
    # Row order matches the clean component table (tab:clean-components-supp).
    objectives = (
        ("mind_distance_only", "MiND distance-only"),
        ("gride_distance_only", "Gride distance-only"),
        ("standard_angle_only", "Full angle-only"),
        ("profiled_angle_only", "Profiled angle-only"),
        ("danco", "MiND--Full"),
        ("profiled_danco", "MiND--Profiled"),
        ("danco_gride", "Gride--Full"),
        ("profiled_danco_gride", "Gride--Profiled"),
    )
    head = " & ".join(rf"$\eta={eta:g}$" for eta in etas)
    lines = [
        r"\begin{tabular}{l" + "r" * len(etas) + "}",
        r"\toprule",
        rf"Objective & {head} \\",
        r"\midrule",
    ]
    for method, label in objectives:
        cells = []
        for eta in etas:
            row = signed.loc[(method, eta)]
            cells.append(rf"${row['signed_pe_mean']:+.1f}\pm{row['signed_pe_sd']:.1f}$")
        lines.append(rf"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(OUT / "table_relative_noise_signed.tex", lines)


def main(argv=None):
    global OUT, INTERNAL_OUT
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_paper_dir_argument(parser)
    args = parser.parse_args(argv)
    OUT, INTERNAL_OUT = artifact_dirs(args.paper_dir, "tables")
    image_tables()
    diagnostics_table()
    cnn_peak_table()
    cnn_endpoint_table()
    gsm_sweep_table()
    impl_table()
    sensitivity_table()
    baselines_tables()
    class_survey_table()
    relative_noise_table()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
