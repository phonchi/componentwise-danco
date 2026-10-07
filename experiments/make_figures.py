#!/usr/bin/env python
"""Regenerate every paper figure from the CSVs in ``results/``.

    python experiments/make_figures.py                        # all figures
    python experiments/make_figures.py --figures fusionmain   # one figure
    python experiments/make_figures.py --paper-dir paper      # into paper/figures/

Figures are derived artefacts: nothing here reads a hand-entered number, so a
figure can always be rebuilt from the table it illustrates.  A figure whose
source CSV does not exist yet is skipped with a note rather than failing the
run.  The figures the manuscript includes go to ``--paper-dir``/figures when
that option is given, otherwise to ``results/generated/figures/``;
``fig_kl_diagnosis_main.pdf`` (not included in the manuscript) always goes to
``results/generated/figures/``.

Design constraints kept deliberately narrow, so the figures read as one set: a fixed estimator -> colour assignment (Okabe-Ito, validated for
colour-vision deficiency), one axis per panel, recessive grid, direct value
labels wherever a reader would otherwise have to measure a bar against an axis,
and the manuscript's Times New Roman for every label (``common.use_manuscript_font``).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (REPO_ROOT, add_paper_dir_argument, artifact_dirs,
                    use_manuscript_font)
from paper_methods import load_canonical_cnn_results

CNN_RESULTS = REPO_ROOT / "results" / "cnn"
# Set by main(); defaults are the no-``--paper-dir`` locations.
FIGURES, INTERNAL_FIGURES = artifact_dirs(None, "figures")

# Fixed order and fixed hues: an estimator keeps its colour across all figures,
# and a figure that omits one does not repaint the others.
ESTIMATOR_COLORS = {
    "DANCo": "#0072B2",
    "accDANCo": "#0072B2",
    "DANCo_Gride": "#E69F00",
    "DANCo_nustrip": "#009E73",
    "2NN": "#D55E00",
    "2NN_ref": "#D55E00",
    "DANCo_fast": "#CC79A7",
}
GRID = dict(color="#d8d8d4", linewidth=0.6)
INK = "#2b2b28"

PI_2, PI_3 = np.pi / 2, np.pi / 3
DIAG = REPO_ROOT / "results" / "diagnostics"
# Blue = isotropic/reference geometry; vermillion = scale-heterogeneous data.
C_ISO = "#0072B2"
C_HET = "#D55E00"


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b9b4")
    ax.tick_params(colors=INK, labelsize=9, length=3)
    return ax


def _save(fig, name, directory=None):
    directory = FIGURES if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / name
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
def _refline(ax, y, label, side="right", fontsize=8):
    ax.axhline(y, color="#b9b9b4", linewidth=0.9, linestyle=(0, (4, 3)))
    if side == "left":
        ax.text(0.02, y, label, transform=ax.get_yaxis_transform(),
                fontsize=fontsize, color=INK, ha="left", va="bottom")
        return
    ax.annotate(label, (ax.get_xlim()[1], y), fontsize=fontsize, color=INK,
                ha="right", va="bottom")


def figure_nu_mechanism():
    """Unique Supplementary center-level amplitude diagnostic."""
    import matplotlib.pyplot as plt

    path = DIAG / "nu_toy_gsm_profile.csv"
    if not path.exists():
        print("skip nu mechanism figure: center-level profile missing")
        return None
    prof = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    _style(ax)
    # Deciles are equal-count bins, so the arithmetic mean of the decile
    # midpoints equals the sample mean norm; dividing by it puts this panel on
    # the same normalized amplitude axis as fig_nu_scatter.
    norm_rel = prof["norm_mid"] / prof["norm_mid"].mean()
    ax.axhspan(0.42, PI_3, color=C_HET, alpha=0.06, lw=0)
    ax.errorbar(norm_rel, prof["nu_mean"], yerr=prof["nu_sd"],
                marker="o", markersize=3.5, linewidth=2, color=C_HET,
                elinewidth=1, capsize=2)
    ax.set_xlabel(r"normalized centered norm $a_i/\bar a$", fontsize=8, color=INK)
    ax.set_ylabel(r"$\hat\nu$ (rad)", fontsize=8, color=INK)
    ax.set_title(r"Center-level GSM geometry, $\sigma_S{=}0.25$", fontsize=9,
                 color=INK, loc="left")
    ax.set_ylim(0.42, 1.12)
    _refline(ax, PI_3, r"$\pi/3$", side="right")

    fig.tight_layout()
    return _save(fig, "fig_nu_mechanism.pdf")


def figure_nu_mechanism_main():
    """Focused main-text view of dimension and amplitude effects on nu."""
    import matplotlib.pyplot as plt

    ball_p = DIAG / "nu_toy_ball.csv"
    gsm_p = DIAG / "nu_toy_gsm.csv"
    if not (ball_p.exists() and gsm_p.exists()):
        print("skip focused nu mechanism figure: toy sweep CSVs not found")
        return None
    ball, gsm = pd.read_csv(ball_p), pd.read_csv(gsm_p)

    # Rendered at \linewidth (341pt) after the tight bounding box, the
    # 6.4in figure is scaled by about 0.67; native 12/13pt text renders at
    # the journal caption size (8/8.7pt).
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55))
    for ax in axes:
        _style(ax)
        ax.tick_params(labelsize=12)

    ax = axes[0]
    ax.set_facecolor("#f6f8fa")
    ax.plot(ball["d"], ball["nu"], marker="o", markersize=3.5,
            linewidth=2, color=C_ISO)
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xticks([2, 10, 50, 400])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("ball dimension $d$ (log scale)", fontsize=12, color=INK)
    ax.set_ylabel(r"$\hat\nu$ (rad)", fontsize=12, color=INK)
    ax.set_title("(a) homogeneous amplitudes", fontsize=13, color=INK,
                 loc="left")
    ax.set_ylim(0.40, 1.65)
    _refline(ax, PI_2, r"$\pi/2$", side="left", fontsize=12)
    _refline(ax, PI_3, r"$\pi/3$", side="left", fontsize=12)

    ax = axes[1]
    ax.axhspan(0.40, PI_3, color=C_HET, alpha=0.06, lw=0)
    ax.plot(gsm["sigma_s"], gsm["nu"], marker="o", markersize=3.5,
            linewidth=2, color=C_HET)
    ax.set_xlabel(r"log-amplitude spread $\sigma_S$", fontsize=12, color=INK)
    ax.set_ylabel(r"$\hat\nu$ (rad)", fontsize=12, color=INK)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_title(r"(b) heterogeneous amplitudes, $d{=}70$", fontsize=13,
                 color=INK, loc="left")
    ax.set_ylim(0.40, 1.65)
    ax.margins(x=0.14)
    _refline(ax, PI_2, r"$\pi/2$", side="left", fontsize=12)
    _refline(ax, PI_3, r"$\pi/3$", side="left", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.30)
    return _save(fig, "fig_nu_mechanism_main.pdf")


def figure_diagnostics_scatter():
    """(e) Per-point nu_i against centered norm, before/after normalization."""
    import matplotlib.pyplot as plt

    raw_p = DIAG / "perpoint" / "imagenet_koala_raw.npz"
    con_p = DIAG / "perpoint" / "imagenet_koala_contrast.npz"
    if not (raw_p.exists() and con_p.exists()):
        print("skip diagnostics scatter: per-point npz not found")
        return None
    raw, con = np.load(raw_p), np.load(con_p)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for ax, z, color, title in (
            (axes[0], raw, C_HET, "(a) ImageNet koala, raw"),
            (axes[1], con, C_ISO, "(b) contrast-normalized")):
        _style(ax)
        norms, nu_i = z["norms"], z["nu_i"]
        r = np.corrcoef(norms, nu_i)[0, 1]
        ax.scatter(norms / norms.mean(), nu_i, s=7, alpha=0.35, color=color,
                   linewidths=0)
        coeffs = np.polyfit(norms / norms.mean(), nu_i, 1)
        xs = np.linspace((norms / norms.mean()).min(),
                         (norms / norms.mean()).max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color=INK, linewidth=1.4)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.annotate(f"$r={r:+.2f}$", (0.97, 0.92), xycoords="axes fraction",
                    fontsize=9, color=INK, ha="right")
        ax.set_xlabel(r"normalized centered norm $a_i/\bar a$", fontsize=8, color=INK)
        _refline(ax, PI_3, r"$\pi/3$")
    axes[0].set_ylabel(r"$\hat\nu_i$ (rad)", fontsize=8, color=INK)
    fig.tight_layout()
    return _save(fig, "fig_nu_scatter.pdf")


def figure_kl_diagnosis():
    """(f) Where the argmin goes: KL curves on a working and a failing dataset."""
    import matplotlib.pyplot as plt

    ok_p = DIAG / "perpoint" / "mnist_3_raw.npz"
    bad_p = DIAG / "perpoint" / "imagenet_koala_raw.npz"
    if not (ok_p.exists() and bad_p.exists()):
        print("skip KL diagnosis figure: per-point npz not found")
        return None
    panels = [("(a) MNIST 3 (inside reference range)", np.load(ok_p)),
              ("(b) ImageNet koala (location mismatch)", np.load(bad_p))]
    if any("kl_curves" not in z for _, z in panels):
        print("skip KL diagnosis figure: npz lacks KL curves")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for ax, (title, z) in zip(axes, panels):
        _style(ax)
        kl = z["kl_curves"]
        strip = z["kl_strip"]
        m = np.arange(1, kl.shape[0] + 1)
        total, total_strip = kl.sum(axis=1), kl[:, 0] + strip
        ax.plot(m, kl[:, 0], linewidth=1.7, color="#56B4E9",
                linestyle="-", marker="o", markersize=3, markevery=10,
                label="distance")
        ax.plot(m, kl[:, 1], linewidth=1.7, color=C_HET,
                linestyle=":", marker="s", markersize=3, markevery=10,
                label="angular")
        ax.plot(m, total, linewidth=2.2, color=INK,
                linestyle="-.", marker="^", markersize=3.5, markevery=10,
                label="Full")
        ax.plot(m, total_strip, linewidth=2.2, color="#009E73",
                linestyle="--", marker="D", markersize=3, markevery=10,
                label="Profiled")
        for curve, color, marker in (
                (total, INK, "v"), (total_strip, "#009E73", "D")):
            am = int(np.argmin(curve))
            ax.plot(m[am], curve[am], marker=marker, markersize=7,
                    color=color, markeredgecolor=INK, markeredgewidth=0.5,
                    zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("candidate dimension $m$", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("KL", fontsize=8, color=INK)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right",
                   handlelength=3.4, numpoints=2, markerscale=1.15)
    fig.tight_layout()
    return _save(fig, "fig_kl_diagnosis.pdf")


def figure_kl_diagnosis_main():
    """Focused main-text KL diagnosis for the ImageNet koala class."""
    import matplotlib.pyplot as plt

    path = DIAG / "perpoint" / "imagenet_koala_raw.npz"
    if not path.exists():
        print(f"skip focused KL diagnosis figure: {path} not found")
        return None
    z = np.load(path)
    if "kl_curves" not in z or "kl_strip" not in z:
        print("skip focused KL diagnosis figure: npz lacks KL curves")
        return None

    kl = z["kl_curves"]
    strip = z["kl_strip"]
    m = np.arange(1, kl.shape[0] + 1)
    total, total_strip = kl.sum(axis=1), kl[:, 0] + strip

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    _style(ax)
    ax.plot(m, kl[:, 0], linewidth=1.7, color="#56B4E9",
            linestyle="-", marker="o", markersize=3, markevery=10,
            label="distance")
    ax.plot(m, kl[:, 1], linewidth=1.7, color=C_HET,
            linestyle=":", marker="s", markersize=3, markevery=10,
            label="angular")
    ax.plot(m, total, linewidth=2.2, color=INK,
            linestyle="-.", marker="^", markersize=3.5, markevery=10,
            label="Full")
    ax.plot(m, total_strip, linewidth=2.2, color="#009E73",
            linestyle="--", marker="D", markersize=3, markevery=10,
            label="Profiled")
    for curve, color, marker in (
            (total, INK, "v"), (total_strip, "#009E73", "D")):
        am = int(np.argmin(curve))
        ax.plot(m[am], curve[am], marker=marker, markersize=7,
                color=color, markeredgecolor=INK, markeredgewidth=0.5,
                zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("candidate dimension $m$", fontsize=8, color=INK)
    ax.set_ylabel("KL", fontsize=8, color=INK)
    ax.set_title("ImageNet koala: angular-reference mismatch", fontsize=9,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right",
              handlelength=3.4, numpoints=2, markerscale=1.15)
    fig.tight_layout()
    return _save(fig, "fig_kl_diagnosis_main.pdf", INTERNAL_FIGURES)


def figure_overflow():
    """(g) The silent calibration-side Bessel overflow."""
    import matplotlib.pyplot as plt

    path = DIAG / "overflow_candidate_curve.csv"
    if not path.exists():
        print(f"skip overflow figure: {path} not found")
        return None
    df = pd.read_csv(path)
    df = df[df["calibration_n"] == 500].sort_values("candidate_dim")
    if df.empty:
        print("skip overflow figure: no calibration_n=500 rows")
        return None

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    _style(ax)
    ax.plot(df["candidate_dim"], df["kl_scaled"], linewidth=2.2, color=C_ISO,
            linestyle="-", marker="o", markersize=3, markevery=40,
            label="scaled Bessel (ours)")
    # Stop the dashed path where it first crosses zero.  Some PDF renderers draw
    # a spurious end-to-end segment when a dashed path crosses the linear core
    # of a symmetric-log axis.  Dense square markers continue the failed values
    # without introducing that rendering artefact.
    negative = np.flatnonzero(df["kl_legacy"].to_numpy() < 0)
    split = int(negative[0]) if len(negative) else len(df)
    stable_segment = df.iloc[:split]
    failed_segment = df.iloc[max(split - 1, 0):]
    ax.plot(
        stable_segment["candidate_dim"], stable_segment["kl_legacy"],
        linewidth=1.8, color=C_HET, linestyle="--", marker="s",
        markersize=3, markevery=40, label="unscaled (reference impl.)",
    )
    ax.scatter(
        failed_segment["candidate_dim"], failed_segment["kl_legacy"],
        s=7, color=C_HET, marker="s", linewidths=0, zorder=3,
    )
    affected = df[df["tau_ref"] > 709.78]
    crossing = int(affected.iloc[0]["candidate_dim"])
    ax.axvline(crossing, color="#b9b9b4", linewidth=0.9,
               linestyle=(0, (4, 3)))
    ax.annotate(rf"ordinary Bessel overflows: $m={crossing}$",
                (crossing, ax.get_ylim()[1]),
                fontsize=8, color=INK, ha="right", va="top", rotation=90,
                xytext=(-4, -2), textcoords="offset points")
    ax.axhline(0, color="#b9b9b4", linewidth=0.8)
    ax.set_xlabel(r"candidate dimension $m$ ($N_{\mathrm{ref}}=500$)", fontsize=8,
                  color=INK)
    ax.set_ylabel("von Mises KL", fontsize=8, color=INK)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handlelength=3.4, numpoints=2, markerscale=1.15)
    return _save(fig, "fig_overflow.pdf")


def figure_vm_adequacy():
    """(supp) von Mises fit quality on uniform-ball and image data."""
    import matplotlib.pyplot as plt

    names = [("ball70", "uniform ball $d{=}70$", C_ISO),
             ("koala_raw", "ImageNet koala (raw)", C_HET)]
    if not all((DIAG / f"vm_adequacy_{n}.npz").exists() for n, _, _ in names):
        print("skip VM adequacy figure: npz not found")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (name, title, color) in zip(axes, names):
        _style(ax)
        z = np.load(DIAG / f"vm_adequacy_{name}.npz")
        centers = (z["edges"][:-1] + z["edges"][1:]) / 2
        ax.fill_between(centers, z["hist"], step="mid", alpha=0.35,
                        color=color, linewidth=0, label="empirical")
        # Redraw the fitted von Mises density on the same grid.
        from scipy.special import i0e
        nu, tau = float(z["nu"]), float(z["tau"])
        dens = np.exp(tau * (np.cos(centers - nu) - 1.0)) / i0e(tau)
        dens /= np.trapz(dens, centers)
        ax.plot(centers, dens, color=INK, linewidth=1.6, label="VM fit")
        gap = float(z["max_cdf_gap"])
        ax.annotate(f"max CDF gap {gap:.3f}", (0.03, 0.92),
                    xycoords="axes fraction", fontsize=8, color=INK)
        ax.set_xlim(max(0, nu - 0.6), min(np.pi, nu + 0.6))
        ax.set_xlabel(r"pairwise angle $\theta$ (rad)", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        # One legend per panel: the empirical swatch takes that panel's colour.
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("density", fontsize=8, color=INK)
    fig.tight_layout()
    return _save(fig, "fig_vm_adequacy.pdf")


def figure_noise_curves():
    """(h) MPE against neighborhood-relative Gaussian noise."""
    import matplotlib.pyplot as plt

    path = DIAG / "relative_noise_ablation_summary.csv"
    if not path.exists():
        print(f"skip noise curves: {path} not found")
        return None
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    _style(ax)
    # This panel is printed at 0.65 text width; native 9pt labels shrink below 6pt.
    ax.tick_params(which="both", labelsize=13.5)
    order = [
        ("danco", "MiND–Full", C_ISO, "-", "o"),
        ("danco_gride", "Gride–Full", C_HET, "--", "s"),
        ("profiled_danco", "MiND–Profiled", "#009E73", "-.", "^"),
        ("profiled_danco_gride", "Gride–Profiled", "#CC79A7", ":", "D"),
    ]
    for method, label, color, linestyle, marker in order:
        sub = df[df["method"] == method].sort_values("noise_eta")
        if sub.empty:
            continue
        ax.plot(sub["noise_eta"], sub["mpe_mean"], marker=marker,
                markersize=4.5, linewidth=2, linestyle=linestyle,
                color=color, label=label)
        ax.fill_between(sub["noise_eta"],
                        sub["mpe_mean"] - sub["mpe_sd"],
                        sub["mpe_mean"] + sub["mpe_sd"],
                        color=color, alpha=0.15,
                        linewidth=0)
    ax.set_xlabel(r"relative noise level $\eta$", fontsize=13.5, color=INK)
    ax.set_ylabel("MPE (%)", fontsize=13.5, color=INK)
    ax.legend(frameon=False, fontsize=13.5, loc="upper left", handlelength=3.4,
              numpoints=2, markerscale=1.15)
    return _save(fig, "fig_noise_curves.pdf")


def figure_cnn_paired():
    """(i) Paired estimator profiles across CNN layers, one panel per arch."""
    import matplotlib.pyplot as plt

    df = load_canonical_cnn_results(CNN_RESULTS)
    archs = [a for a in ("alexnet", "vgg16", "resnet18", "resnet34")
             if a in set(df["arch"])]
    series = [
        ("id_nustrip", "DANCo_nustrip", "Gride–Profiled", "-.", "^"),
        ("twonn_ref", "2NN", "TWO-NN", "--", "s"),
        ("id_mle", "MLE", "MLE", ":", "D"),
    ]
    colors = dict(ESTIMATOR_COLORS, MLE="#CC79A7")

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9), squeeze=False)
    for ax, arch in zip(axes.ravel(), archs):
        _style(ax)
        sub = df[df["arch"] == arch]
        depth_max = max(sub["depth"].max(), 1)
        for col, key, label, linestyle, marker in series:
            # Median with IQR band: the spread is cross-category and heavy
            # tailed, so a mean +- sd band would cross zero on the log axis.
            grouped = sub.groupby("layer_idx").agg(
                depth=("depth", "first"), med=(col, "median"),
                lo=(col, lambda v: v.quantile(0.25)),
                hi=(col, lambda v: v.quantile(0.75))).sort_index()
            rel = grouped["depth"] / depth_max
            ax.plot(rel, grouped["med"], marker=marker, markersize=4,
                    linewidth=2, linestyle=linestyle, color=colors[key],
                    label=label)
            ax.fill_between(rel, grouped["lo"], grouped["hi"],
                            color=colors[key], alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        # The four-panel figure is reduced to 0.9\linewidth (scale 0.645):
        # native 12.5pt ticks render at 8pt. Label only explicit major
        # ticks; the automatic minor log labels would render below 6pt.
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.set_yticks([10, 20, 50, 100, 200, 400])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", labelsize=12.5)
        ax.set_title(arch, fontsize=13, color=INK, loc="left")
        ax.set_xlabel("relative depth", fontsize=13, color=INK)
        ax.set_ylabel("ID", fontsize=13, color=INK)
    for ax in axes.ravel()[len(archs):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=13, loc="lower center",
               ncols=3, bbox_to_anchor=(0.5, 0.01), handlelength=3.4,
               numpoints=2, markerscale=1.15)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, "fig_cnn_paired.pdf")


def figure_error_vs_d():
    """(j) Signed relative error against true dimension, DANCo vs baselines."""
    import matplotlib.pyplot as plt

    base_p = REPO_ROOT / "results" / "baselines" / "baselines_benchmark.csv"
    impl_p = REPO_ROOT / "results" / "impl_comparison" / "impl_comparison.csv"
    if not (base_p.exists() and impl_p.exists()):
        print("skip error-vs-d figure: source CSVs not found")
        return None
    base = pd.read_csv(base_p).dropna(subset=["dhat"])
    ours = pd.read_csv(impl_p)
    ours = ours[(ours["impl"] == "ours") & (ours["status"] != "cal")].dropna(
        subset=["dhat"])
    ours["estimator"] = "MiND–Full (DANCo)"
    df = pd.concat([base[["manifold", "true_d", "estimator", "dhat"]],
                    ours[["manifold", "true_d", "estimator", "dhat"]]])
    df["rel"] = (df["dhat"] - df["true_d"]) / df["true_d"]

    show = [
        ("MiND–Full (DANCo)", "#009E73", "-", "o"),
        ("TwoNN", "#D55E00", "--", "s"),
        ("ESS", "#0072B2", "-.", "^"),
        ("MLE", "#CC79A7", ":", "D"),
        ("MiND_ML", "#E69F00", (0, (5, 1, 1, 1)), "P"),
    ]
    label = {"TwoNN": "TWO-NN", "MiND_ML": "MiND–ML"}

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    _style(ax)
    ax.axhline(0, color="#b9b9b4", linewidth=0.9)
    for est, color, linestyle, marker in show:
        sub = df[df["estimator"] == est]
        per = sub.groupby("manifold").agg(d=("true_d", "first"),
                                          rel=("rel", "mean"))
        curve = per.groupby("d")["rel"].mean().sort_index() * 100
        ax.plot(curve.index, curve.values, marker=marker, markersize=4.5,
                linewidth=2, linestyle=linestyle, color=color,
                label=label.get(est, est))
        ax.scatter(per["d"], per["rel"] * 100, s=10, alpha=0.35,
                   color=color, linewidths=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10, 20, 40, 70])
    ax.set_xticklabels(["1", "2", "4", "10", "20", "40", "70"])
    ax.set_xlabel("true dimension $d$", fontsize=9, color=INK)
    ax.set_ylabel("signed relative error (%)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncols=2,
              handlelength=3.4, numpoints=2, markerscale=1.15)
    return _save(fig, "fig_error_vs_d.pdf")


def figure_runtime():
    """(supp) Matched-stage data and calibration wall times."""
    import matplotlib.pyplot as plt

    data_paths = [
        REPO_ROOT / "results" / "runtime" / "runtime_data_side.csv",
        REPO_ROOT / "results" / "runtime" / "runtime_data_side_d100000.csv",
    ]
    calibration_path = REPO_ROOT / "results" / "runtime" / \
        "runtime_calibration.csv"
    if not all(path.exists() for path in data_paths) or not calibration_path.exists():
        print("skip runtime figure: corrected runtime CSVs not found")
        return None
    data = pd.concat([pd.read_csv(path) for path in data_paths])
    calibration = pd.read_csv(calibration_path)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5))
    ax = axes[0]
    _style(ax)
    med_data = data.groupby(["D", "backend"], as_index=False)["seconds"].median()
    for backend, linestyle, marker in (
            ("cpu", "-", "o"), ("gpu", "--", "s")):
        series = med_data[med_data["backend"] == backend].sort_values("D")
        if series.empty:
            continue
        ax.plot(series["D"], series["seconds"], marker=marker, markersize=4.5,
                linewidth=2, linestyle=linestyle, color="#56B4E9",
                label=f"{backend.upper()} observed-data statistics")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ambient dimension $D$", fontsize=9, color=INK)
    ax.set_title(r"(a) observed-data geometry ($N=2500$)", fontsize=9,
                 color=INK, loc="left")
    ax.grid(axis="both", **GRID)

    ax = axes[1]
    _style(ax)
    med_cal = calibration.groupby(["maxDim", "stage", "backend"], as_index=False)[
        "seconds"
    ].median()
    stage_styles = [
        ("spline_evaluate", "precomputed spline", C_ISO),
        ("full_mc_evaluate", "new MC references", C_HET),
        ("cached_mc_evaluate", "reused MC references", "#009E73"),
    ]
    for stage, label, color in stage_styles:
        for backend, linestyle, marker in (
                ("cpu", "-", "o"), ("gpu", "--", "s")):
            series = med_cal[
                (med_cal["stage"] == stage) & (med_cal["backend"] == backend)
            ].sort_values("maxDim")
            if series.empty:
                continue
            ax.plot(series["maxDim"], series["seconds"], marker=marker,
                    markersize=4.5, linewidth=2, linestyle=linestyle,
                    color=color, label=f"{label} ({backend.upper()})")
    ax.set_yscale("log")
    ax.set_xticks([50, 100, 200, 400])
    ax.set_xlabel(r"candidate bound $m_{\max}$", fontsize=9, color=INK)
    ax.set_title(r"(b) complete evaluation ($N=500$)", fontsize=9,
                 color=INK, loc="left")
    ax.grid(axis="both", **GRID)
    axes[0].set_ylabel("wall time (s)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left",
                   handlelength=3.0, numpoints=2, markerscale=1.15)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncols=2,
                   bbox_to_anchor=(0.5, -0.30), handlelength=3.0,
                   numpoints=2, markerscale=1.15, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    return _save(fig, "fig_runtime.pdf")


# Centered-norm CV of the 17 image classes in the class survey (Supplementary
# Table S9) mapped to the lognormal GSM log-amplitude spread.  For X_i = S_i Z_i
# the amplitude and Gaussian factors are independent, so
# 1 + CV(a)^2 = (1 + CV_S^2)(1 + CV_Z^2) with CV_S^2 = exp(sigma_S^2) - 1 and
# CV_Z = 0.085 the coefficient of variation of ||Z_i|| at d = 70 (the chi_70
# value 0.0847 rounded as in Supplementary S3.5).
GSM_CV_Z = 0.085


def gsm_sigma_from_cv(cv):
    """Log-amplitude spread sigma_S whose GSM reproduces centered-norm CV ``cv``."""
    return float(np.sqrt(np.log((1.0 + cv ** 2) / (1.0 + GSM_CV_Z ** 2))))


def image_cv_band():
    """(min, max, median) centered-norm CV over the recorded image class survey."""
    path = REPO_ROOT / "results" / "std_datasets" / "class_survey.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    column = [c for c in frame.columns if "cv" in c.lower()][0]
    return (float(frame[column].min()), float(frame[column].max()),
            float(frame[column].median()))


def figure_fusion_ablation_main(source=None):
    """GSM amplitude sweep for the main text: three MiND objectives against
    the log-amplitude spread, with the image-class range and the known-amplitude
    control."""
    import matplotlib.pyplot as plt

    path = Path(source) if source else DIAG / "fusion_ablation_gsm_summary.csv"
    if not path.exists():
        print(f"skip focused fusion ablation figure: {path} not found")
        return None
    frame = pd.read_csv(path)
    frame = frame[(frame["suite"] == "gsm")
                  & ((frame["noise_sigma"] - 0.0).abs() < 1e-9)]
    raw = frame[frame["intervention"] == "raw"]
    series = [
        ("mind_distance_only", "MiND distance-only", "#56B4E9", "--", "s"),
        ("danco", "MiND–Full", C_ISO, "-", "o"),
        ("profiled_danco", "MiND–Profiled", "#009E73", "-.", "^"),
    ]
    missing = [m for m, *_ in series if m not in set(raw["method"])]
    if missing:
        print(f"skip focused fusion ablation figure: missing {missing}")
        return None

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    _style(ax)
    band = image_cv_band()
    if band is not None:
        lo, hi, _ = band
        ax.axvspan(gsm_sigma_from_cv(lo), gsm_sigma_from_cv(hi),
                   color="#e8e8e3", zorder=0, linewidth=0,
                   label="image-class range (Table S9)")
    ax.axhline(70, color="#b9b9b4", linewidth=1.1, linestyle=(0, (4, 3)),
               label="true $d=70$", zorder=1)
    for method, label, color, linestyle, marker in series:
        sub = raw[raw["method"] == method].sort_values("scale_sigma")
        ax.fill_between(sub["scale_sigma"],
                        sub["estimate_mean"] - sub["estimate_sd"],
                        sub["estimate_mean"] + sub["estimate_sd"],
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(sub["scale_sigma"], sub["estimate_mean"], marker=marker,
                markersize=4.5, linewidth=2, linestyle=linestyle,
                color=color, label=label, zorder=3)
    control = frame[(frame["intervention"] == "latent_descaled")
                    & (frame["method"] == "danco")
                    & (frame["scale_sigma"] <= 0.36)]
    if not control.empty:
        control = control.sort_values("scale_sigma").iloc[-1]
        ax.errorbar(control["scale_sigma"], control["estimate_mean"],
                    yerr=control["estimate_sd"], fmt="D", markersize=5.5,
                    color="#E69F00", markeredgecolor=INK, markeredgewidth=0.6,
                    ecolor=INK, elinewidth=1.1, capsize=2.5, zorder=4,
                    label="MiND–Full, known amplitudes removed")
    ax.set_xlabel(r"log-amplitude spread $\sigma_S$", fontsize=9.5, color=INK)
    ax.set_ylabel("estimated ID", fontsize=9.5, color=INK)
    ax.tick_params(labelsize=9.5)
    ax.set_xlim(-0.01, 0.36)
    ax.set_ylim(0, 90)
    ax.set_title(r"Gaussian scale mixture, $d=70$ in $D=100$, 30 replicates",
                 fontsize=10, color=INK, loc="left")
    # Journal-size legend text (native 9pt renders at 8pt) no longer fits
    # inside the axes without crossing the MiND--Full curve, so it sits below.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, loc="lower center",
               ncols=2, bbox_to_anchor=(0.5, 0.0), handlelength=2.4,
               numpoints=2, columnspacing=1.5, labelspacing=0.3)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    return _save(fig, "fig_fusion_ablation_main.pdf")


def figure_graphical_abstract():
    """Graphical abstract: delegated to ``make_graphical_abstract.py``.

    The submission graphical abstract recomputes real component curves and
    reads the amplitude-sweep summary; run
    ``python experiments/make_graphical_abstract.py`` to regenerate it.
    """
    print("graphical abstract is generated by experiments/make_graphical_abstract.py")
    return None


FIGURE_FUNCS = {
    "mechanism": figure_nu_mechanism,
    "mechanismmain": figure_nu_mechanism_main,
    "scatter": figure_diagnostics_scatter,
    "kl": figure_kl_diagnosis,
    "klmain": figure_kl_diagnosis_main,
    "overflow": figure_overflow,
    "vm": figure_vm_adequacy,
    "noisecurves": figure_noise_curves,
    "cnnpaired": figure_cnn_paired,
    "errord": figure_error_vs_d,
    "runtime": figure_runtime,
    "fusionmain": figure_fusion_ablation_main,
    "graphical": figure_graphical_abstract,
}


def main(argv=None):
    global FIGURES, INTERNAL_FIGURES
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--figures", default="all",
                        help=f"comma-separated subset of {sorted(FIGURE_FUNCS)}")
    add_paper_dir_argument(parser)
    args = parser.parse_args(argv)
    FIGURES, INTERNAL_FIGURES = artifact_dirs(args.paper_dir, "figures")

    import matplotlib
    matplotlib.use("Agg")
    use_manuscript_font()

    wanted = (list(FIGURE_FUNCS) if args.figures == "all"
              else [f.strip() for f in args.figures.split(",") if f.strip()])
    unknown = [f for f in wanted if f not in FIGURE_FUNCS]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; "
                         f"choose from {sorted(FIGURE_FUNCS)}")
    made = [FIGURE_FUNCS[name]() for name in wanted]
    n = sum(m is not None for m in made)
    n_expected = sum(name != "graphical" for name in wanted)
    print(f"\n{n}/{n_expected} figure(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
