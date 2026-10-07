"""Draw the graphical abstract from the archived paper evidence.

    python experiments/make_graphical_abstract.py                     # -> results/generated/
    python experiments/make_graphical_abstract.py --paper-dir paper    # PDF/PNG -> paper/figures/

Geometry and distribution sketches explain the method; they are not observed
experimental data. Result comparisons read archived relative-noise and GSM
summaries. Rendering never rewrites those results. ``--curves-only --rep R``
retains the earlier single-sample KL diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from common import (GENERATED_DIR, MANUSCRIPT_FONT, REPO_ROOT,
                    add_paper_dir_argument, artifact_dirs, use_manuscript_font)

DIAG = REPO_ROOT / "results" / "diagnostics"
FIGURES = artifact_dirs(None, "figures")[0]
DESIGN = GENERATED_DIR / "graphical_abstract"   # editable SVG, preview, checks
CURVES = DIAG / "graphical_abstract_curves.csv"
NOISE = DIAG / "relative_noise_ablation_summary.csv"
GSM = DIAG / "fusion_ablation_gsm_summary.csv"

WIDTH, HEIGHT = 1328, 531
INK, MUTED, RULE = "#172D39", "#5D6A72", "#D8E1E5"
BLUE, GREEN = "#0072B2", "#008568"
BLUE_LIGHT, GREEN_LIGHT = "#F0F7FB", "#EFF8F4"
FONT = MANUSCRIPT_FONT


def compute_curves(rep: int = 0) -> pd.DataFrame:
    from danco import DANCo

    n, d, ambient, k, max_dim = 2500, 70, 100, 10, 100
    sigma_s = 0.25
    generator = np.random.default_rng(rep)
    latent = generator.standard_normal((n, d))
    scales = np.exp(sigma_s * generator.standard_normal(n))
    observed = np.zeros((n, ambient))
    observed[:, :d] = scales[:, None] * latent
    estimator = DANCo(N=n, k=k, version="DANCo", random_state=700 + rep,
                      calibration_seeding="stream")
    stats = estimator.evaluate_all(observed, maxDim=max_dim)
    kl = np.asarray(stats["kl_divergence"], dtype=float)
    strip = np.asarray(stats["kl_ang_strip"], dtype=float)
    frame = pd.DataFrame({
        "m": np.arange(1, max_dim + 1),
        "kl_distance": kl[:, 0],
        "kl_angle_full": kl[:, 1],
        "kl_angle_profiled": strip,
    })
    frame["sum_full"] = frame["kl_distance"] + frame["kl_angle_full"]
    frame["sum_profiled"] = frame["kl_distance"] + frame["kl_angle_profiled"]
    frame.attrs["de_full"] = float(stats["de_standard"])
    frame.attrs["de_profiled"] = float(stats["de_nustrip"])
    frame.attrs["rep"] = rep
    return frame



def load_evidence(noise_path=NOISE, gsm_path=GSM):
    """Select exactly the comparisons named in the manuscript abstract."""
    noise = pd.read_csv(noise_path)
    gsm = pd.read_csv(gsm_path)

    def one(frame, mask, metric, reps, datasets):
        selected = frame.loc[mask]
        if len(selected) != 1:
            raise ValueError(f"expected one {metric} row, found {len(selected)}")
        row = selected.iloc[0]
        if int(row["n_reps"]) != reps or int(row["n_datasets"]) != datasets:
            raise ValueError("replicate or dataset count does not match the GA label")
        if not np.isfinite(row[metric]):
            raise ValueError(f"non-finite {metric}")
        return float(row[metric])

    nmask = np.isclose(noise["noise_eta"], 0.4)
    gmask = ((gsm["suite"] == "gsm") & (gsm["intervention"] == "raw")
             & np.isclose(gsm["noise_sigma"], 0)
             & np.isclose(gsm["scale_sigma"], 0.25))
    return {
        "noise_mind": one(noise, nmask & (noise["method"] == "danco"),
                          "mpe_mean", 10, 24),
        "noise_gride": one(noise, nmask & (noise["method"] == "danco_gride"),
                           "mpe_mean", 10, 24),
        "gsm_full": one(gsm, gmask & (gsm["method"] == "danco"),
                        "estimate_mean", 30, 1),
        "gsm_profiled": one(gsm, gmask & (gsm["method"] == "profiled_danco"),
                            "estimate_mean", 30, 1),
    }


def draw(evidence, output_dir=FIGURES, design_dir=DESIGN):
    """Compose an original vector schematic with two data-grounded readouts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
    from matplotlib.path import Path as MplPath

    serif = use_manuscript_font()
    pt = 72 * (13.28 / 2.54) / WIDTH
    plt.rcParams.update({"svg.hashsalt": "danco-ga-v2"})
    fig = plt.figure(figsize=(13.28 / 2.54, 5.31 / 2.54), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    def label(x, y, text, size=27, color=INK, weight="normal", ha="left", family=FONT):
        return ax.text(x, y, text, fontsize=size * pt, color=color,
                       fontweight=weight, ha=ha, va="center", fontfamily=family)

    def line(points, color=RULE, width=1.0, **kwargs):
        x, y = zip(*points)
        ax.plot(x, y, color=color, linewidth=width, **kwargs)

    def arrow(start, end, color=MUTED, width=0.9):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=7, linewidth=width,
                                    color=color, shrinkA=0, shrinkB=0))

    def elbow(points, color):
        """Orthogonal connector with 6-unit rounded corners."""
        vertices, codes = [points[0]], [MplPath.MOVETO]
        for i in range(1, len(points) - 1):
            previous, corner, following = map(np.array, (points[i-1], points[i], points[i+1]))
            before = corner + 6 * (previous-corner) / np.linalg.norm(previous-corner)
            after = corner + 6 * (following-corner) / np.linalg.norm(following-corner)
            vertices.extend([before, corner, after])
            codes.extend([MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3])
        vertices.append(points[-1]); codes.append(MplPath.LINETO)
        ax.add_patch(FancyArrowPatch(path=MplPath(vertices, codes), arrowstyle="-|>",
                                    mutation_scale=7, linewidth=0.9, color=color))

    # Clear headline and reading order. White space carries the grouping.
    label(40, 43, "Interpretable intrinsic dimension", 44, weight="bold", family=serif)
    label(1288, 45, "Distance + angle calibration", 27, MUTED, ha="right")
    line([(40, 80), (1288, 80)], width=0.65)
    label(142, 116, "Neighborhood", 29, weight="bold", ha="center")
    label(312, 116, "Separate & calibrate", 30, weight="bold")
    label(975, 116, "Measured gains", 30, weight="bold")

    # Shared neighborhood: designed illustration, never a claimed projection.
    cx, cy = 141, 274
    rng = np.random.default_rng(73)
    cloud = rng.normal(size=(30, 2)) * [64, 64] + [cx, cy]
    cloud = cloud[(cloud[:, 0] > 48) & (cloud[:, 0] < 235)
                  & (cloud[:, 1] > 170) & (cloud[:, 1] < 380)]
    ax.scatter(cloud[:, 0], cloud[:, 1], s=7, c="#BECBD1", linewidths=0, zorder=3)
    for r in (42, 86):
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="#A8BFCC",
                            linewidth=0.65, linestyle=(0, (2.5, 2.5))))
    theta = np.linspace(-0.72, 0.82, 40)
    arc = np.column_stack((cx+31*np.cos(theta), cy+31*np.sin(theta)))
    ax.add_patch(Polygon(np.vstack(([cx, cy], arc)), closed=True,
                         facecolor="#C4E6DA", edgecolor=GREEN, linewidth=0.6))
    for point, color in [((173, 247), BLUE), ((198, 337), GREEN), ((202, 218), BLUE)]:
        line([(cx, cy), point], color, 0.8)
        ax.scatter(*point, s=18, c=color, zorder=5, linewidths=0.5, edgecolors="white")
    ax.scatter(cx, cy, s=29, c=INK, edgecolors="white", linewidths=0.7, zorder=6)
    label(111, 295, r"$x_i$", 30, ha="center")
    label(176, 277, r"$\theta$", 31, GREEN)
    label(84, 189, r"$r_{k_2}$", 29, BLUE)
    label(82, 239, r"$r_{k_1}$", 29, BLUE)
    label(142, 404, "Shared geometry", 27, ha="center")
    label(142, 436, "Schematic", 27, MUTED, ha="center")

    elbow([(238, 230), (275, 230), (275, 222), (304, 222)], BLUE)
    elbow([(238, 334), (275, 334), (275, 382), (304, 382)], GREEN)

    # Distance branch: a ratio across larger neighbor orders.
    ax.add_patch(Rectangle((312, 157), 465, 136, facecolor=BLUE_LIGHT, edgecolor="none"))
    ax.add_patch(Rectangle((312, 157), 4, 136, facecolor=BLUE, edgecolor="none"))
    label(333, 184, "Distance", 30, weight="bold")
    label(753, 184, "MiND → Gride", 28, BLUE, "bold", ha="right")
    for radius, color, dash in [(18, MUTED, (0, (2, 2))), (36, BLUE, "solid")]:
        ax.add_patch(Circle((376, 243), radius, fill=False, edgecolor=color,
                            linewidth=0.8, linestyle=dash))
    ax.scatter(376, 243, s=15, c=INK, zorder=5)
    line([(376, 243), (399, 215)], BLUE, 1.1)
    line([(376, 243), (363, 256)], MUTED, 0.8)
    label(438, 226, "Higher-order ratios", 29, BLUE)
    label(438, 265, "Closed-form KL", 27, MUTED)

    # Angle branch: translate only the reference location; preserve widths.
    ax.add_patch(Rectangle((312, 313), 465, 141, facecolor=GREEN_LIGHT, edgecolor="none"))
    ax.add_patch(Rectangle((312, 313), 4, 141, facecolor=GREEN, edgecolor="none"))
    label(333, 342, "Angle", 30, weight="bold")
    label(753, 342, "Full → Profiled", 28, GREEN, "bold", ha="right")
    for left, peak1, peak2, right in [(332, 350, 374, 397), (427, 452, 452, 482)]:
        xx = np.linspace(left, right, 120)
        line([(left, 427), (right, 427)], "#C5D9D2", 0.6)
        ax.plot(xx, 427-48*np.exp(-0.5*((xx-peak1)/8.5)**2), color=MUTED,
                linestyle=(0, (2, 1.5)), linewidth=0.8)
        ax.plot(xx, 427-38*np.exp(-0.5*((xx-peak2)/12.5)**2), color=GREEN, linewidth=1.15)
    arrow((400, 403), (421, 403), GREEN, 0.8)
    label(505, 384, "Align direction", 27, GREEN)
    label(505, 422, "Keep concentration", 27, MUTED)

    # Sum discrepancies on a common candidate grid, then minimize once.
    elbow([(777, 224), (815, 224), (815, 290), (846, 290)], BLUE)
    elbow([(777, 382), (815, 382), (815, 311), (846, 311)], GREEN)
    label(799, 202, r"$\Delta_{\rm dist}$", 29, BLUE, ha="center")
    label(822, 417, r"$\Delta_{\rm ang}$", 29, GREEN, ha="center")
    ax.add_patch(Circle((866, 300), 23, facecolor=INK, edgecolor="none"))
    label(866, 300, r"$\Sigma$", 39, "white", ha="center")
    label(866, 346, r"$\arg\min_m$", 32, ha="center")
    arrow((890, 300), (942, 300), INK, 1.0)
    label(919, 270, r"$\hat d$", 38, ha="center")
    line([(956, 151), (956, 465)], width=0.65)

    # Evidence 1: bars share a zero baseline and a common percentage scale.
    label(979, 177, "Noise error (MPE)", 28, weight="bold")
    for y, name, value, color in [
        (218, "MiND–Full", evidence["noise_mind"], MUTED),
        (267, "Gride–Full", evidence["noise_gride"], BLUE),
    ]:
        label(979, y, name, 27, color)
        label(1288, y, f"{value:.1f}%", 34, color, "bold", ha="right")
        ax.add_patch(Rectangle((979, y+15), 309*value/30, 7, facecolor=color, edgecolor="none"))
    label(979, 309, r"$\eta=0.4$; 24 manifolds", 27, MUTED)

    # Evidence 2: a controlled mean-ID comparison, not a universal recovery claim.
    label(979, 348, "MiND on GSM", 28, weight="bold")
    label(979, 378, "Mean ID; true d = 70", 27, MUTED)
    label(979, 410, f"{evidence['gsm_full']:.1f}", 34, MUTED, "bold")
    arrow((1100, 410), (1148, 410), GREEN, 1.1)
    label(1288, 410, f"{evidence['gsm_profiled']:.1f}", 34, GREEN, "bold", ha="right")
    label(980, 444, "Full", 27, MUTED)
    label(1288, 444, "Profiled", 27, GREEN, ha="right")
    label(979, 472, r"$\sigma_S=0.25$; 30 replicates", 27, MUTED)

    # Application strip: secondary to the two mechanisms.
    line([(40, 488), (1288, 488)], width=0.65)
    for i, height in enumerate((16, 24, 32, 24, 16)):
        x = 44+i*28
        ax.add_patch(Rectangle((x, 512-height/2), 12, height,
                               facecolor="#E5EDF0", edgecolor="#839BA8", linewidth=0.55))
        if i < 4:
            line([(x+13, 512), (x+27, 512)], "#839BA8", 0.55)
    label(210, 512, "CNN layer diagnostics", 28, weight="bold")
    label(1288, 512, r"Stable evaluation through $m=400$", 27, MUTED, ha="right")

    output_dir, design_dir = Path(output_dir), Path(design_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [(text, text.get_window_extent(renderer)) for text in ax.texts]
    collisions = [(a.get_text(), b.get_text())
                  for i, (a, aa) in enumerate(bounds) for b, bb in bounds[i+1:]
                  if aa.overlaps(bb)]
    outside = [text.get_text() for text, box in bounds
               if box.x0 < 0 or box.y0 < 0
               or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
    (design_dir / "layout-check.json").write_text(json.dumps({
        "overlapping_text": collisions, "text_outside_canvas": outside,
        "minimum_font_pt": min(t.get_fontsize() for t in ax.texts),
    }, indent=2)+'\n')
    fig.savefig(output_dir / "graphical_abstract.pdf", metadata={"CreationDate": None})
    fig.savefig(output_dir / "graphical_abstract.png", dpi=600)
    svg_path = design_dir / "graphical_abstract.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    svg = svg_path.read_text()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r"<svg\b", '<svg role="img" aria-labelledby="ga-title ga-desc"', svg, count=1)
    end = svg.index(">")+1
    svg = (svg[:end] + '<title id="ga-title">Interpretable intrinsic dimension</title>'
           '<desc id="ga-desc">Shared neighbor geometry is calibrated through distance '
           'and angle components, with measured noise and amplitude-mismatch comparisons.</desc>'
           + svg[end:])
    svg_path.write_text('<?xml version="1.0" encoding="utf-8"?>\n'+svg)
    (design_dir / "preview.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>DANCo graphical abstract</title><style>'
        'body{margin:0;background:#eef1f3;padding:40px;font-family:sans-serif}'
        'main{max-width:1400px;margin:auto;background:white;padding:24px}'
        'svg{width:100%;height:auto;display:block}</style><main>'+svg+'</main></html>')
    (design_dir / "evidence.json").write_text(json.dumps({
        "values": evidence, "noise_source": str(NOISE.relative_to(REPO_ROOT)),
        "gsm_source": str(GSM.relative_to(REPO_ROOT)),
        "noise_condition": {"eta": 0.4, "replicates": 10, "manifolds": 24},
        "gsm_condition": {"sigma_s": 0.25, "replicates": 30, "d": 70, "D": 100},
        "illustration": "fixed schematic geometry; not experimental data",
        "canvas_cm": [13.28, 5.31], "png_dpi": 600,
    }, indent=2)+'\n')
    plt.close(fig)
    print(f"wrote {output_dir / 'graphical_abstract.pdf'} and .png")
    print(f"editable SVG and preview: {design_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_paper_dir_argument(parser)
    parser.add_argument("--design-dir", type=Path, default=DESIGN,
                        help="where the editable SVG, HTML preview and layout "
                             "checks are written")
    parser.add_argument("--rep", type=int, default=0,
                        help="replicate for the optional --curves-only diagnostic")
    parser.add_argument("--curves-only", action="store_true",
                        help="recompute the earlier single-sample KL CSV; do not draw")
    args = parser.parse_args(argv)
    if args.curves_only:
        curves = compute_curves(args.rep)
        CURVES.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(CURVES, index=False)
        print(f"wrote {CURVES}")
    else:
        draw(load_evidence(), artifact_dirs(args.paper_dir, "figures")[0],
             args.design_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
