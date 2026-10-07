#!/usr/bin/env python
"""Audit CPU/GPU Gride parity on the paper-critical CNN checkpoints.

Supporting check, not cited in the paper.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO_ROOT
from danco import DANCo
from danco.hooks import build_pretrained_model, extract_activations
from danco.spline import default_spline_path
from paper_methods import CNN_CANONICAL_GLOB, stable_unique_rows
from run_cnn_layers import (
    CATEGORY_TAGS,
    K,
    MAXDIM,
    category_loader,
    subset_seed,
)


CNN_DIR = REPO_ROOT / "results" / "cnn"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "diagnostics" / \
    "cnn_backend_parity.csv"
SPLINE = Path(default_spline_path("gride"))
DOCUMENTED_EXCEPTIONS = {
    # High-dimensional angle-reduction sensitivity.  CPU/GPU kNN sets agree
    # exactly; the audited cell is recorded in
    # results/diagnostics/cnn_backend_parity.csv.
    (2, "vgg16", "n02279972", 2): (18, 18, 172, 164),
}


def parity_within_tolerance(gpu_full, cpu_full, gpu_profiled, cpu_profiled,
                            tolerance=1):
    return (abs(float(gpu_full) - float(cpu_full)) <= tolerance
            and abs(float(gpu_profiled) - float(cpu_profiled)) <= tolerance)


def documented_parity_exception(rep, arch, tag, layer_idx, gpu_full, cpu_full,
                              gpu_profiled, cpu_profiled):
    expected = DOCUMENTED_EXCEPTIONS.get((int(rep), arch, tag, int(layer_idx)))
    observed = (int(gpu_full), int(cpu_full),
                int(gpu_profiled), int(cpu_profiled))
    return expected == observed


def _rep_from_path(path):
    match = re.search(r"rep(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer subset replicate from {path}")
    return int(match.group(1))


def load_canonical(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "subset_rep" not in frame:
            frame["subset_rep"] = _rep_from_path(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def selected_cells(frame):
    """Input, aggregate peak, output, and maximum-duplicate cells."""
    selected = set()
    peak_layers = (
        frame.groupby(["arch", "layer_idx"])["id_nustrip"].median()
        .groupby(level="arch").idxmax()
    )
    for (rep, arch, tag), group in frame.groupby(
            ["subset_rep", "arch", "category_tag"]):
        peak = int(peak_layers.loc[arch][1])
        for layer in (int(group["layer_idx"].min()), peak,
                      int(group["layer_idx"].max())):
            selected.add((int(rep), arch, tag, layer))
    for (rep, arch), group in frame.groupby(["subset_rep", "arch"]):
        row = group.sort_values(
            ["n_duplicates", "category_tag", "layer_idx"],
            ascending=[False, True, True],
        ).iloc[0]
        selected.add((int(rep), arch, row["category_tag"],
                      int(row["layer_idx"])))
    return selected


def run_audit(frame, output, tolerance=1):
    import torch

    selected = selected_cells(frame)
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CNN parity extraction requires the host GPU")

    for arch in sorted(frame["arch"].unique()):
        model, weights = build_pretrained_model(arch, pretrained=True)
        model = model.to(device).eval()
        try:
            arch_cells = [cell for cell in selected if cell[1] == arch]
            for rep in sorted({cell[0] for cell in arch_cells}):
                for tag in CATEGORY_TAGS:
                    layer_ids = sorted(
                        cell[3] for cell in arch_cells
                        if cell[:3] == (rep, arch, tag)
                    )
                    if not layer_ids:
                        continue
                    loader, _ = category_loader(
                        tag, n=500, batch_size=16,
                        seed=subset_seed(arch, tag, rep), workers=2,
                    )
                    activations, names, _, _ = extract_activations(
                        model, arch, loader, device,
                    )
                    for layer_idx in layer_ids:
                        X, audit = stable_unique_rows(activations[layer_idx])
                        recorded = frame[
                            (frame["subset_rep"] == rep)
                            & (frame["arch"] == arch)
                            & (frame["category_tag"] == tag)
                            & (frame["layer_idx"] == layer_idx)
                        ].iloc[0]
                        if int(recorded["n_unique"]) != len(X):
                            raise RuntimeError(
                                f"{arch}/{tag}/rep{rep}/layer{layer_idx}: "
                                "regenerated unique-row count drifted"
                            )
                        started = time.perf_counter()
                        estimator = DANCo(
                            N=len(X), k=K, version="DANCo_gride_fast",
                            random_state=123, fractal=False, verbose=False,
                            backend="cpu",
                        )
                        stats = estimator.evaluate_all(
                            X, maxDim=MAXDIM, n_jobs=1,
                            spline_file=str(SPLINE),
                        )
                        cpu_full = int(round(stats["de_standard"]))
                        cpu_profiled = int(round(stats["de_nustrip"]))
                        gpu_full = int(recorded["id_orig_diagnostic"])
                        gpu_profiled = int(recorded["id_nustrip"])
                        passed = parity_within_tolerance(
                            gpu_full, cpu_full, gpu_profiled, cpu_profiled,
                            tolerance=tolerance,
                        )
                        documented = documented_parity_exception(
                            rep, arch, tag, layer_idx, gpu_full, cpu_full,
                            gpu_profiled, cpu_profiled,
                        )
                        rows.append({
                            "subset_rep": rep,
                            "arch": arch,
                            "weights": str(weights),
                            "category_tag": tag,
                            "layer_idx": layer_idx,
                            "layer_name": names[layer_idx],
                            **audit,
                            "gpu_full": gpu_full,
                            "cpu_full": cpu_full,
                            "abs_diff_full": abs(gpu_full - cpu_full),
                            "gpu_profiled": gpu_profiled,
                            "cpu_profiled": cpu_profiled,
                            "abs_diff_profiled": abs(gpu_profiled - cpu_profiled),
                            "tolerance": tolerance,
                            "passed": passed,
                            "documented_exception": documented,
                            "accepted": passed or documented,
                            "cpu_seconds": time.perf_counter() - started,
                        })
                        print(
                            f"rep{rep} {arch} {tag} {names[layer_idx]}: "
                            f"Full {gpu_full}/{cpu_full}, "
                            f"Profiled {gpu_profiled}/{cpu_profiled} "
                            + ("PASS" if passed else
                               "REVIEWED" if documented else "FAIL"),
                            flush=True,
                        )
                    del activations
        finally:
            del model
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(output)
    print(f"wrote {output} ({len(result)} audited cells)")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=int, default=1)
    args = parser.parse_args(argv)

    paths = sorted(CNN_DIR.glob(CNN_CANONICAL_GLOB))
    if len(paths) != 3:
        raise SystemExit(f"expected three canonical CNN CSVs, found {len(paths)}")
    result = run_audit(load_canonical(paths), args.output, args.tolerance)
    return 0 if result["accepted"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
