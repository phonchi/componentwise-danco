#!/usr/bin/env python
"""Layer-by-layer intrinsic dimension of pretrained CNNs (Main Figure 5, Table S10).

    python experiments/run_cnn_layers.py --archs alexnet --dry-run
    python experiments/run_cnn_layers.py --archs alexnet
    python experiments/run_cnn_layers.py            # all four, overnight

For each architecture and each of the seven
ImageNet single-object categories, 500 images are pushed through the network
once, activations are captured at every checkpoint (see :mod:`danco.hooks`), and
each checkpoint's ``(500, D)`` matrix gets an ID estimate.

The pretrained spline surfaces are what make this feasible: a Monte-Carlo
calibration to ``maxDim=400`` per layer would dominate the run by orders of
magnitude, and ``N=500`` sits inside the spline's fitted ``N`` range.
``--estimator`` picks the distance strategy -- ``gride`` (the paper's
protocol, ``DANCo_gride_fast`` on the Gride-trained surface) or ``mind``
(``DANCo_fast`` on the MiND surface; not used in the paper).
``fractal=False`` -- layer estimates are integer candidate dimensions.

Both angle columns come out of one pass, so every row carries the Profiled
estimate (Gride--Profiled, the value plotted in Main Figure 5) *and* the Full
one as a diagnostic.

Resume is per *category*: rows are appended to the CSV as they finish, but a
category is only recorded as complete in the JSON sidecar once all its layers
are done.  A rerun drops rows belonging to incomplete categories and redoes
them, so an interrupted run cannot leave a half-category silently in the table.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO_ROOT
from danco import DANCo
from danco.hooks import build_pretrained_model, extract_activations, layer_checkpoints
from danco.spline import default_spline_path
from paper_methods import (
    PAPER_MLE_ID,
    PAPER_TWONN_ID,
    paper_mle,
    paper_twonn,
    spline_method_ids,
    stable_unique_rows,
)

CATEGORY_TAGS = ["n02086240", "n01882714", "n02087394", "n02094433",
                 "n02100583", "n02100735", "n02279972"]
ALL_ARCHS = ["alexnet", "vgg16", "resnet18", "resnet34"]
IMAGENET_DIR = REPO_ROOT / "data" / "imagenet_training_single_objs"
ESTIMATORS = {
    # name -> (DANCo version, spline surface)
    "gride": ("DANCo_gride_fast", Path(default_spline_path("gride"))),
    "mind": ("DANCo_fast", Path(default_spline_path("mind"))),
}

N_SAMPLES = 500
BATCH_SIZE = 16
MAXDIM = 400
K = 10
DANCO_SEED = 123  # unused by the spline versions (no sampling); for the record


def subset_seed(arch, tag, rep=0):
    """The image-subset seed, keyed to the *canonical* positions.

    The seed is ``1000 + arch_index * 100 + category_index`` with the indices
    taken in the full ``ALL_ARCHS`` / ``CATEGORY_TAGS`` lists, so that
    ``--archs alexnet --categories n01882714`` draws the same 500 images as the
    full sweep does.

    ``rep > 0`` shifts the stream far outside the canonical block so that
    repeated-subsampling runs draw disjoint pseudo-random subsets while
    ``rep=0`` remains bit-identical to the historical protocol.
    """
    ai = ALL_ARCHS.index(arch) if arch in ALL_ARCHS else 0
    ci = CATEGORY_TAGS.index(tag) if tag in CATEGORY_TAGS else 0
    return 1000 + ai * 100 + ci + 100000 * rep


def category_loader(tag, n=N_SAMPLES, batch_size=BATCH_SIZE, seed=123,
                    workers=2):
    """DataLoader over one synset: resize 256, centre-crop 224, normalise.

    The subset is a fixed permutation of the folder, so a category with more
    than ``n`` images always contributes the same ``n``.
    """
    import torch
    from torchvision import datasets, transforms

    folder = IMAGENET_DIR / tag
    if not folder.exists():
        raise SystemExit(
            f"no such category folder: {folder}. The ImageNet subset is a "
            "manual download (figshare 8a039f58c7b84a215b6d)."
        )
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(folder), transform)
    if len(dataset) == 0:
        raise SystemExit(f"empty ImageFolder for {tag}: {folder}")
    n_loaded = min(int(n), len(dataset))
    order = np.random.default_rng(seed).permutation(len(dataset))
    subset = torch.utils.data.Subset(dataset, order[:n_loaded].tolist())
    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
                  pin_memory=torch.cuda.is_available())
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(subset, **kwargs), n_loaded


def estimate_layer(X, backend, maxDim=MAXDIM, n_jobs=1, backend_options=None,
                   diagnostics=False, version="DANCo_gride_fast",
                   spline_file=None):
    """ID of one activation matrix under the shared paper preprocessing.

    Exact duplicate rows are removed once, in stable first-occurrence order,
    before Gride--Profiled, reference TWO-NN, and MLE read the matrix.  The GPU
    backend's separate near-duplicate filter is disabled for paper runs.

    ``diagnostics=True`` adds the paired-profile columns the paper's CNN
    section needs: an MLE (Levina--Bickel) estimate on the same activations,
    and the per-layer norm and neighbor-distance statistics that link the
    layer-wise $\\hat\\nu$ to the scale-heterogeneity mechanism and to the
    norm-growth account of Schulte & R\\"ugamer. All additions are
    deterministic and leave the original columns untouched.
    """
    X, duplicate_audit = stable_unique_rows(X)
    n = X.shape[0]
    full_method_id, profiled_method_id = spline_method_ids(version)
    resolved_backend_options = dict(backend_options or {})
    if backend == "gpu":
        # Exact duplicates were removed once for all three estimators.  Do not
        # let the GPU backend silently apply a second, near-duplicate policy.
        resolved_backend_options.setdefault("dup_threshold", None)
    estimator = DANCo(N=n, k=K, version=version, random_state=DANCO_SEED,
                      fractal=False, verbose=False, backend=backend,
                      backend_options=resolved_backend_options)
    t0 = time.perf_counter()
    stats = estimator.evaluate_all(X, maxDim=maxDim, n_jobs=n_jobs,
                                   spline_file=str(spline_file))
    danco_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    twonn_ref = paper_twonn(X, assume_unique=True)
    twonn_ref_s = time.perf_counter() - t0
    out = dict(
        **duplicate_audit,
        id_nustrip=int(round(stats["de_nustrip"])),
        id_orig_diagnostic=int(round(stats["de_standard"])),
        dhat=float(stats["dhat"]),
        twonn_ref=float(twonn_ref),
        twonn_ref_s=twonn_ref_s,
        mu_nu=float(stats["mu_nu"]),
        mu_tau=float(stats["mu_tau"]),
        danco_s=danco_s,
        method_id_full=full_method_id,
        method_id_profiled=profiled_method_id,
        method_id_twonn=PAPER_TWONN_ID,
        method_id_mle=PAPER_MLE_ID,
        backend_dtype=(resolved_backend_options.get("dtype", "float32")
                       if backend == "gpu" else "float64"),
        backend_dup_threshold=(
            "none" if resolved_backend_options.get("dup_threshold") is None
            else resolved_backend_options["dup_threshold"]
        ),
    )
    if diagnostics:
        from danco.backends import get_backend

        X64 = np.asarray(X, dtype=np.float64)
        dists, _ = get_backend("cpu").knn(X64, K, n_jobs=max(n_jobs, 1))
        centred = X64 - X64.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centred, axis=1)
        t0 = time.perf_counter()
        mle = paper_mle(X64, k=K, assume_unique=True)
        out.update(
            id_mle=mle,
            mle_s=time.perf_counter() - t0,
            r1_mean=float(dists[:, 0].mean()),
            rk1_mean=float(dists[:, -1].mean()),
            norm_mean=float(norms.mean()),
            norm_cv=float(norms.std() / norms.mean()),
        )
    return out


# ---------------------------------------------------------------------------
# resume bookkeeping
# ---------------------------------------------------------------------------
def load_resume(csv_path, json_path):
    """Rows worth keeping, and the (arch, tag) pairs already finished."""
    if not csv_path.exists() or not json_path.exists():
        return [], set()
    state = json.loads(json_path.read_text())
    completed = {tuple(pair) for pair in state.get("completed", [])}
    df = pd.read_csv(csv_path)
    keep = [row for row in df.to_dict("records")
            if (row["arch"], row["category_tag"]) in completed]
    dropped = len(df) - len(keep)
    if dropped:
        print(f"resume: dropping {dropped} row(s) from unfinished categories")
    return keep, completed


def write_csv(rows, path):
    """Atomic rewrite -- an interrupted run must not truncate the table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def dry_run(archs, categories, args):
    """List what would be processed, with real shapes, without estimating."""
    import torch

    version, spline_file = ESTIMATORS[args.estimator]
    full_method_id, profiled_method_id = spline_method_ids(version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  n={args.nsamples}  maxDim={args.maxdim}  k={K}  "
          f"estimator={args.estimator} ({version})")
    print(f"spline={spline_file} exists={spline_file.exists()}")
    for ai, arch in enumerate(archs):
        model, weights = build_pretrained_model(arch, pretrained=args.pretrained)
        model = model.to(device).eval()
        _, names, depths = layer_checkpoints(model, arch)
        print(f"\n{arch}  weights={weights}  checkpoints={len(names)}")
        tag = categories[0]
        loader, n_loaded = category_loader(
            tag, n=min(args.batch_size, args.nsamples),
            batch_size=args.batch_size, seed=subset_seed(arch, tag),
            workers=0)
        acts, names, depths, profile = extract_activations(model, arch, loader,
                                                           device)
        print(f"  probe category {tag}: {n_loaded} images "
              f"({profile['feature_extract_wall_s']:.1f}s)")
        print(f"  {'idx':>3} {'layer':<12}{'depth':>6}{'D':>10}"
              f"{'maxDim':>8}")
        for li, (name, depth, act) in enumerate(zip(names, depths, acts)):
            print(f"  {li:>3} {name:<12}{int(depth):>6}{act.shape[1]:>10}"
                  f"{min(args.maxdim, act.shape[1]):>8}")
        print(f"  would estimate {len(names)} layer(s) x "
              f"{len(categories)} categories = "
              f"{len(names) * len(categories)} rows")
        del model, acts
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results/cnn/cnn_layer_ids_rep0.csv")
    parser.add_argument("--archs", default=",".join(ALL_ARCHS),
                        help="comma-separated torchvision architectures")
    parser.add_argument("--categories", default=",".join(CATEGORY_TAGS),
                        help="comma-separated ImageNet synset ids")
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--nsamples", type=int, default=N_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--maxdim", type=int, default=MAXDIM)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="CPU threads, used only with --backend cpu")
    parser.add_argument("--no-pretrained", dest="pretrained",
                        action="store_false",
                        help="random weights; for pipeline smoke tests only")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS),
                        default="gride",
                        help="distance strategy: 'gride' (DANCo_gride_fast, "
                             "the paper protocol) or 'mind' (DANCo_fast, "
                             "not used in the paper)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="add MLE and per-layer norm/NN-distance columns "
                             "(paired-profile runs)")
    parser.add_argument("--subset-rep", type=int, default=0,
                        help="repeated-subsampling replicate; 0 keeps the "
                             "historical image subsets")
    parser.add_argument("--dry-run", action="store_true",
                        help="list layers and activation shapes, estimate "
                             "nothing")
    args = parser.parse_args(argv)

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if args.dry_run:
        return dry_run(archs, categories, args)

    import torch
    import sklearn
    import skdim
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = Path(args.output)
    json_path = csv_path.with_suffix(".summary.json")
    rows, completed = load_resume(csv_path, json_path) if args.resume else ([],
                                                                           set())
    version, spline_file = ESTIMATORS[args.estimator]
    if not spline_file.exists():
        raise SystemExit(f"missing spline file {spline_file}")

    backend_options = {}
    if args.backend == "gpu":
        backend_options["dup_threshold"] = None
    print(f"archs={archs}\ncategories={categories}\n"
          f"device={device} backend={args.backend} n={args.nsamples} "
          f"maxDim={args.maxdim} estimator={args.estimator} ({version})\n"
          f"output={csv_path}", flush=True)
    wall0 = time.perf_counter()
    failures = []

    for ai, arch in enumerate(archs):
        model = None
        try:
            model, weights = build_pretrained_model(arch,
                                                    pretrained=args.pretrained)
            model = model.to(device).eval()
            _, names, depths = layer_checkpoints(model, arch)
            print(f"\n[{ai + 1}/{len(archs)}] {arch}: weights={weights}, "
                  f"checkpoints={len(names)}", flush=True)
            for ci, tag in enumerate(categories):
                if (arch, tag) in completed:
                    print(f"  skip completed {arch} {tag}")
                    continue
                cat_t0 = time.perf_counter()
                try:
                    loader, n_loaded = category_loader(
                        tag, n=args.nsamples, batch_size=args.batch_size,
                        seed=subset_seed(arch, tag, args.subset_rep))
                    acts, names, depths, profile = extract_activations(
                        model, arch, loader, device)
                    print(f"  {tag}: {n_loaded} images, shapes="
                          f"{profile['per_layer_shapes']}, "
                          f"{profile['feature_extract_wall_s']:.1f}s",
                          flush=True)
                    for li, X in enumerate(acts):
                        t0 = time.perf_counter()
                        est = estimate_layer(X, args.backend,
                                             maxDim=args.maxdim,
                                             n_jobs=args.n_jobs,
                                             backend_options=backend_options,
                                             diagnostics=args.diagnostics,
                                             version=version,
                                             spline_file=spline_file)
                        rows.append(dict(
                            arch=arch, weights=weights, category_idx=ci,
                            subset_rep=int(args.subset_rep),
                            estimator=args.estimator,
                            category_tag=tag, layer_idx=li,
                            layer_name=names[li], depth=int(depths[li]),
                            embedding_dim=int(X.shape[1]), n=int(X.shape[0]),
                            delta_orig_minus_nustrip=(
                                est["id_orig_diagnostic"] - est["id_nustrip"]),
                            layer_wall_s=time.perf_counter() - t0,
                            feature_extract_wall_s=profile[
                                "feature_extract_wall_s"],
                            **est))
                        write_csv(rows, csv_path)
                        print(f"    {li + 1}/{len(acts)} {names[li]:<10} "
                              f"D={X.shape[1]:<7} "
                              f"id_nustrip={est['id_nustrip']:<4} "
                              f"orig={est['id_orig_diagnostic']:<4} "
                              f"2NN={est['twonn_ref']:.2f}  "
                              f"{rows[-1]['layer_wall_s']:.1f}s", flush=True)
                    completed.add((arch, tag))
                    print(f"  completed {arch} {tag} in "
                          f"{time.perf_counter() - cat_t0:.1f}s", flush=True)
                except Exception as exc:  # keep going: one bad category
                    failures.append(dict(arch=arch, category_tag=tag,
                                         error=f"{type(exc).__name__}: {exc}"))
                    print(f"  ERROR {arch} {tag}: {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
                finally:
                    acts = None
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    json_path.write_text(json.dumps(dict(
                        archs=archs, categories=categories,
                        subset_rep=args.subset_rep, nsamples=args.nsamples,
                        maxdim=args.maxdim, k=K,
                        estimator=args.estimator, version=version,
                        backend=args.backend, device=str(device),
                        backend_options=backend_options,
                        duplicate_policy="stable exact-row deduplication shared by all estimators",
                        method_ids={
                            "full": full_method_id,
                            "profiled": profiled_method_id,
                            "twonn": PAPER_TWONN_ID,
                            "mle": PAPER_MLE_ID,
                        },
                        software={
                            "numpy": np.__version__,
                            "scikit-learn": sklearn.__version__,
                            "scikit-dimension": skdim.__version__,
                            "torch": torch.__version__,
                            "torchvision": torchvision.__version__,
                            "cuda_runtime": torch.version.cuda,
                        },
                        pretrained=args.pretrained,
                        completed=sorted(list(p) for p in completed),
                        failures=failures,
                        wall_s=time.perf_counter() - wall0,
                    ), indent=2))
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nwrote {csv_path} ({len(rows)} rows) and {json_path}")
    if failures:
        print(f"{len(failures)} category failure(s); see the JSON sidecar")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
