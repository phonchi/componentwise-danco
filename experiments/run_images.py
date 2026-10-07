#!/usr/bin/env python
"""Main Table 2 -- intrinsic dimension of raw image classes.

    python experiments/run_images.py --dataset mnist
    python experiments/run_images.py --dataset imagenet --subset koala

Three datasets (MNIST test digits 3 and 7, CIFAR-10 training classes 2 and 3,
two ImageNet single-object classes).  All three use ``k=10``,
``maxDim=MAXDIM`` (200), ``fractal=True`` and ``random_state = 43 + seed``,
and all three feed the estimator *raw* pixels as float64 -- no normalisation,
no PCA.

The one protocol difference between them is estimator lifetime, and it is not
cosmetic:

* MNIST and ImageNet build a **fresh estimator per class**, so each class is
  calibrated from an empty cache;
* CIFAR builds the estimators **once per seed** and runs both classes through
  them, so class 3 inherits the cache class 2 grew.

The per-class statistics (nu, tau, dhat) written to ``image_statistics_*.csv``
are properties of the data, not of the seed -- they are computed before
calibration is ever consulted -- so they are read off the first seed.

Outputs: ``results/tables/image_estimates_<dataset>.csv`` (seed-averaged
estimates per class), ``results/tables/image_statistics_<dataset>.csv`` and
the per-seed long table ``results/tables/raw/image_estimates_<dataset>_raw.csv``
(not tracked).
"""

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    IMAGE_SEED_BASE, REPO_ROOT, apply_subset,
    make_parser, resolve_estimators, save_raw, save_table, two_nn_reference_row,
    two_nn_row,
)
from danco import DANCo

DATA_DIR = REPO_ROOT / "data"
IMAGENET_DIR = DATA_DIR / "imagenet_training_single_objs"
IMAGENET_CATEGORIES = {"koala": "n01882714", "butterfly": "n02279972"}
CROP = 224


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def load_mnist():
    """Digits 3 and 7 of the MNIST *test* split, flattened."""
    from torchvision import datasets, transforms

    mnist = datasets.MNIST(root=str(DATA_DIR), train=False, download=True,
                           transform=transforms.ToTensor())
    x = mnist.data.numpy()
    y = mnist.targets.numpy()
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    return {str(digit): flat[y == digit] for digit in (3, 7)}


def load_cifar():
    """Classes 2 (bird) and 3 (cat) of the CIFAR-10 *train* split."""
    import torchvision.datasets as ds
    import torchvision.transforms as T

    cifar = ds.CIFAR10(root=str(DATA_DIR), train=True, download=True,
                       transform=T.ToTensor())
    x = np.asarray(cifar.data)
    y = np.asarray(cifar.targets)
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    return {str(label): flat[y == label] for label in (2, 3)}


def _load_category(tag):
    """Centre-cropped 224x224 RGB images of one ImageNet synset, flattened.

    Greyscale files and anything smaller than the crop are dropped -- which is
    why ``N`` is a property of the folder rather than a parameter.
    """
    import matplotlib.image as mpimg

    folder = IMAGENET_DIR / tag / "0"
    files = sorted(glob.glob(str(folder / "*")))
    if not files:
        raise SystemExit(
            f"no images under {folder}. The ImageNet subset is a manual "
            "download (figshare 8a039f58c7b84a215b6d) and is not in git."
        )
    crops = []
    for path in files:
        img = mpimg.imread(path)
        if img.ndim != 3:
            continue
        w, h = img.shape[:2]
        if w < CROP or h < CROP:
            continue
        half = CROP // 2
        crops.append(img[w // 2 - half:w // 2 + half,
                         h // 2 - half:h // 2 + half, :])
    if not crops:
        raise SystemExit(f"every image under {folder} was filtered out")
    stacked = np.array(crops)
    return stacked.reshape(stacked.shape[0], CROP * CROP * 3).astype(float)


def load_imagenet():
    return {name: _load_category(tag)
            for name, tag in IMAGENET_CATEGORIES.items()}


# Candidate bound for the image experiments (Supplementary S3.1).
MAXDIM = 200

DATASETS = {
    # name: (loader, N override or None, share one estimator across classes?)
    "mnist": (load_mnist, None, False),
    "cifar": (load_cifar, 5000, True),
    "imagenet": (load_imagenet, None, False),
}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run_dataset(name, keys, args):
    loader, n_override, share = DATASETS[name]
    print(f"\n{name}: loading...", flush=True)
    datasets = loader()
    datasets, is_subset = apply_subset(datasets, args.subset)
    for label, data in datasets.items():
        print(f"  class {label}: {data.shape}", flush=True)

    labels = list(datasets)
    per_seed = {col: []
                for col in ("aDANCo", "DANCo_G", "DANCo_nus", "DANCo_G_nus")}
    raw = []
    first_stats = {}

    for seed in range(args.seeds):
        shared = {}
        for label in labels:
            data = datasets[label]
            N = n_override if n_override is not None else data.shape[0]
            if N != data.shape[0]:
                data = data[:N]

            def build(version):
                return DANCo(N=N, k=10, version=version,
                             random_state=IMAGE_SEED_BASE + seed, fractal=True,
                             verbose=False, backend=args.backend)

            if share:
                estimators = shared.setdefault(
                    "est", {"danco": build("DANCo"),
                            "gride": build("DANCo_gride")})
            else:
                estimators = {"danco": build("DANCo"),
                              "gride": build("DANCo_gride")}

            t0 = time.time()
            stats = estimators["danco"].evaluate_all(
                data, maxDim=MAXDIM, n_jobs=args.n_jobs, spline_file=None)
            stats_g = estimators["gride"].evaluate_all(
                data, maxDim=MAXDIM, n_jobs=args.n_jobs, spline_file=None)
            values = {
                "aDANCo": stats["de_standard"],
                "DANCo_nus": stats["de_nustrip"],
                "DANCo_G": stats_g["de_standard"],
                "DANCo_G_nus": stats_g["de_nustrip"],
            }
            for col, value in values.items():
                per_seed[col].append((label, value))
                raw.append(dict(dataset=name, seed=IMAGE_SEED_BASE + seed,
                                label=label, estimator=col, value=value))
            first_stats.setdefault(label, (stats, stats_g))
            print(f"  seed {IMAGE_SEED_BASE + seed} class {label}: "
                  f"aDANCo={values['aDANCo']:.4f} "
                  f"G={values['DANCo_G']:.4f} "
                  f"nus={values['DANCo_nus']:.4f} "
                  f"G_nus={values['DANCo_G_nus']:.4f}  "
                  f"({time.time() - t0:.1f}s)", flush=True)

    estimates = pd.DataFrame(index=labels)
    for col, pairs in per_seed.items():
        estimates[col] = [
            np.mean([v for lab, v in pairs if lab == label]) for label in labels
        ]
    if "twonn" in keys:
        subset = {label: datasets[label][:n_override] if n_override else
                  datasets[label] for label in labels}
        twonn_ref = two_nn_reference_row(subset)
        estimates["2NN_ref"] = [twonn_ref[label] for label in labels]
        for label in labels:
            raw.append(dict(dataset=name, seed=None, label=label,
                            estimator="2NN_ref", value=twonn_ref[label]))

    if "twonn_legacy" in keys:
        subset = {label: datasets[label][:n_override] if n_override else
                  datasets[label] for label in labels}
        twonn = two_nn_row(subset, n_jobs=args.n_jobs)
        estimates["2NN"] = [twonn[label] for label in labels]
        for label in labels:
            raw.append(dict(dataset=name, seed=None, label=label,
                            estimator="2NN", value=twonn[label]))

    statistics = pd.DataFrame(index=labels)
    statistics["nu"] = [first_stats[l][0]["mu_nu"] for l in labels]
    statistics["tau"] = [first_stats[l][0]["mu_tau"] for l in labels]
    statistics["dhat"] = [first_stats[l][0]["dhat"] for l in labels]
    statistics["dhat_G"] = [first_stats[l][1]["dhat"] for l in labels]
    return estimates, statistics, raw, is_subset


def main(argv=None):
    parser = make_parser(
        description=__doc__.split("\n\n")[0],
        default_output="results/tables/",
        default_estimators="danco,gride,nustrip,twonn",
    )
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"],
                        default="all")
    args = parser.parse_args(argv)
    keys = resolve_estimators(args.estimators)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        estimates, statistics, raw, is_subset = run_dataset(name, keys, args)

        out_dir = (args.output if args.output.suffix == ""
                   else args.output.parent)
        print()
        print(estimates.to_string(float_format=lambda v: f"{v:10.6f}"))
        print(statistics.to_string(float_format=lambda v: f"{v:10.6f}"))
        save_table(estimates, Path(out_dir) / f"image_estimates_{name}.csv")
        save_table(statistics, Path(out_dir) / f"image_statistics_{name}.csv")
        save_raw(raw, Path(out_dir) / f"image_estimates_{name}.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
