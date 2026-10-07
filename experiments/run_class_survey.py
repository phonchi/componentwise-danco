#!/usr/bin/env python
"""How common is the nu-hat < 1 regime across image classes?

    python experiments/run_class_survey.py

The survey covers 17 classes: the 10 CIFAR-10 classes and the 7 ImageNet
single-object synsets (Supplementary Table S9).

Sampling protocol (fixed before looking at any results):

* **CIFAR-10**: all 10 classes of the *train* split, the first 5000 images
  of each class in dataset order (i.e. every training image; 5000 is the
  full per-class count), raw 32x32x3 pixels flattened to D=3072 float64.
* **ImageNet subset**: every synset directory currently present under
  ``data/imagenet_training_single_objs/`` whose name starts with ``n``
  (the figshare single-object subset; the ``mix`` folder is excluded by
  the rule).  Images are loaded exactly as in ``run_images.py``'s
  ``_load_category``: centre crop 224x224, greyscale files and images
  smaller than the crop dropped, so N is a property of the folder.

For each class only the *data-side* angle statistics are computed -- no
calibration, no dimension estimate -- via the estimator's own kNN + angle
kernels (``per_point_stats`` reused from ``run_angle_diagnostics``, k=10):

* ``mu_nu``: circular mean of the per-point von Mises mean directions
  nu_i, the quantity whose value below 1 defines the regime;
* ``norm_cv``: coefficient of variation of the centred norms
  ||x_i - mean(x)|| (the norm-heterogeneity proxy);
* ``pearson_r``: Pearson correlation between nu_i and the centred norm
  (the mechanism predicts it negative).

Output: ``results/std_datasets/class_survey.csv`` with one row per class
(source, class, N, D, mu_nu, norm_cv, pearson_r) and a printed summary of
the nu-hat < 1 proportion per source and the overall nu-hat range.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import REPO_ROOT
from run_angle_diagnostics import K, per_point_stats
from run_images import IMAGENET_DIR, _load_category
from danco._kernels.angles import indnComb
from danco.backends import get_backend

CIFAR_PER_CLASS = 5000
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck")


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def iter_cifar_classes():
    """Yield ('cifar10', '<label> <name>', X) for all 10 train classes."""
    import torchvision.datasets as ds

    cifar = ds.CIFAR10(root=str(REPO_ROOT / "data"), train=True,
                       download=True)
    x = np.asarray(cifar.data)
    y = np.asarray(cifar.targets)
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    for label in range(10):
        X = flat[y == label][:CIFAR_PER_CLASS]
        yield "cifar10", f"{label}_{CIFAR_CLASSES[label]}", X


def iter_imagenet_classes():
    """Yield ('imagenet', synset, X) for every n* folder present on disk."""
    if not IMAGENET_DIR.exists():
        raise SystemExit(
            f"{IMAGENET_DIR} not found; the ImageNet subset is a manual "
            "download (figshare 8a039f58c7b84a215b6d).")
    synsets = sorted(p.name for p in IMAGENET_DIR.iterdir()
                     if p.is_dir() and p.name.startswith("n"))
    if not synsets:
        raise SystemExit(f"no n* synset folders under {IMAGENET_DIR}")
    for tag in synsets:
        yield "imagenet", tag, _load_category(tag)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=["cifar10", "imagenet", "all"],
                        default="all")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results" / "std_datasets"
                        / "class_survey.csv")
    args = parser.parse_args(argv)

    backend = get_backend("cpu")
    combs = indnComb(K)

    iterators = []
    if args.source in ("cifar10", "all"):
        iterators.append(iter_cifar_classes())
    if args.source in ("imagenet", "all"):
        iterators.append(iter_imagenet_classes())

    rows = []
    for iterator in iterators:
        for source, label, X in iterator:
            t0 = time.time()
            nu_i, _, norms, mu_nu, _ = per_point_stats(X, backend, combs)
            r = pearsonr(nu_i, norms)
            rows.append(dict(
                source=source, **{"class": label},
                N=X.shape[0], D=X.shape[1],
                mu_nu=float(mu_nu),
                norm_cv=float(norms.std() / norms.mean()),
                pearson_r=float(r.statistic),
            ))
            print(f"  {source}/{label}: N={X.shape[0]} D={X.shape[1]} "
                  f"mu_nu={mu_nu:.4f} cv={rows[-1]['norm_cv']:.3f} "
                  f"r={rows[-1]['pearson_r']:+.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nwrote {len(df)} rows -> {args.output}")

    print("\n=== summary ===")
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        below = int((sub["mu_nu"] < 1.0).sum())
        print(f"  {source}: nu-hat < 1.0 in {below}/{len(sub)} classes; "
              f"range [{sub['mu_nu'].min():.4f}, {sub['mu_nu'].max():.4f}]")
    below = int((df["mu_nu"] < 1.0).sum())
    print(f"  overall: {below}/{len(df)} classes with nu-hat < 1.0; "
          f"range [{df['mu_nu'].min():.4f}, {df['mu_nu'].max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
