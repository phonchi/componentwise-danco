#!/usr/bin/env python
"""Retrain the FastDANCo calibration splines.

    python experiments/train_splines.py --d-max 400 \
        --n-range 450,500,580,640,700 --replicas 35 --seed 42 \
        --output results/splines/fastdanco_splines_400_retrained.pkl

A thin CLI over the four functions in :mod:`danco.spline_training`, kept
verbatim so a retrained surface is comparable to the shipped
``fastdanco_splines_400.pkl``.  The defaults reproduce that file's grid --
``d in 1..400``, five ``N`` values in 450..700, 35 hyperballs per cell -- which
is ``400 x 5 x 35 = 70000`` calibration runs and takes hours.  Nothing is
started without ``--yes``.

The shipped MiND pickle records no seed, so an exact reproduction of that file
is impossible; what a retrained surface can establish is *statistical*
equivalence.  The shipped Gride pickle records ``seed=42`` and the same grid.
``--seed`` fixes this run's hyperballs so at least the retraining is
reproducible: replica ``r`` of every cell uses ``random_state = seed + r``,
i.e. the same ball sequence for every ``(d, N)``, as in the shipped surface.
This correlates cells and is worth knowing about when interpreting the
surface's smoothness.

The estimator injected into the training module is this package's ``DANCo``:
the trainer resolves ``DANCo`` from its module globals and has none of its own.

``--variant gride`` trains the same grid with the Gride distance strategy
(``version='DANCo_gride'``) instead of MiND.  The hyperball sequence is
identical -- replica ``r`` still uses ``random_state = seed + r`` -- so the two
surfaces are built from the *same* balls and differ only in the distance
statistic.  The variant is recorded in the pickle and checked at load time.
"""

import argparse
import pickle
import time
from pathlib import Path

from common import REPO_ROOT
from danco import DANCo
from danco import spline_training as trainer

DEFAULT_N_RANGE = [450, 500, 580, 640, 700]

VARIANT_VERSIONS = {"mind": "DANCo", "gride": "DANCo_gride"}
DEFAULT_OUTPUTS = {
    "mind": "results/splines/fastdanco_splines_400.pkl",
    "gride": "results/splines/fastdanco_splines_gride_400.pkl",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--d-max", type=int, default=400,
                        help="largest intrinsic dimension on the grid")
    parser.add_argument("--d-min", type=int, default=1)
    parser.add_argument("--n-range",
                        default=",".join(str(n) for n in DEFAULT_N_RANGE),
                        help="comma-separated sample sizes")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--replicas", type=int, default=35,
                        help="hyperballs averaged per (d, N) cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=sorted(VARIANT_VERSIONS),
                        default="mind",
                        help="distance strategy the surface is trained for")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to results/splines/"
                             "fastdanco_splines[_gride]_400.pkl per variant")
    parser.add_argument("--no-validate", dest="validate",
                        action="store_false",
                        help="skip the fit-quality report")
    parser.add_argument("--yes", action="store_true",
                        help="actually run; without it the CLI only reports "
                             "the size of the job")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = REPO_ROOT / DEFAULT_OUTPUTS[args.variant]

    d_range = range(args.d_min, args.d_max + 1)
    n_range = [int(n) for n in args.n_range.split(",") if n.strip()]
    cells = len(d_range) * len(n_range)
    print(f"grid: d in [{args.d_min}, {args.d_max}] x N in {n_range}")
    print(f"      {cells} cells x {args.replicas} replicas = "
          f"{cells * args.replicas} calibration runs")
    print(f"variant: {args.variant}")
    print(f"output: {args.output}")
    if not args.yes:
        print("\nnot started (pass --yes to run; expect hours)")
        return 0

    # The trainer resolves DANCo from its module globals and defines none of
    # its own.  It only ever calls
    # DANCo(k=..., N=...), so a factory fixing the version is a drop-in.
    version = VARIANT_VERSIONS[args.variant]

    def _factory(k, N):
        return DANCo(k=k, N=N, version=version)

    trainer.DANCo = _factory

    t0 = time.time()
    stats = trainer.generate_calibration_data(
        k=args.k, d_range=d_range, N_range=n_range,
        n_replicas=args.replicas, random_state=args.seed)
    splines = trainer.fit_splines(stats, d_range, n_range)
    if args.validate:
        trainer.validate_splines(splines, stats, d_range, n_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_splines(splines, args.k, d_range, n_range,
                         output_file=str(args.output))
    # The trainer's saver predates per-variant surfaces; stamp the variant
    # into the pickle so danco.spline.load_splines can refuse a mismatch.
    with open(args.output, "rb") as f:
        payload = pickle.load(f)
    payload["variant"] = args.variant
    payload["seed"] = args.seed
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    print(f"stamped variant={args.variant!r} seed={args.seed}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
