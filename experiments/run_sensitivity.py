#!/usr/bin/env python
"""k- and N-sensitivity of the DANCo variants on the 24-manifold benchmark.

    python experiments/run_sensitivity.py --mode k
    python experiments/run_sensitivity.py --mode n
    python experiments/run_sensitivity.py --mode both --reps 5

Two sweeps answering the "is k=10 cherry-picked?" and "how does accuracy
scale with sample size?" questions:

* ``k`` mode: k in {5, 10, 15, 20} at N=2500. DANCo-Gride keeps its
  coupling k1 = ceil(k/2), k2 = 2*k1 so the neighborhood budget matches.
* ``n`` mode: N in {625, 1250, 2500, 5000} at k=10; each N draws its own
  manifolds so subsampling artifacts cannot leak between levels.

Seed protocol: data ``BenchmarkManifolds(random_state=3000 + rep)``,
estimator ``random_state=600 + rep``; both disjoint from the seeds in
run_baselines (including its noise sweep). Resume via the output CSV.

Output: results/baselines/sensitivity.csv with columns
mode, k, n, manifold, true_d, estimator, rep, dhat, seconds.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from common import REPO_ROOT, two_nn_reference
from danco import DANCo

DATA_SEED_BASE = 3000
EST_SEED_BASE = 600
KS = (5, 10, 15, 20)
NS = (625, 1250, 2500, 5000)
OUT = REPO_ROOT / "results" / "baselines" / "sensitivity.csv"

VARIANTS = {
    "accDANCo": ("DANCo", "de_standard"),
    "DANCo_nustrip": ("DANCo", "de_nustrip"),
    "DANCo_Gride": ("DANCo_gride", "de_standard"),
    "DANCo_Gride_nustrip": ("DANCo_gride", "de_nustrip"),
}


def generate(rep, n):
    from skdim.datasets import BenchmarkManifolds

    bm = BenchmarkManifolds(random_state=DATA_SEED_BASE + rep)
    data = bm.generate(n=n)
    truth = {name: bm._dict_truth[name][0] for name in data}
    return data, truth


def sweep_cells(mode):
    if mode == "k":
        return [(k, 2500) for k in KS]
    return [(10, n) for n in NS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["k", "n", "both"], default="both")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)

    modes = ["k", "n"] if args.mode == "both" else [args.mode]
    rows, done = [], set()
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {(r["mode"], int(r["k"]), int(r["n"]), r["manifold"],
                 r["estimator"], int(r["rep"])) for r in rows}
        print(f"resume: {len(done)} rows present")

    for mode in modes:
        for rep in range(args.reps):
            for k, n in sweep_cells(mode):
                data, truth = generate(rep, n)
                estimators = {
                    version: DANCo(N=n, k=k, version=version,
                                   random_state=EST_SEED_BASE + rep,
                                   fractal=True, verbose=False)
                    for version in {v for v, _ in VARIANTS.values()}
                }
                for name, X in data.items():
                    if (mode, k, n, name, "2NN_ref", rep) not in done:
                        t0 = time.time()
                        val = two_nn_reference(X)
                        rows.append(dict(mode=mode, k=k, n=n, manifold=name,
                                         true_d=truth[name],
                                         estimator="2NN_ref",
                                         rep=rep, dhat=val,
                                         seconds=round(time.time() - t0, 2)))
                    cache = {}
                    for col, (version, stat) in VARIANTS.items():
                        if (mode, k, n, name, col, rep) in done:
                            continue
                        if version not in cache:
                            t0 = time.time()
                            stats = estimators[version].evaluate_all(
                                X, maxDim=100, n_jobs=args.n_jobs)
                            cache[version] = (stats, time.time() - t0)
                        stats, secs = cache[version]
                        rows.append(dict(mode=mode, k=k, n=n, manifold=name,
                                         true_d=truth[name], estimator=col,
                                         rep=rep, dhat=float(stats[stat]),
                                         seconds=round(secs, 2)))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.output.with_suffix(".tmp")
                pd.DataFrame(rows).to_csv(tmp, index=False)
                tmp.replace(args.output)
                print(f"mode={mode} rep={rep} k={k} n={n}: "
                      f"{len(rows)} rows total", flush=True)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
