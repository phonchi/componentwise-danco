#!/usr/bin/env python
"""Nine reference ID estimators on the 24-manifold benchmark, with a noise sweep.

    python experiments/run_baselines.py --estimators MLE,TwoNN --manifolds M1,M5 --reps 2
    python experiments/run_baselines.py --reps 20                 # background job
    python experiments/run_baselines.py --noise-sweep --reps 20   # much longer
    python experiments/run_baselines.py --summarize               # tables from the CSVs

Each replicate regenerates every manifold from
``BenchmarkManifolds(random_state=REP_SEED_BASE + r)``, with
``REP_SEED_BASE = 1000``. Results are written to ``results/baselines/``.

**Manifolds are always generated as a full set of 24, then filtered.**  The 24
come off a single RNG in a fixed order, so ``--manifolds M1,M5`` must not change
what M1 contains; generating all of them and dropping the rest keeps a smoke test
comparable with the full sweep.

**Parameters.**  skdim defaults throughout, with two deliberate exceptions:

* ``k = 10`` wherever a neighbourhood size is exposed (``MLE.K``,
  ``MiND_ML.k``), matching the ``k`` DANCo is run with -- otherwise the
  comparison confounds the estimator with its neighbourhood.
* ``MiND_ML.D = 100`` instead of the skdim default 10.  ``D`` is the upper bound
  of the likelihood search, and seven of the 24 manifolds have a true dimension
  above 10 (up to 70 for M10d); left at the default, MiND_ML would report a
  saturated 10 for all of them and the comparison would be a measurement of the
  bound.  100 is DANCo's ``maxDim`` in ``run_impl_comparison.py``, so both estimators
  get the same ceiling.  ``--mind-dmax`` overrides it.

**Robustness.**  An estimator that raises, returns NaN/inf, or exceeds
``--timeout`` seconds on one dataset is recorded as a NaN row with the reason in
``status`` and the sweep continues; one bad (estimator, manifold) pair must not
cost a multi-hour run.  The timeout is a ``SIGALRM`` on the main thread, so it
interrupts pure-Python and most NumPy loops but cannot preempt a single long call
inside a C extension that never returns to the interpreter.

**Resume.**  Both CSVs are the checkpoint: on start the existing file is read and
every (estimator, manifold, rep[, sigma, realization]) key in it is skipped.
Rows are flushed after each manifold via an atomic replace, so an interrupted run
loses at most the manifold in flight and a rerun of the same command finishes
without redoing anything.
"""

import argparse
import os
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# protocol constants
# ---------------------------------------------------------------------------
N_POINTS = 2500          # same sample size as the DANCo benchmark runs
K = 10                   # neighbourhood size, aligned with the DANCo runs
REP_SEED_BASE = 1000     # data seed for rep r; shared by the matched implementation comparison
NOISE_SEED_BASE = 90000  # entropy for the noise realizations (data-seed free)
DEFAULT_REPS = 20
DEFAULT_TIMEOUT = 300.0
SIGMAS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
N_REALIZATIONS = 5
ERROR_RATE_TOL = 0.1     # |dhat - d| / d above which an estimate is "wrong"

OUTPUT_DIR = REPO_ROOT / "results" / "baselines"
BENCHMARK_CSV = "baselines_benchmark.csv"
NOISE_CSV = "baselines_noise.csv"

BENCHMARK_COLUMNS = ["manifold", "true_d", "estimator", "rep", "dhat",
                     "seconds", "status"]
NOISE_COLUMNS = ["manifold", "true_d", "estimator", "rep", "sigma",
                 "realization", "dhat", "seconds", "status"]
BENCHMARK_KEY = ("estimator", "manifold", "rep")
NOISE_KEY = ("estimator", "manifold", "rep", "sigma", "realization")


def build_estimators(names, mind_dmax=100):
    """Build the canonical paper baselines.

    TWO-NN and MLE use the shared exact-deduplication contracts; the remaining
    estimators use the scikit-dimension constructors below. Built fresh per
    call because fitted estimators retain ``dimension_`` from the previous fit.
    """
    import skdim.id as skid
    from paper_methods import PaperMLE, PaperTwoNN

    factories = {
        "MLE": lambda: PaperMLE(k=K),
        "TwoNN": PaperTwoNN,
        "lPCA": lambda: skid.lPCA(verbose=False),
        "ESS": lambda: skid.ESS(),
        "MADA": lambda: skid.MADA(),
        "TLE": lambda: skid.TLE(),
        "CorrInt": lambda: skid.CorrInt(),      # k1=10, k2=20 -- already at k
        "MiND_ML": lambda: skid.MiND_ML(k=K, D=mind_dmax),
        "FisherS": lambda: skid.FisherS(),
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise SystemExit(f"unknown estimator(s) {unknown}; "
                         f"choose from {list(factories)}")
    return {name: factories[name]() for name in names}


ALL_ESTIMATORS = ["MLE", "TwoNN", "lPCA", "ESS", "MADA", "TLE", "CorrInt",
                  "MiND_ML", "FisherS"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def generate_manifolds(rep, n=N_POINTS):
    """The full 24-manifold suite for one replicate: ``({name: X}, {name: d})``.

    Walk ``_dict_truth`` and call the per-manifold generator, preserving
    scikit-dimension's ambient dimensions and generator arguments.
    """
    from skdim.datasets import BenchmarkManifolds

    benchmark = BenchmarkManifolds(random_state=REP_SEED_BASE + rep)
    datasets, truth = {}, {}
    for name, (d, dim, _) in benchmark._dict_truth.items():
        datasets[name] = benchmark.dict_gen[name](n=n, dim=dim, d=d)
        truth[name] = float(benchmark.truth.loc[name, "Intrinsic Dimension"])
    return datasets, truth


def select_manifolds(datasets, spec):
    """Filter by short name (``M1`` -> ``M1_Sphere``, ``M5`` -> both helices).

    A token matches a manifold when it is the whole name or a prefix that does
    not cut a number in half, so ``M1`` selects ``M1_Sphere`` but not
    ``M10a_Cubic``.
    """
    if not spec:
        return datasets
    wanted = [t.strip() for t in spec.split(",") if t.strip()]
    keep, unmatched = [], []
    for token in wanted:
        hits = [name for name in datasets
                if name == token
                or (name.startswith(token) and not name[len(token):][:1].isdigit())]
        if not hits:
            unmatched.append(token)
        keep.extend(hits)
    if unmatched:
        raise SystemExit(f"--manifolds matched nothing for {unmatched}\n"
                         f"available: {list(datasets)}")
    return {name: datasets[name] for name in datasets if name in set(keep)}


def add_noise(X, sigma, rep, manifold_index, sigma_index, realization):
    """Additive isotropic Gaussian noise from a stream independent of the data.

    The realization RNG is keyed on (rep, manifold, sigma, realization) only --
    it never touches the ``BenchmarkManifolds`` generator -- so the same clean
    manifold is seen under every noise level and the noise cannot be confounded
    with a different draw of the manifold itself.
    """
    if sigma == 0.0:
        return X
    rng = np.random.default_rng(
        [NOISE_SEED_BASE, rep, manifold_index, sigma_index, realization])
    return X + rng.standard_normal(X.shape) * sigma


# ---------------------------------------------------------------------------
# one estimate, defensively
# ---------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def estimate(estimator, X, timeout=DEFAULT_TIMEOUT):
    """``(dhat, seconds, status)`` -- never raises.

    ``status`` is ``ok``, ``timeout``, ``nan``, or ``error: <type>: <message>``;
    ``dhat`` is NaN for anything but ``ok``.
    """
    can_alarm = hasattr(signal, "SIGALRM") and timeout and timeout > 0
    previous = None
    if can_alarm:
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X)
        dhat = float(getattr(estimator, "dimension_", np.nan))
        status = "ok" if np.isfinite(dhat) else "nan"
    except _Timeout:
        dhat, status = np.nan, "timeout"
    except BaseException as exc:  # noqa: BLE001 -- a bad estimator must not stop the sweep
        message = str(exc).replace("\n", " ")[:120]
        dhat, status = np.nan, f"error: {type(exc).__name__}: {message}"
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
    if status != "ok":
        dhat = float("nan")
    return dhat, time.perf_counter() - t0, status


# ---------------------------------------------------------------------------
# CSV checkpointing
# ---------------------------------------------------------------------------
def load_rows(path, columns, key):
    """Existing rows plus the set of finished keys (``([], set())`` if absent)."""
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing column(s) {missing}; move it aside "
                         "or pass --no-resume")
    rows = df[columns].to_dict("records")
    done = {_key(row, key) for row in rows}
    print(f"resume: {len(rows)} row(s) already in {path}")
    return rows, done


def _key(row, key):
    """Hashable key with floats normalised, so 0.01 from CSV matches 0.01."""
    out = []
    for field in key:
        value = row[field]
        out.append(round(float(value), 6) if isinstance(value, float) else value)
    return tuple(out)


def write_csv(rows, path, columns):
    """Atomic rewrite: an interrupted flush must not truncate the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def run_benchmark(args, names, path):
    rows, done = ([], set()) if not args.resume else load_rows(
        path, BENCHMARK_COLUMNS, BENCHMARK_KEY)
    new = skipped = 0

    for rep in range(args.reps):
        datasets, truth = generate_manifolds(rep, n=args.n_points)
        datasets = select_manifolds(datasets, args.manifolds)
        print(f"\nrep {rep} (data seed {REP_SEED_BASE + rep}): "
              f"{len(datasets)} manifold(s)", flush=True)
        for manifold, X in datasets.items():
            for name in names:
                if (name, manifold, rep) in done:
                    skipped += 1
                    continue
                estimator = build_estimators([name], args.mind_dmax)[name]
                dhat, seconds, status = estimate(estimator, X, args.timeout)
                rows.append(dict(manifold=manifold, true_d=truth[manifold],
                                 estimator=name, rep=rep, dhat=dhat,
                                 seconds=seconds, status=status))
                done.add((name, manifold, rep))
                new += 1
                flag = "" if status == "ok" else f"  [{status}]"
                print(f"  {manifold:<18} d={truth[manifold]:<5.0f} {name:<8}"
                      f" dhat={dhat:8.3f}  {seconds:7.2f}s{flag}", flush=True)
            write_csv(rows, path, BENCHMARK_COLUMNS)
    print(f"\n{new} new row(s), {skipped} skipped -> {path}")
    return rows


def run_noise_sweep(args, names, path):
    rows, done = ([], set()) if not args.resume else load_rows(
        path, NOISE_COLUMNS, NOISE_KEY)
    new = skipped = 0
    sigmas = args.sigmas

    for rep in range(args.reps):
        datasets, truth = generate_manifolds(rep, n=args.n_points)
        datasets = select_manifolds(datasets, args.manifolds)
        order = list(datasets)
        print(f"\nrep {rep} (data seed {REP_SEED_BASE + rep}): "
              f"{len(datasets)} manifold(s) x {len(sigmas)} sigma(s)", flush=True)
        for manifold, X in datasets.items():
            mi = order.index(manifold)
            for si, sigma in enumerate(sigmas):
                # sigma = 0 is the clean manifold: every realization would be
                # the identical array, so one row stands for all of them.
                n_real = 1 if sigma == 0.0 else args.realizations
                for realization in range(n_real):
                    Xn = add_noise(X, sigma, rep, mi, si, realization)
                    for name in names:
                        key = (name, manifold, rep, round(float(sigma), 6),
                               realization)
                        if key in done:
                            skipped += 1
                            continue
                        estimator = build_estimators([name],
                                                     args.mind_dmax)[name]
                        dhat, seconds, status = estimate(estimator, Xn,
                                                         args.timeout)
                        rows.append(dict(
                            manifold=manifold, true_d=truth[manifold],
                            estimator=name, rep=rep, sigma=float(sigma),
                            realization=realization, dhat=dhat,
                            seconds=seconds, status=status))
                        done.add(key)
                        new += 1
                        flag = "" if status == "ok" else f"  [{status}]"
                        print(f"  {manifold:<18} s={sigma:<5.2f} r{realization} "
                              f"{name:<8} dhat={dhat:8.3f}  {seconds:6.2f}s"
                              f"{flag}", flush=True)
            write_csv(rows, path, NOISE_COLUMNS)
    print(f"\n{new} new row(s), {skipped} skipped -> {path}")
    return rows


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def _stats(group, n_manifolds):
    """MPE, error rate and failure count for one long-format slice.

    MPE is the mean percentage error over manifolds, computed on the
    *per-manifold mean* over replicates rather than a single run, and
    the denominator is the number of manifolds actually present -- a subset run
    of this script is a legitimate experiment, not a broken table, so scaling by
    a fixed 24 would understate it.  A failed estimate counts as an error for
    the error rate (cmFSA convention: relative error above 10%) and is excluded
    from MPE, which is reported alongside ``n_fail`` so a low MPE bought by
    dropped rows is visible.
    """
    relative = (group["dhat"] - group["true_d"]).abs() / group["true_d"]
    n_fail = int(group["dhat"].isna().sum())
    error_rate = float(((relative > ERROR_RATE_TOL) | relative.isna()).mean())
    per_manifold = group.groupby("manifold").apply(
        lambda g: (np.nanmean(g["dhat"]) - g["true_d"].iloc[0])
        / g["true_d"].iloc[0] if g["dhat"].notna().any() else np.nan,
        include_groups=False)
    mpe = 100.0 / n_manifolds * np.nansum(per_manifold.abs())
    return mpe, error_rate, n_fail


def summarize_benchmark(path):
    if not path.exists():
        print(f"[summarize] no such file: {path}")
        return
    df = pd.read_csv(path)
    n_manifolds = df["manifold"].nunique()
    print(f"\n=== {path.name}: {len(df)} rows, {n_manifolds} manifold(s), "
          f"reps {sorted(df['rep'].unique())} ===")

    print("\nper-manifold mean +- sd over reps")
    cells = df.groupby(["manifold", "estimator"]).agg(
        mean=("dhat", "mean"), sd=("dhat", "std")).reset_index()
    cells["cell"] = [f"{m:.2f}+-{0.0 if pd.isna(s) else s:.2f}"
                     for m, s in zip(cells["mean"], cells["sd"])]
    wide = cells.pivot(index="manifold", columns="estimator", values="cell")
    truth = df.groupby("manifold")["true_d"].first()
    wide.insert(0, "true_d", truth.reindex(wide.index).astype(int))
    print(wide.to_string())

    print("\nper-estimator summary")
    header = (f"  {'estimator':<10}{'MPE':>9}{'err_rate':>10}{'n_fail':>8}"
              f"{'mean_s':>9}{'total_s':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, group in df.groupby("estimator"):
        mpe, error_rate, n_fail = _stats(group, n_manifolds)
        print(f"  {name:<10}{mpe:>9.2f}{error_rate:>10.3f}{n_fail:>8}"
              f"{group['seconds'].mean():>9.2f}{group['seconds'].sum():>10.1f}")


def summarize_noise(path):
    if not path.exists():
        print(f"[summarize] no such file: {path}")
        return
    df = pd.read_csv(path)
    n_manifolds = df["manifold"].nunique()
    print(f"\n=== {path.name}: {len(df)} rows, {n_manifolds} manifold(s), "
          f"sigmas {sorted(df['sigma'].unique())} ===")

    print("\nMPE by sigma")
    mpe_table = {}
    err_table = {}
    for (name, sigma), group in df.groupby(["estimator", "sigma"]):
        mpe, error_rate, _ = _stats(group, n_manifolds)
        mpe_table.setdefault(name, {})[sigma] = mpe
        err_table.setdefault(name, {})[sigma] = error_rate
    print(pd.DataFrame(mpe_table).T.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\nerror rate by sigma")
    print(pd.DataFrame(err_table).T.to_string(float_format=lambda v: f"{v:8.3f}"))


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--estimators", default=",".join(ALL_ESTIMATORS),
                        help="comma-separated subset of "
                             f"{ALL_ESTIMATORS}")
    parser.add_argument("--manifolds", default=None,
                        help="comma-separated short names (M1,M5) or full names;"
                             " default is all 24")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="independent replicates; rep r uses data seed "
                             f"{REP_SEED_BASE}+r")
    parser.add_argument("--noise-sweep", action="store_true",
                        help="Gaussian noise sweep instead of the clean "
                             "benchmark")
    parser.add_argument("--summarize", action="store_true",
                        help="print tables from the existing CSVs and exit")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="cap on BLAS/OpenMP threads (-1 = library default)")
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-points", type=int, default=N_POINTS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="seconds per estimator call before it is recorded "
                             "as a NaN")
    parser.add_argument("--realizations", type=int, default=N_REALIZATIONS,
                        help="noise realizations per sigma (sigma=0 always has "
                             "one: the clean data)")
    parser.add_argument("--sigmas", type=str, default=None,
                        help="comma-separated noise levels "
                             f"(default: {','.join(str(s) for s in SIGMAS)})")
    parser.add_argument("--mind-dmax", type=int, default=100,
                        help="MiND_ML search ceiling; skdim's default of 10 "
                             "saturates on the high-dimensional manifolds")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore the existing CSV and recompute everything")
    args = parser.parse_args(argv)

    args.sigmas = (tuple(float(s) for s in args.sigmas.split(",") if s.strip())
                   if args.sigmas else SIGMAS)
    benchmark_path = args.outdir / BENCHMARK_CSV
    noise_path = args.outdir / NOISE_CSV

    if args.summarize:
        if args.noise_sweep:
            summarize_noise(noise_path)
        else:
            summarize_benchmark(benchmark_path)
        return 0

    names = [n.strip() for n in args.estimators.split(",") if n.strip()]
    build_estimators(names, args.mind_dmax)   # fail fast on a typo
    print(f"estimators={names}  reps={args.reps}  N={args.n_points}  k={K}  "
          f"timeout={args.timeout}s  "
          f"{'noise sweep' if args.noise_sweep else 'clean benchmark'}")

    limiter = None
    if args.n_jobs and args.n_jobs > 0:
        try:
            from threadpoolctl import threadpool_limits
            limiter = threadpool_limits(limits=args.n_jobs)
        except ImportError:
            print("threadpoolctl not installed; --n-jobs ignored")
    try:
        if args.noise_sweep:
            run_noise_sweep(args, names, noise_path)
            summarize_noise(noise_path)
        else:
            run_benchmark(args, names, benchmark_path)
            summarize_benchmark(benchmark_path)
    finally:
        if limiter is not None:
            limiter.unregister()
    return 0


if __name__ == "__main__":
    sys.exit(main())
