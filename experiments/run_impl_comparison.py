#!/usr/bin/env python
"""MiND--Full clean-benchmark rows, with stock and legacy-Bessel provenance arms.

    python experiments/run_impl_comparison.py --impls ours --reps 20   # paper
    python experiments/run_impl_comparison.py --manifolds M1_Sphere,M9_Affine --reps 1
    python experiments/run_impl_comparison.py --summarize   # tables from the CSV

Purpose.  The paper uses only the ``ours`` arm: ``--impls ours --reps 20``
produces the MiND--Full rows of the clean benchmark (Main Table 1, Figure S5)
on the same 20 data replicates as :mod:`run_baselines`.  The command-line
defaults (all arms, ``--reps 5``) are not the paper setting.  The
``skdim_stock`` and ``ours_legacy_bessel`` arms compare against
``skdim.id.DANCo`` (scikit-dimension 0.3.7); the paper does not report them.

Three arms:

* ``skdim_stock``  -- ``skdim.id.DANCo(k=10, D=100)``, all defaults otherwise:
  the reference implementation.
* ``ours``         -- ``danco.DANCo(N=2500, k=10, version='DANCo',
  fractal=True)``, ``evaluate(X, maxDim=100)``: this repo's implementation.
* ``ours_legacy_bessel`` -- ``ours`` plus ``use_legacy_bessel=True``, isolating
  the Bessel-overflow path.  At ``maxDim=100`` the overflow (first affecting
  candidate 278 at N = 500) cannot trigger, so this arm agreeing with ``ours`` to
  the last digit is itself a sanity data point, not a redundancy.

Data.  Exactly the protocol of :mod:`run_baselines`: rep ``r`` regenerates the
full 24-manifold suite from ``BenchmarkManifolds(random_state=REP_SEED_BASE+r)``
with ``N = 2500`` points, then filters by ``--manifolds``.  The generator
functions are *imported* from ``run_baselines`` rather than re-derived, so the
two CSVs are computed on byte-identical data and can be joined by
``(manifold, rep)``.

Calibration is shared per (arm, rep), which changes nothing.  Both
implementations grow their Monte-Carlo calibration strictly sequentially from
dimension 1 off a single RNG seeded with the rep seed, and neither consumes RNG
draws anywhere else, so the cache entries for dimensions ``1..m`` are identical
whether the cache is grown lazily to ``m`` inside one fit or pre-grown to 100
up front.  Pre-growing once per rep and handing the same cache to every
per-manifold estimator therefore reproduces the spec-literal "fresh estimator
per dataset" numbers exactly (verified to the last digit for both
implementations) while paying the calibration cost once per rep instead of
24 times.  For our estimator the two Bessel arms share one cache as well:
``use_legacy_bessel`` only affects the KL evaluation, never the calibration
statistics.  No estimate depends on the order manifolds are visited in, which
is what makes CSV resume sound here.  Consequence for the ``seconds`` column: it times the
per-manifold estimate only; the once-per-rep calibration cost is printed and
recorded under the pseudo-manifold ``__calibration__`` (status ``cal``,
``true_d`` NaN), which summaries ignore.

Robustness and resume are inherited from :mod:`run_baselines`: a per-cell
``SIGALRM`` timeout (default 600 s) records a NaN row with ``status='timeout'``
and the sweep continues; the CSV is the checkpoint, flushed atomically after
every manifold, and finished ``(impl, manifold, rep)`` keys are skipped on
restart.  ``--skip-slow`` additionally skips ``skdim_stock`` on the manifolds
whose ambient dimension exceeds ``--slow-ambient-dim`` (default 25): the stock
data-side statistic runs a per-point likelihood search whose cost grows with
ambient dimension, and those cells are the ones that flirt with the timeout.

This script is additive: it writes only to ``results/impl_comparison/`` and
touches nothing under ``src/``, ``tests/`` or ``results/tables/``.
"""

import argparse
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baselines import (  # noqa: E402  (protocol shared with the baselines)
    ERROR_RATE_TOL, N_POINTS, REP_SEED_BASE, generate_manifolds, load_rows,
    select_manifolds, write_csv,
)

K = 10
MAXDIM = 100
DEFAULT_REPS = 5           # skdim DANCo is slow; 5 reps suffice for mean+-sd
DEFAULT_TIMEOUT = 600.0
CAL_MANIFOLD = "__calibration__"   # pseudo-row carrying the per-rep cal time

OUTPUT_DIR = REPO_ROOT / "results" / "impl_comparison"
CSV_NAME = "impl_comparison.csv"
COLUMNS = ["manifold", "true_d", "impl", "rep", "dhat", "seconds", "status"]
KEY = ("impl", "manifold", "rep")

ALL_IMPLS = ["skdim_stock", "ours", "ours_legacy_bessel"]


# ---------------------------------------------------------------------------
# calibration, once per (arm, rep)
# ---------------------------------------------------------------------------
def build_ours_calibration(rep):
    """Grow our calibration cache to MAXDIM with the rep-seeded RNG.

    This is the same loop ``danco.DANCo.evaluate`` runs lazily (hyperball in
    dimension d, then the three-number statistic), executed up front so the
    cache can be shared read-only by every per-manifold estimator of the rep.
    """
    from danco import DANCo

    donor = DANCo(N=N_POINTS, k=K, version="DANCo", fractal=True,
                  random_state=REP_SEED_BASE + rep)
    cache = donor.calibration_statistic
    while cache["maxdim"] < MAXDIM:
        d = cache["maxdim"] + 1
        cache["statistics"].append(
            donor.danco_statistic(donor.hyperBall(N_POINTS, d, center=[0] * d)))
        cache["maxdim"] = d
    return cache


def build_skdim_calibration(rep):
    """Grow the stock calibration to MAXDIM, exactly as ``fit`` would.

    ``skdim.id.DANCo`` only grows its calibration inside ``fit``; to pay for it
    once per rep we replicate the three attribute assignments ``fit`` performs
    before growth (``_k``, ``_D``, ``random_state_``) and then drive the stock
    growth loop itself, so every defect in the stock calibration path is
    faithfully present in the shared cache.
    """
    import skdim.id as skid
    from sklearn.utils import check_random_state

    donor = skid.DANCo(k=K, D=MAXDIM, random_state=REP_SEED_BASE + rep)
    donor._k = K
    donor._D = MAXDIM
    donor.random_state_ = check_random_state(donor.random_state)
    cal = donor._DancoCalibrationData(N_POINTS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while cal["maxdim"] < MAXDIM:
            cal = donor._increaseMaxDimByOne(cal)
    return cal


class CalibrationPool:
    """Lazy per-(family, rep) calibration caches, so resume pays for nothing.

    ``ours`` and ``ours_legacy_bessel`` share one cache (the Bessel switch does
    not touch calibration); ``skdim_stock`` has its own.  A cache is built the
    first time a cell of its rep actually needs computing, and its build time
    is reported to ``on_built`` so the caller can log it as a pseudo-row.
    """

    _FAMILY = {"skdim_stock": "skdim", "ours": "ours",
               "ours_legacy_bessel": "ours"}
    _BUILDERS = {"skdim": build_skdim_calibration, "ours": build_ours_calibration}

    def __init__(self, on_built=None):
        self._caches = {}
        self.on_built = on_built

    def get(self, impl, rep):
        family = self._FAMILY[impl]
        key = (family, rep)
        if key not in self._caches:
            t0 = time.perf_counter()
            self._caches[key] = self._BUILDERS[family](rep)
            seconds = time.perf_counter() - t0
            print(f"  [{family} calibration for rep {rep}: {seconds:.1f}s]",
                  flush=True)
            if self.on_built is not None:
                self.on_built(family, rep, seconds)
        return self._caches[key]


# ---------------------------------------------------------------------------
# one estimate, defensively (same contract as run_baselines.estimate)
# ---------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _fit_once(impl, X, rep, calibration):
    """One estimate with a pre-grown calibration cache; may raise."""
    seed = REP_SEED_BASE + rep
    if impl == "skdim_stock":
        import skdim.id as skid

        estimator = skid.DANCo(k=K, D=MAXDIM, calibration_data=calibration,
                               ver="DANCo", fractal=True, random_state=seed)
        estimator.fit(X)
        return float(estimator.dimension_)

    from danco import DANCo

    estimator = DANCo(N=N_POINTS, k=K, version="DANCo", fractal=True,
                      random_state=seed, calibration_data=calibration,
                      use_legacy_bessel=(impl == "ours_legacy_bessel"))
    dimension, _ = estimator.evaluate(X, maxDim=MAXDIM)
    return float(dimension)


def estimate(impl, X, rep, calibration, timeout=DEFAULT_TIMEOUT):
    """``(dhat, seconds, status)`` -- never raises."""
    can_alarm = hasattr(signal, "SIGALRM") and timeout and timeout > 0
    previous = None
    if can_alarm:
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dhat = _fit_once(impl, X, rep, calibration)
        status = "ok" if np.isfinite(dhat) else "nan"
    except _Timeout:
        dhat, status = np.nan, "timeout"
    except BaseException as exc:  # noqa: BLE001 -- one bad cell must not stop the sweep
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
# sweep
# ---------------------------------------------------------------------------
def slow_manifolds(datasets, threshold):
    """Names whose ambient dimension exceeds ``threshold``."""
    return {name for name, X in datasets.items() if X.shape[1] > threshold}


def run_comparison(args, impls, path):
    rows, done = ([], set()) if not args.resume else load_rows(
        path, COLUMNS, KEY)
    new = skipped = 0

    def note_calibration(family, rep, seconds):
        """Record a build once; a rerun that rebuilds must not duplicate it."""
        key = (f"cal_{family}", CAL_MANIFOLD, rep)
        if key not in done:
            rows.append(dict(manifold=CAL_MANIFOLD, true_d=np.nan,
                             impl=f"cal_{family}", rep=rep, dhat=np.nan,
                             seconds=seconds, status="cal"))
            done.add(key)

    for rep in range(args.reps):
        pool = CalibrationPool(on_built=note_calibration)
        datasets, truth = generate_manifolds(rep, n=N_POINTS)
        datasets = select_manifolds(datasets, args.manifolds)
        slow = slow_manifolds(datasets, args.slow_ambient_dim)
        print(f"\nrep {rep} (data seed {REP_SEED_BASE + rep}): "
              f"{len(datasets)} manifold(s)", flush=True)
        for manifold, X in datasets.items():
            for impl in impls:
                if (impl, manifold, rep) in done:
                    skipped += 1
                    continue
                if args.skip_slow and impl == "skdim_stock" and manifold in slow:
                    dhat, seconds, status = np.nan, 0.0, "skipped: slow"
                else:
                    calibration = pool.get(impl, rep)
                    dhat, seconds, status = estimate(impl, X, rep, calibration,
                                                     args.timeout)
                rows.append(dict(manifold=manifold, true_d=truth[manifold],
                                 impl=impl, rep=rep, dhat=dhat,
                                 seconds=seconds, status=status))
                done.add((impl, manifold, rep))
                new += 1
                flag = "" if status == "ok" else f"  [{status}]"
                print(f"  {manifold:<18} d={truth[manifold]:<5.0f} "
                      f"{impl:<19} dhat={dhat:8.3f}  {seconds:7.2f}s{flag}",
                      flush=True)
            write_csv(rows, path, COLUMNS)
        write_csv(rows, path, COLUMNS)
    print(f"\n{new} new row(s), {skipped} skipped -> {path}")
    return rows


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _impl_stats(group, n_manifolds):
    """MPE / error rate / failure count, same conventions as run_baselines.

    MPE is the mean absolute percentage error of the per-manifold mean over
    reps; a failed estimate is excluded from MPE but counts as an error for the
    error rate (cmFSA convention: relative error above 10%).
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


def summarize(path, top=5):
    if not path.exists():
        print(f"[summarize] no such file: {path}")
        return
    df = pd.read_csv(path)
    cal = df[df["manifold"] == CAL_MANIFOLD]
    df = df[df["manifold"] != CAL_MANIFOLD]
    n_manifolds = df["manifold"].nunique()
    print(f"\n=== {path.name}: {len(df)} rows, {n_manifolds} manifold(s), "
          f"reps {sorted(df['rep'].unique())} ===")
    if len(cal):
        per = cal.groupby("impl")["seconds"].mean()
        print("calibration time per rep: "
              + ", ".join(f"{name} {secs:.0f}s" for name, secs in per.items()))

    print("\nper-manifold mean +- sd over reps")
    cells = df.groupby(["manifold", "impl"]).agg(
        mean=("dhat", "mean"), sd=("dhat", "std")).reset_index()
    cells["cell"] = [f"{m:.2f}+-{0.0 if pd.isna(s) else s:.2f}"
                     for m, s in zip(cells["mean"], cells["sd"])]
    wide = cells.pivot(index="manifold", columns="impl", values="cell")
    truth = df.groupby("manifold")["true_d"].first()
    wide.insert(0, "true_d", truth.reindex(wide.index).astype(int))
    order = [c for c in ["true_d"] + ALL_IMPLS if c in wide.columns]
    print(wide[order].to_string())

    print("\nper-implementation summary")
    header = (f"  {'impl':<19}{'MPE':>9}{'err_rate':>10}{'n_fail':>8}"
              f"{'mean_s':>9}{'total_s':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in [i for i in ALL_IMPLS if i in set(df["impl"])]:
        group = df[df["impl"] == name]
        mpe, error_rate, n_fail = _impl_stats(group, n_manifolds)
        print(f"  {name:<19}{mpe:>9.2f}{error_rate:>10.3f}{n_fail:>8}"
              f"{group['seconds'].mean():>9.2f}{group['seconds'].sum():>10.1f}")

    have = set(df["impl"])
    if {"skdim_stock", "ours"} <= have:
        means = df.pivot_table(index="manifold", values="dhat",
                               columns="impl", aggfunc="mean")
        gap = (means["skdim_stock"] - means["ours"]).abs()
        table = pd.DataFrame({
            "true_d": truth.reindex(gap.index),
            "ours": means["ours"],
            "skdim_stock": means["skdim_stock"],
            "abs_gap": gap,
        }).sort_values("abs_gap", ascending=False).head(top)
        print(f"\ntop {top} manifolds where skdim_stock and ours disagree most "
              "(|mean dhat difference|)")
        print(table.to_string(float_format=lambda v: f"{v:8.2f}"))


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--impls", default=",".join(ALL_IMPLS),
                        help=f"comma-separated subset of {ALL_IMPLS}")
    parser.add_argument("--manifolds", default=None,
                        help="comma-separated short names (M1,M5) or full "
                             "names; default is all 24")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="independent replicates; rep r uses data seed "
                             f"{REP_SEED_BASE}+r (same stream as run_baselines)")
    parser.add_argument("--summarize", action="store_true",
                        help="print tables from the existing CSV and exit")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="seconds per (impl, manifold) cell before it is "
                             "recorded as a NaN")
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip skdim_stock on manifolds with ambient "
                             "dimension above --slow-ambient-dim instead of "
                             "burning the timeout on them")
    parser.add_argument("--slow-ambient-dim", type=int, default=25,
                        help="ambient-dimension threshold for --skip-slow")
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore the existing CSV and recompute everything")
    args = parser.parse_args(argv)

    path = args.outdir / CSV_NAME
    if args.summarize:
        summarize(path)
        return 0

    impls = [i.strip() for i in args.impls.split(",") if i.strip()]
    unknown = [i for i in impls if i not in ALL_IMPLS]
    if unknown:
        raise SystemExit(f"unknown impl(s) {unknown}; choose from {ALL_IMPLS}")
    print(f"impls={impls}  reps={args.reps}  N={N_POINTS}  k={K}  "
          f"maxDim={MAXDIM}  timeout={args.timeout}s"
          + ("  skip-slow" if args.skip_slow else ""))

    run_comparison(args, impls, path)
    summarize(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
