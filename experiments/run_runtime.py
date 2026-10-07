#!/usr/bin/env python
"""Matched runtime and peak-memory benchmark for the DANCo implementation.

The benchmark separates four stages whose costs answer different questions:

``data_side``
    kNN search plus the distance and angular statistics.  This is the only
    stage that excludes calibration and candidate search.
``spline_evaluate``
    Complete evaluation with the packaged spline calibration.
``full_mc_evaluate``
    Complete evaluation with a cold Monte-Carlo calibration cache.
``cached_mc_evaluate``
    Complete evaluation after that Monte-Carlo cache has been populated.

Every measured repeat runs in a fresh subprocess.  Reported RSS is therefore
per repeat rather than the cumulative high-water mark of the driver process.
One fresh subprocess is discarded as warm-up for each cell.  The requested
candidate bound is never allowed to exceed the ambient dimension; in
particular, the ``maxDim`` sweep raises ``D`` to preserve the requested search
range instead of relying on ``DANCo.evaluate`` to clip it silently.
When CUDA is available, every selected stage runs once per backend; Monte-Carlo
reference generation remains on CPU in both cases, so the complete-stage
comparison changes only the observed-data backend.

Examples
--------
``python experiments/run_runtime.py --repeats 5``
``python experiments/run_runtime.py --sweeps D --values 100,1000 --repeats 2``
``python experiments/run_runtime.py --sweeps maxDim --stages full_mc_evaluate``
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:  # module import in tests versus direct script execution
    from .common import REPO_ROOT
except ImportError:  # pragma: no cover - exercised by the CLI
    from common import REPO_ROOT

from danco import DANCo


BASE = {"N": 2500, "D": 400, "k": 10, "maxDim": 100}
SWEEPS = {
    "N": (625, 1250, 2500, 5000, 10000),
    "D": (100, 1000, 10000, 100000),
    "k": (5, 10, 15, 20),
    "maxDim": (50, 100, 200, 400),
}
STAGES = (
    "data_side",
    "spline_evaluate",
    "full_mc_evaluate",
    "cached_mc_evaluate",
)
STAGE_LABELS = {
    "data_side": "observed-data statistics only",
    "spline_evaluate": "complete evaluation with precomputed spline references",
    "full_mc_evaluate": "complete evaluation with newly generated Monte Carlo references",
    "cached_mc_evaluate": "complete evaluation with reused Monte Carlo references",
}
DEFAULT_STAGES = ("data_side",)
DEFAULT_SWEEPS = ("N", "D", "k")
OUT = REPO_ROOT / "results" / "runtime" / "runtime_data_side.csv"


def stage_label(stage):
    """Human-readable label that states the timed workload accurately."""
    try:
        return STAGE_LABELS[stage]
    except KeyError as exc:
        raise ValueError(f"unknown stage {stage!r}") from exc


def effective_config(sweep, value):
    """Return a benchmark cell without a silently truncated candidate range."""
    if sweep not in SWEEPS:
        raise ValueError(f"unknown sweep {sweep!r}")
    config = dict(BASE)
    config[sweep] = int(value)
    if sweep == "maxDim":
        # The packaged spline surface is trained for N in [450, 700]; use the
        # CNN protocol's N=500 so spline and Monte-Carlo stages are comparable.
        config["N"] = 500
        config["D"] = max(config["D"], config["maxDim"])
    validate_config(config)
    return config


def validate_config(config):
    required = {"N", "D", "k", "maxDim"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing runtime configuration fields: {sorted(missing)}")
    if config["N"] <= config["k"] + 1:
        raise ValueError("N must exceed k + 1")
    if config["D"] < 2 or config["maxDim"] < 1:
        raise ValueError("D >= 2 and maxDim >= 1 are required")
    if config["maxDim"] > config["D"]:
        raise ValueError(
            "candidate dimensions would be silently truncated: "
            f"maxDim={config['maxDim']} exceeds D={config['D']}"
        )


def _rss_mb():
    scale = 1024.0 if sys.platform != "darwin" else 1024.0 ** 2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def _cuda_peak_mb(backend):
    if backend != "gpu":
        return float("nan")
    import torch

    return torch.cuda.max_memory_allocated() / 1024.0 ** 2


def _measure_worker(payload):
    config = payload["config"]
    validate_config(config)
    stage = payload["stage"]
    backend = payload["backend"]
    dtype = payload["dtype"]
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    stage_label(stage)

    generator = np.random.default_rng(payload["seed"])
    np_dtype = np.float32 if dtype == "float32" else np.float64
    X = generator.standard_normal((config["N"], config["D"])).astype(
        np_dtype, copy=False,
    )
    backend_options = {"dtype": dtype} if backend == "gpu" else None
    version = "DANCo_fast" if stage == "spline_evaluate" else "DANCo"
    estimator = DANCo(
        N=config["N"],
        k=config["k"],
        version=version,
        random_state=payload["seed"],
        fractal=False,
        verbose=False,
        backend=backend,
        backend_options=backend_options,
        calibration_seeding="derived",
    )

    # Compile the angle kernel in the same process so measured repeats exclude
    # one-off JIT latency.  A warm-up in the driver cannot do this because each
    # measurement intentionally has its own process for an isolated RSS peak.
    warm_n = min(config["N"], max(config["k"] + 2, 32))
    estimator.danco_statistic(X[:warm_n], n_jobs=payload["n_jobs"])

    if backend == "gpu":
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    if stage == "cached_mc_evaluate":
        estimator.evaluate_all(X, maxDim=config["maxDim"],
                               n_jobs=payload["n_jobs"])

    started = time.perf_counter()
    if stage == "data_side":
        estimator.danco_statistic(X, n_jobs=payload["n_jobs"])
    else:
        # Complete evaluation = every step of Algorithm S1, i.e. both angular
        # objectives from one pass.
        estimator.evaluate_all(X, maxDim=config["maxDim"],
                               n_jobs=payload["n_jobs"])
    if backend == "gpu":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "rss_mb": _rss_mb(),
        "cuda_peak_mb": _cuda_peak_mb(backend),
    }


def _run_subprocess(payload):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json",
        json.dumps(payload, separators=(",", ":")),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "runtime worker failed:\n" + (completed.stderr or completed.stdout)
        )
    return json.loads(completed.stdout)


def _parse_csv(value, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _has_cuda():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sweeps", default=",".join(DEFAULT_SWEEPS))
    parser.add_argument("--values", default=None)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_json:
        print(json.dumps(_measure_worker(json.loads(args.worker_json))))
        return 0
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    load = os.getloadavg()[0]
    if load > args.max_load and not args.force:
        raise SystemExit(
            f"load average {load:.1f} > {args.max_load}; re-run on an idle "
            "machine or pass --force"
        )

    sweeps = _parse_csv(args.sweeps)
    stages = _parse_csv(args.stages)
    unknown_sweeps = set(sweeps) - set(SWEEPS)
    unknown_stages = set(stages) - set(STAGES)
    if unknown_sweeps or unknown_stages:
        parser.error(
            f"unknown sweeps={sorted(unknown_sweeps)} "
            f"or stages={sorted(unknown_stages)}"
        )
    if args.values and len(sweeps) != 1:
        parser.error("--values requires exactly one selected sweep")

    has_cuda = _has_cuda()
    rows = []
    for sweep in sweeps:
        values = _parse_csv(args.values, int) if args.values else SWEEPS[sweep]
        for value in values:
            config = effective_config(sweep, value)
            for stage in stages:
                backends = ("cpu", "gpu") if has_cuda else ("cpu",)
                for backend in backends:
                    base_payload = {
                        "config": config,
                        "stage": stage,
                        "backend": backend,
                        "dtype": args.dtype,
                        "n_jobs": args.n_jobs,
                    }
                    _run_subprocess({**base_payload, "seed": 9_000_000})
                    cell = []
                    for repeat in range(args.repeats):
                        measured = _run_subprocess({
                            **base_payload,
                            "seed": 9_000_001 + repeat,
                        })
                        row = {
                            "sweep": sweep,
                            **config,
                            "effective_max_dim": config["maxDim"],
                            "backend": backend,
                            "stage": stage,
                            "stage_label": stage_label(stage),
                            "dtype": args.dtype,
                            "seconds": measured["seconds"],
                            "rss_mb": measured["rss_mb"],
                            "cuda_peak_mb": measured["cuda_peak_mb"],
                            "repeat": repeat,
                        }
                        rows.append(row)
                        cell.append(row)
                    median = np.median([row["seconds"] for row in cell])
                    print(
                        f"{sweep}={value} {backend} {stage}: "
                        f"median {median:.3f}s",
                        flush=True,
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
