"""Shared plumbing for the reproduction scripts.

Two things live here because getting any of them subtly wrong silently
invalidates a table:

**The seed protocol.**  For each seed, *one* estimator per variant is built
and walks the whole dataset list.  The calibration cache only
extends: candidate m always consumes the m-th draw of the estimator's RNG, so a
reference replicate is a deterministic function of (random_state, N) and does
not depend on which datasets were evaluated first.  :func:`seeded_estimators`
and :func:`evaluate_sequence` encode the protocol; the seed constants below
are kept as named constants so a change is visible in a diff rather than
buried in a loop.

**Both angle columns from one pass.**  Full and Profiled share every
expensive computation and differ only in the angular KL, so one
``evaluate_all`` fills both columns (``stats['de_standard']`` and
``stats['de_nustrip']``).  Running two estimator objects instead gives the
same numbers, because their calibration caches grow identically -- same seed,
same dhat, same order.  Gride does *not* come free: a different distance model
means a different ``dhat``, hence a different calibration sequence.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from danco import DANCo, two_nn_legacy  # noqa: E402
from paper_methods import paper_twonn  # noqa: E402

# Every generator (tables, figures, graphical abstract, reference-range row)
# writes under this directory unless ``--paper-dir`` points it at a manuscript
# tree.  The directory is not tracked.
GENERATED_DIR = REPO_ROOT / "results" / "generated"


def add_paper_dir_argument(parser):
    """Add ``--paper-dir`` to an argparse parser (shared by the generators)."""
    parser.add_argument(
        "--paper-dir", type=Path, default=None, metavar="DIR",
        help="manuscript directory: the tables and figures that the paper "
             "\\input's or \\includegraphics's are written to DIR/tables and "
             "DIR/figures instead of results/generated/{tables,figures}; "
             "other generated outputs always go to results/generated/",
    )
    return parser


def artifact_dirs(paper_dir, kind):
    """Return ``(tex_dir, generated_dir)`` for ``kind`` in {tables, figures}.

    ``tex_dir`` receives the artifacts the manuscript includes; it is
    ``<paper_dir>/<kind>`` when ``paper_dir`` is given and
    ``results/generated/<kind>`` otherwise.  ``generated_dir`` is always
    ``results/generated/<kind>`` and receives everything else.
    """
    if kind not in ("tables", "figures"):
        raise ValueError(f"kind must be 'tables' or 'figures', got {kind!r}")
    generated = GENERATED_DIR / kind
    tex = (Path(paper_dir) / kind) if paper_dir is not None else generated
    return tex, generated

__all__ = [
    "REPO_ROOT", "GENERATED_DIR", "add_paper_dir_argument", "artifact_dirs",
    "MANUSCRIPT_FONT", "use_manuscript_font",
    "BENCHMARK_SEED", "EST_SEED_BASE", "HIGHID_BENCH_SEED",
    "HIGHID_SEED_BASE", "IMAGE_SEED_BASE", "N_SEEDS",
    "ESTIMATORS", "add_common_args", "make_parser", "resolve_estimators",
    "apply_subset", "seeded_estimators", "evaluate_sequence", "two_nn_row",
    "two_nn_reference_row",
    "save_table", "save_raw", "mpe",
]

# ---------------------------------------------------------------------------
# manuscript typography
# ---------------------------------------------------------------------------
# Pattern Recognition allows one typeface, Times New Roman (paper/pr-fonts.tex
# sets it for the text and the TikZ figure).  Every matplotlib figure is set
# in the same face so the figures read as part of the page rather than as
# pasted-in sans-serif panels.  A missing font is an error, never a
# substitution: a silently substituted face would ship an inconsistent PDF.
MANUSCRIPT_FONT = "Times New Roman"
_FONT_DIRS = (
    Path.home() / ".local/share/fonts/TimesNewRoman",
    Path("/usr/share/fonts/truetype/msttcorefonts"),
)


def use_manuscript_font():
    """Register Times New Roman with matplotlib and select it for text and math.

    Returns the family name so callers can pass it explicitly where needed.
    Type 42 (TrueType) embedding keeps the figure text searchable and passes
    the publisher's PDF checks; ``mathtext.fontset='custom'`` maps every math
    style onto the same family so ``$\\hat\\nu$`` and ``$\\sigma_S$`` do not fall
    back to DejaVu.
    """
    import matplotlib
    from matplotlib import font_manager as fm

    for directory in _FONT_DIRS:
        if directory.is_dir():
            for path in directory.glob("*.ttf"):
                fm.fontManager.addfont(str(path))
    fm.findfont(MANUSCRIPT_FONT, fallback_to_default=False)
    font = MANUSCRIPT_FONT
    matplotlib.rcParams.update({
        "font.family": font,
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.it": font + ":italic",
        "mathtext.bf": font + ":bold",
        "mathtext.bfit": font + ":bold:italic",
        "mathtext.sf": font,
        "mathtext.tt": font,
        "mathtext.cal": font + ":italic",
        "mathtext.fallback": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return font


# ---------------------------------------------------------------------------
# the seed protocol
# ---------------------------------------------------------------------------
BENCHMARK_SEED = 71      # BenchmarkManifolds(random_state=71)
HIGHID_BENCH_SEED = 0    # BenchmarkManifolds(random_state=0), high-ID suite
EST_SEED_BASE = 35       # estimator random_state = 35 + seed (benchmark)
HIGHID_SEED_BASE = 78    # estimator random_state = 78 + seed (high-ID suite)
IMAGE_SEED_BASE = 43     # estimator random_state = 43 + seed (image classes)
N_SEEDS = 5
N_BENCHMARK_MANIFOLDS = 24  # the MPE denominator (size of the benchmark suite)

# estimator key -> (version, output column, needs its own estimator object)
ESTIMATORS = {
    "danco": ("DANCo", "DANCo", True),
    "acc": ("DANCo", "accDANCo", False),        # same run as 'danco'
    "nustrip": ("DANCo", "DANCo_nustrip", False),  # same run as 'danco'
    "gride": ("DANCo_gride", "DANCo_Gride", True),
    "gride_nus": ("DANCo_gride", "DANCo_Gride_nustrip", False),  # same run as 'gride'
    "fast": ("DANCo_fast", "DANCo_fast", True),
    "twonn": (None, "2NN_ref", False),
    "twonn_legacy": (None, "2NN", False),
}
DEFAULT_ESTIMATORS = "danco,acc,gride,nustrip,twonn"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_common_args(parser, default_output, default_estimators=None):
    parser.add_argument("--output", type=Path, default=Path(default_output),
                        help="output directory (default: %(default)s)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds to average (default: %(default)s)")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="threads for the CPU neighbour search")
    parser.add_argument(
        "--estimators", default=default_estimators or DEFAULT_ESTIMATORS,
        help=f"comma-separated subset of {sorted(ESTIMATORS)} "
             "(default: %(default)s)")
    parser.add_argument(
        "--subset", default=None,
        help="comma-separated dataset names to run instead of all (smoke test)")
    return parser


def resolve_estimators(spec):
    """Parse ``--estimators`` into an ordered, de-duplicated list of keys."""
    keys = []
    for raw in spec.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in ESTIMATORS:
            raise SystemExit(
                f"unknown estimator {key!r}; choose from {sorted(ESTIMATORS)}"
            )
        if key not in keys:
            keys.append(key)
    if not keys:
        raise SystemExit("--estimators selected nothing")
    return keys


def apply_subset(datasets, subset):
    """Filter an ordered ``{name: data}`` mapping, preserving order."""
    if not subset:
        return datasets, False
    wanted = [s.strip() for s in subset.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in datasets]
    if unknown:
        raise SystemExit(
            f"--subset names not in this dataset: {unknown}\n"
            f"available: {list(datasets)}"
        )
    return {name: datasets[name] for name in datasets if name in wanted}, True


# ---------------------------------------------------------------------------
# running the protocol
# ---------------------------------------------------------------------------
def seeded_estimators(keys, seed_index, seed_base, backend="cpu", **kwargs):
    """One estimator object per *independent* variant, for one seed.

    Returns ``{key: estimator}`` covering only the keys that need their own
    object; ``acc``/``nustrip`` read extra columns off the ``danco`` run and
    ``twonn`` and ``twonn_legacy`` are not DANCo estimator objects.
    """
    estimators = {}
    for key in keys:
        version, _, standalone = ESTIMATORS[key]
        if not standalone:
            continue
        estimators[key] = DANCo(
            random_state=seed_base + seed_index, version=version,
            backend=backend, **kwargs,
        )
    return estimators


def evaluate_sequence(estimator, datasets, maxDim=100, n_jobs=-1):
    """Walk ``datasets`` in order with a single estimator.

    The calibration cache grows as it goes and is never reset.  Candidate m
    always consumes the m-th draw of the estimator's RNG, so the cache contents
    for a given (random_state, N) do not depend on the evaluation order.
    Returns ``{name: (de_standard, de_nustrip, stats)}``.
    """
    out = {}
    for name, data in datasets.items():
        stats = estimator.evaluate_all(data, maxDim=maxDim, n_jobs=n_jobs)
        out[name] = (stats["de_standard"], stats["de_nustrip"], stats)
    return out


def two_nn_row(datasets, n_jobs=-1):
    """TwoNN for every dataset -- deterministic, so no seed loop."""
    return {name: two_nn_legacy(data, n_jobs=n_jobs) for name, data in datasets.items()}


def two_nn_reference(X):
    """Reference TWO-NN (Facco et al., 2017): the discarded-tail linear fit.

    Delegates to scikit-dimension's ``TwoNN`` with its defaults
    (``discard_fraction=0.1``), the same estimator the baseline table uses,
    so every paper-facing TWO-NN value comes from one implementation.
    Deterministic; computed on float64 via CPU neighbors.
    """
    return paper_twonn(X)


def two_nn_reference_row(datasets):
    """Reference TWO-NN for every dataset."""
    return {name: two_nn_reference(data) for name, data in datasets.items()}


def mpe(estimates, truth, denominator=N_BENCHMARK_MANIFOLDS):
    """Mean percentage error over the 24-manifold benchmark.

    The denominator is fixed at 24 even when fewer manifolds were run, so that
    the number stays comparable with the full-suite tables.  Under ``--subset``
    the value is therefore not meaningful.
    """
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return 100.0 / denominator * np.sum(np.abs(estimates - truth) / truth)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="index")
    print(f"wrote {path}")
    return path


def save_raw(records, path):
    """Per-seed values, long format: one row per (seed, dataset, estimator)."""
    path = Path(path)
    raw_path = path.parent / "raw" / f"{path.stem}_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(raw_path, index=False)
    print(f"wrote {raw_path}")
    return raw_path


def make_parser(description, default_output, default_estimators=None):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_common_args(parser, default_output, default_estimators)
