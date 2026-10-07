"""Canonical estimator contracts used by manuscript tables and figures."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodContract:
    display_name: str
    implementation: str
    preprocessing: str
    neighborhood: str
    calibration: str
    refinement: str
    backend: str
    intentional_exception: str = ""


PAPER_TWONN_ID = "twonn_facco_skdim_0.3.7_discard0.1_exactdedup"
PAPER_MLE_ID = "mle_levina_bickel_skdim_0.3.7_k10_exactdedup"
CNN_CANONICAL_GLOB = "cnn_layer_ids_rep[0-2].csv"
MIND_FULL_MC_ID = "mind_full_mc_fractal"
MIND_PROFILED_MC_ID = "mind_profiled_mc_fractal"
MIND_FULL_SPLINE_ID = "mind_full_spline_integer_gpu"
MIND_PROFILED_SPLINE_ID = "mind_profiled_spline_integer_gpu"
GRIDE_FULL_MC_ID = "gride_full_mc_fractal"
GRIDE_PROFILED_MC_ID = "gride_profiled_mc_fractal"
GRIDE_FULL_SPLINE_ID = "gride_full_spline_integer_gpu"
GRIDE_PROFILED_SPLINE_ID = "gride_profiled_spline_integer_gpu"
MIND_DISTANCE_ID = "mind_distance_only_mc_fractal"
GRIDE_DISTANCE_ID = "gride_distance_only_mc_fractal"
FULL_ANGLE_ID = "full_angle_only_mc_fractal"
PROFILED_ANGLE_ID = "profiled_angle_only_mc_fractal"
LPCA_ID = "lpca_skdim_0.3.7_default"
ESS_ID = "ess_skdim_0.3.7_default"
MADA_ID = "mada_skdim_0.3.7_default"
TLE_ID = "tle_skdim_0.3.7_default"
CORRINT_ID = "corrint_skdim_0.3.7_k10_k20"
MIND_ML_ID = "mind_ml_skdim_0.3.7_k10_d100"
FISHERS_ID = "fishers_skdim_0.3.7_default"

LEGACY_METHOD_IDS = frozenset({"twonn_legacy_untrimmed_full_sample_mle"})


def _contract(display_name, implementation, *, preprocessing="raw rows",
              neighborhood="k=10", calibration="none", refinement="native",
              backend="cpu float64", intentional_exception=""):
    return MethodContract(
        display_name=display_name,
        implementation=implementation,
        preprocessing=preprocessing,
        neighborhood=neighborhood,
        calibration=calibration,
        refinement=refinement,
        backend=backend,
        intentional_exception=intentional_exception,
    )


METHOD_CONTRACTS = {
    PAPER_TWONN_ID: _contract(
        "TWO-NN", "skdim.id.TwoNN(discard_fraction=0.1)",
        preprocessing="stable exact-row deduplication; float64",
        neighborhood="first/second neighbor ratio",
    ),
    PAPER_MLE_ID: _contract(
        "MLE", "skdim.id.MLE(K=10).dimension_",
        preprocessing="stable exact-row deduplication; float64",
    ),
    MIND_FULL_MC_ID: _contract(
        "MiND--Full", "danco.DANCo(version='DANCo').evaluate_all(X)['de_standard']",
        calibration="Monte Carlo hyperballs", refinement="fractional",
    ),
    MIND_PROFILED_MC_ID: _contract(
        "MiND--Profiled", "danco.DANCo(version='DANCo').evaluate_all(X)['de_nustrip']",
        calibration="Monte Carlo hyperballs", refinement="fractional",
    ),
    MIND_FULL_SPLINE_ID: _contract(
        "MiND--Full", "danco.DANCo(version='DANCo_fast').evaluate_all(X)['de_standard']",
        preprocessing="shared stable exact-row deduplication",
        calibration="MiND spline surface", refinement="integer",
        backend="gpu float32; duplicate filter disabled",
        intentional_exception="archived CNN FastDANCo-style calibration",
    ),
    MIND_PROFILED_SPLINE_ID: _contract(
        "MiND--Profiled", "danco.DANCo(version='DANCo_fast').evaluate_all(X)['de_nustrip']",
        preprocessing="shared stable exact-row deduplication",
        calibration="MiND spline surface", refinement="integer",
        backend="gpu float32; duplicate filter disabled",
        intentional_exception="archived CNN FastDANCo-style calibration",
    ),
    GRIDE_FULL_MC_ID: _contract(
        "Gride--Full", "danco.DANCo(version='DANCo_gride').evaluate_all(X)['de_standard']",
        neighborhood="(k1,k2)=(5,10)", calibration="Monte Carlo hyperballs",
        refinement="fractional",
    ),
    GRIDE_PROFILED_MC_ID: _contract(
        "Gride--Profiled", "danco.DANCo(version='DANCo_gride').evaluate_all(X)['de_nustrip']",
        neighborhood="(k1,k2)=(5,10)", calibration="Monte Carlo hyperballs",
        refinement="fractional",
    ),
    GRIDE_FULL_SPLINE_ID: _contract(
        "Gride--Full", "danco.DANCo(version='DANCo_gride_fast').evaluate_all(X)['de_standard']",
        preprocessing="shared stable exact-row deduplication",
        neighborhood="(k1,k2)=(5,10)", calibration="Gride spline surface",
        refinement="integer", backend="gpu float32; duplicate filter disabled",
        intentional_exception="CNN FastDANCo-style calibration",
    ),
    GRIDE_PROFILED_SPLINE_ID: _contract(
        "Gride--Profiled",
        "danco.DANCo(version='DANCo_gride_fast').evaluate_all(X)['de_nustrip']",
        preprocessing="shared stable exact-row deduplication",
        neighborhood="(k1,k2)=(5,10)", calibration="Gride spline surface",
        refinement="integer", backend="gpu float32; duplicate filter disabled",
        intentional_exception="CNN FastDANCo-style calibration",
    ),
    MIND_DISTANCE_ID: _contract(
        "MiND distance-only", "MiND distance KL argmin",
        calibration="Monte Carlo hyperballs", refinement="fractional",
    ),
    GRIDE_DISTANCE_ID: _contract(
        "Gride distance-only", "Gride distance KL argmin",
        neighborhood="(k1,k2)=(5,10)", calibration="Monte Carlo hyperballs",
        refinement="fractional",
    ),
    FULL_ANGLE_ID: _contract(
        "Full angle-only", "full von Mises KL argmin",
        calibration="Monte Carlo hyperballs", refinement="fractional",
    ),
    PROFILED_ANGLE_ID: _contract(
        "Profiled angle-only", "mean-direction-profiled von Mises KL argmin",
        calibration="Monte Carlo hyperballs", refinement="fractional",
    ),
    LPCA_ID: _contract("lPCA", "skdim.id.lPCA(verbose=False)"),
    ESS_ID: _contract("ESS", "skdim.id.ESS()"),
    MADA_ID: _contract("MADA", "skdim.id.MADA()"),
    TLE_ID: _contract("TLE", "skdim.id.TLE()"),
    CORRINT_ID: _contract(
        "CorrInt", "skdim.id.CorrInt(k1=10,k2=20)",
        neighborhood="k1=10, k2=20",
    ),
    MIND_ML_ID: _contract(
        "MiND--ML", "skdim.id.MiND_ML(k=10,D=100)",
    ),
    FISHERS_ID: _contract("FisherS", "skdim.id.FisherS()"),
}


_COMPONENT_METHODS = (
    MIND_DISTANCE_ID, GRIDE_DISTANCE_ID, FULL_ANGLE_ID, PROFILED_ANGLE_ID,
    MIND_FULL_MC_ID, MIND_PROFILED_MC_ID, GRIDE_FULL_MC_ID,
    GRIDE_PROFILED_MC_ID,
)

_BASELINE_METHODS = (
    MIND_FULL_MC_ID, PAPER_TWONN_ID, PAPER_MLE_ID, LPCA_ID, ESS_ID, MADA_ID,
    TLE_ID, CORRINT_ID, MIND_ML_ID, FISHERS_ID,
)

PAPER_ARTIFACT_METHODS = {
    "table_baselines_main": _BASELINE_METHODS,
    "table_baselines_full": _BASELINE_METHODS,
    "fig_error_vs_d": (
        MIND_FULL_MC_ID, PAPER_TWONN_ID, PAPER_MLE_ID, ESS_ID, MIND_ML_ID,
    ),
    "table_sensitivity": (
        MIND_FULL_MC_ID, GRIDE_FULL_MC_ID, MIND_PROFILED_MC_ID,
        GRIDE_PROFILED_MC_ID, PAPER_TWONN_ID,
    ),
    "table_images": (
        MIND_FULL_MC_ID, GRIDE_FULL_MC_ID, MIND_PROFILED_MC_ID,
        GRIDE_PROFILED_MC_ID, PAPER_TWONN_ID,
    ),
    "fig_noise_curves": (
        MIND_FULL_MC_ID, GRIDE_FULL_MC_ID, MIND_PROFILED_MC_ID,
        GRIDE_PROFILED_MC_ID,
    ),
    "fig_fusion_ablation_main": _COMPONENT_METHODS,
    "fig_kl_diagnosis": _COMPONENT_METHODS,
    "fig_cnn_paired": (
        GRIDE_PROFILED_SPLINE_ID, PAPER_TWONN_ID, PAPER_MLE_ID,
    ),
    "table_cnn_peak_summary": (
        GRIDE_FULL_SPLINE_ID, GRIDE_PROFILED_SPLINE_ID,
    ),
}

SPLINE_METHOD_IDS = {
    "DANCo_fast": (MIND_FULL_SPLINE_ID, MIND_PROFILED_SPLINE_ID),
    "DANCo_nustrip_fast": (MIND_FULL_SPLINE_ID, MIND_PROFILED_SPLINE_ID),
    "DANCo_gride_fast": (GRIDE_FULL_SPLINE_ID, GRIDE_PROFILED_SPLINE_ID),
    "DANCo_gride_nustrip_fast": (GRIDE_FULL_SPLINE_ID, GRIDE_PROFILED_SPLINE_ID),
}


def spline_method_ids(version):
    try:
        return SPLINE_METHOD_IDS[version]
    except KeyError as exc:
        raise ValueError(f"no paper spline method IDs for version {version!r}") from exc


def method_contract_rows():
    """Flatten the artifact registry for the checked-in audit ledger."""
    rows = []
    for artifact, method_ids in PAPER_ARTIFACT_METHODS.items():
        for method_id in method_ids:
            contract = METHOD_CONTRACTS[method_id]
            rows.append({
                "artifact": artifact,
                "method_id": method_id,
                "display_name": contract.display_name,
                "implementation": contract.implementation,
                "preprocessing": contract.preprocessing,
                "neighborhood": contract.neighborhood,
                "calibration": contract.calibration,
                "refinement": contract.refinement,
                "backend": contract.backend,
                "status": ("INTENTIONAL_EXCEPTION"
                           if contract.intentional_exception else "UNIFIED"),
                "intentional_exception": contract.intentional_exception,
            })
    return rows


def _cnn_rep(path):
    match = re.search(r"rep(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer CNN replicate from {path}")
    return int(match.group(1))


def load_canonical_cnn_results(cnn_dir):
    """Load the complete three-replicate CNN result set or fail closed."""
    cnn_dir = Path(cnn_dir)
    paths = sorted(cnn_dir.glob(CNN_CANONICAL_GLOB))
    if len(paths) != 3:
        raise ValueError(
            f"expected three canonical CNN CSVs, found {len(paths)} in {cnn_dir}"
        )
    required = {
        "arch", "category_tag", "subset_rep", "layer_idx", "n_raw",
        "n_unique", "n_duplicates", "id_orig_diagnostic", "id_nustrip",
        "twonn_ref", "id_mle", "method_id_full", "method_id_profiled",
        "method_id_twonn", "method_id_mle", "backend_dtype",
        "backend_dup_threshold",
    }
    expected_methods = {
        "method_id_full": GRIDE_FULL_SPLINE_ID,
        "method_id_profiled": GRIDE_PROFILED_SPLINE_ID,
        "method_id_twonn": PAPER_TWONN_ID,
        "method_id_mle": PAPER_MLE_ID,
    }
    frames = []
    for path in paths:
        rep = _cnn_rep(path)
        if rep not in (0, 1, 2):
            raise ValueError(f"unexpected CNN replicate {rep} in {path}")
        frame = pd.read_csv(path)
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} is missing canonical columns {sorted(missing)}")
        if len(frame) != 224:
            raise ValueError(f"{path} requires 224 rows, found {len(frame)}")
        keys = ["arch", "category_tag", "subset_rep", "layer_idx"]
        if frame.duplicated(keys).any():
            raise ValueError(f"{path} contains duplicate CNN result keys")
        if set(frame["subset_rep"].astype(int)) != {rep}:
            raise ValueError(f"{path} carries the wrong subset_rep")
        if not ((frame["n_raw"] == 500)
                & frame["n_unique"].between(450, 500)
                & (frame["n_duplicates"] == frame["n_raw"] - frame["n_unique"])
                ).all():
            raise ValueError(f"{path} violates the shared-dedup sample contract")
        if (set(frame["estimator"]) != {"gride"}
                or set(frame["backend_dtype"]) != {"float32"}
                or set(frame["backend_dup_threshold"]) != {"none"}
                or not np.isfinite(frame[
                    ["id_orig_diagnostic", "id_nustrip", "twonn_ref", "id_mle"]
                ].to_numpy(dtype=float)).all()):
            raise ValueError(f"{path} violates the canonical estimator/backend contract")
        for column, method_id in expected_methods.items():
            if set(frame[column]) != {method_id}:
                raise ValueError(f"{path} carries a wrong {column}")
        if {"twonn", "mle_n_degenerate"} & set(frame):
            raise ValueError(f"{path} carries a retired estimator column")

        sidecar = path.with_suffix(".summary.json")
        if not sidecar.exists():
            raise ValueError(f"missing CNN sidecar {sidecar}")
        summary = json.loads(sidecar.read_text())
        duplicate_threshold = summary.get(
            "backend_options", {}
        ).get("dup_threshold", 1)
        if (int(summary.get("subset_rep", -1)) != rep
                or len(summary.get("completed", [])) != 28
                or summary.get("failures")
                or summary.get("backend") != "gpu"
                or summary.get("estimator") != "gride"
                or summary.get("version") != "DANCo_gride_fast"
                or duplicate_threshold is not None
                or summary.get("software", {}).get("scikit-dimension") != "0.3.7"
                or summary.get("method_ids") != {
                    "full": GRIDE_FULL_SPLINE_ID,
                    "profiled": GRIDE_PROFILED_SPLINE_ID,
                    "twonn": PAPER_TWONN_ID,
                    "mle": PAPER_MLE_ID,
                }):
            raise ValueError(f"{sidecar} violates the canonical CNN run contract")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 672:
        raise ValueError(f"canonical CNN result requires 672 rows, found {len(result)}")
    return result


def stable_unique_rows(X):
    """Remove exact duplicate rows while retaining first-occurrence order."""
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D sample matrix, got shape {array.shape}")
    _, first = np.unique(array, axis=0, return_index=True)
    unique = np.ascontiguousarray(array[np.sort(first)])
    audit = {
        "n_raw": int(len(array)),
        "n_unique": int(len(unique)),
        "n_duplicates": int(len(array) - len(unique)),
    }
    return unique, audit


def _paper_rows(X, assume_unique):
    rows = np.asarray(X) if assume_unique else stable_unique_rows(X)[0]
    return np.ascontiguousarray(rows, dtype=np.float64)


def paper_twonn(X, *, assume_unique=False):
    """Facco TWO-NN contract used by every manuscript artifact."""
    import skdim.id as skid

    rows = _paper_rows(X, assume_unique)
    if len(rows) < 3:
        return float("nan")
    return float(skid.TwoNN().fit(rows).dimension_)


def paper_mle(X, *, k=10, assume_unique=False):
    """Levina--Bickel MLE contract used by baseline and CNN results."""
    import skdim.id as skid

    rows = _paper_rows(X, assume_unique)
    if len(rows) <= k:
        return float("nan")
    return float(skid.MLE(K=k).fit(rows).dimension_)


class PaperTwoNN:
    """Small sklearn-style adapter for the canonical paper TWO-NN."""

    def fit(self, X):
        self.dimension_ = paper_twonn(X)
        return self


class PaperMLE:
    """Small sklearn-style adapter for the canonical paper MLE."""

    def __init__(self, k=10):
        self.k = int(k)

    def fit(self, X):
        self.dimension_ = paper_mle(X, k=self.k)
        return self
