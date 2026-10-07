from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.make_tables import (
    cnn_peak_summary,
    combined_baseline_cells,
    estimator_summary,
    primary_corrected_rows,
)
from experiments.run_runtime import STAGE_LABELS, _measure_worker, stage_label
from experiments.export_reference_envelope import summarize_reference_envelope


def test_primary_corrected_rows_require_twenty_complete_replicates():
    rows = []
    for rep in range(20):
        for manifold in ("M1", "M2"):
            rows.append({
                "manifold": manifold,
                "true_d": 10.0,
                "impl": "ours",
                "rep": rep,
                "dhat": 10.0,
                "seconds": 0.1,
                "status": "ok",
            })
        rows.append({
            "manifold": "__calibration__",
            "true_d": np.nan,
            "impl": "cal_ours",
            "rep": rep,
            "dhat": np.nan,
            "seconds": 1.0,
            "status": "cal",
        })
    frame = pd.DataFrame(rows)

    selected = primary_corrected_rows(frame, expected_reps=20,
                                      expected_manifolds=2)

    assert len(selected) == 40
    assert set(selected["rep"]) == set(range(20))
    with pytest.raises(ValueError, match="20 complete replicates"):
        primary_corrected_rows(frame[frame["rep"] < 19], expected_reps=20,
                               expected_manifolds=2)


def test_estimator_summary_counts_failures_as_errors_but_not_mpe_values():
    frame = pd.DataFrame([
        {"manifold": "M1", "true_d": 10.0, "dhat": 10.0,
         "seconds": 1.0},
        {"manifold": "M1", "true_d": 10.0, "dhat": np.nan,
         "seconds": 2.0},
        {"manifold": "M2", "true_d": 20.0, "dhat": 24.0,
         "seconds": 3.0},
        {"manifold": "M2", "true_d": 20.0, "dhat": 20.0,
         "seconds": 4.0},
    ])

    summary = estimator_summary(frame)

    assert summary["n_fail"] == 1
    assert summary["error_rate"] == pytest.approx(0.5)
    assert summary["mpe"] == pytest.approx(5.0)
    assert summary["median_seconds"] == pytest.approx(2.5)


def test_full_baseline_cells_include_corrected_danco_as_tenth_estimator():
    baselines = pd.DataFrame([
        {"estimator": "MLE", "manifold": "M1", "true_d": 10.0,
         "dhat": 9.0},
        {"estimator": "MLE", "manifold": "M1", "true_d": 10.0,
         "dhat": 11.0},
    ])
    corrected = pd.DataFrame([
        {"manifold": "M1", "true_d": 10.0, "dhat": 10.0},
        {"manifold": "M1", "true_d": 10.0, "dhat": 12.0},
    ])

    cells = combined_baseline_cells(baselines, corrected)

    assert set(cells["estimator"]) == {"DANCo_corrected", "MLE"}
    danco = cells[cells["estimator"] == "DANCo_corrected"].iloc[0]
    assert danco["mean"] == pytest.approx(11.0)


def test_cnn_peak_summary_requires_peak_and_largest_gap_to_coincide():
    frame = pd.DataFrame([
        {"arch": "net", "layer_idx": 0, "layer_name": "input", "depth": 0,
         "id_nustrip": 20, "id_orig_diagnostic": 18, "mu_nu": 1.1,
         "delta_orig_minus_nustrip": -2},
        {"arch": "net", "layer_idx": 1, "layer_name": "block", "depth": 2,
         "id_nustrip": 80, "id_orig_diagnostic": 30, "mu_nu": 0.8,
         "delta_orig_minus_nustrip": -50},
        {"arch": "net", "layer_idx": 2, "layer_name": "head", "depth": 4,
         "id_nustrip": 25, "id_orig_diagnostic": 22, "mu_nu": 1.0,
         "delta_orig_minus_nustrip": -3},
    ])

    summary = cnn_peak_summary(frame)

    assert summary.iloc[0]["layer_name"] == "block"
    assert summary.iloc[0]["relative_depth"] == pytest.approx(0.5)
    assert summary.iloc[0]["profiled"] == pytest.approx(80)
    assert summary.iloc[0]["full_minus_profiled"] == pytest.approx(-50)

    mismatch = frame.copy()
    mismatch.loc[mismatch["layer_idx"] == 0,
                 "delta_orig_minus_nustrip"] = -60
    with pytest.raises(ValueError, match="largest Full--Profiled gap"):
        cnn_peak_summary(mismatch)


def test_reference_envelope_uses_all_verified_rows():
    frame = pd.DataFrame({
        "N": [500, 500, 500, 500, 600, 600, 600, 600],
        "seed": [43, 43, 44, 44, 43, 43, 44, 44],
        "candidate_dim": [1, 2, 1, 2, 1, 2, 1, 2],
        "nu_ref": [0.0, 1.45, 1.10, 1.48, 1.08, 1.47, 1.09, 1.51],
        "tau_ref": [0.0, 3.0, 2.1, 3.1, 2.5, 3.5, 2.6, 3.6],
        "dhat_ref": [1.0, 6.0, 6.0, 7.0, 6.0, 7.0, 6.0, 7.0],
    })

    summary = summarize_reference_envelope(frame, expected_ns=(500, 600),
                                           expected_seeds=(43, 44),
                                           max_dim=2)

    assert summary["n_rows"] == 8
    assert summary["n_valid_angular_rows"] == 7
    assert summary["nu_min"] == pytest.approx(1.08)
    assert summary["nu_max"] == pytest.approx(1.51)
    with pytest.raises(ValueError, match="complete reference grid"):
        summarize_reference_envelope(frame.iloc[:-1], expected_ns=(500, 600),
                                     expected_seeds=(43, 44), max_dim=2)


def test_gsm_latent_descaling_is_complete_and_paired():
    path = Path("results/diagnostics/fusion_ablation_gsm.csv")
    frame = pd.read_csv(path)
    signatures = ["distance_model", "angle_mode", "objective"]
    cells = ["scale_sigma", "intervention", "rep"]

    counts = frame.groupby(cells).size()
    assert set(counts) == {8}
    assert set(frame.loc[frame["intervention"] == "raw", "scale_sigma"]) == {
        0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
    }
    assert set(frame["ambient_d"]) == {100}
    assert set(frame["max_dim"]) == {100}
    latent = frame[frame["intervention"] == "latent_descaled"]
    assert set(latent["scale_sigma"]) == {0.25}
    assert set(latent["rep"]) == set(range(30))
    assert latent.groupby("rep")[signatures].size().eq(8).all()

    keys = ["rep", "data_seed", "calibration_seed"]
    raw_keys = frame[
        (frame["intervention"] == "raw") & (frame["scale_sigma"] == 0.25)
    ][keys].drop_duplicates().sort_values("rep").reset_index(drop=True)
    latent_keys = latent[keys].drop_duplicates().sort_values("rep").reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(raw_keys, latent_keys)


def test_runtime_stage_labels_name_complete_evaluations():
    assert stage_label("data_side") == "observed-data statistics only"
    assert stage_label("spline_evaluate") == (
        "complete evaluation with precomputed spline references"
    )
    assert stage_label("full_mc_evaluate") == (
        "complete evaluation with newly generated Monte Carlo references"
    )
    assert stage_label("cached_mc_evaluate") == (
        "complete evaluation with reused Monte Carlo references"
    )
    assert set(STAGE_LABELS) == {
        "data_side", "spline_evaluate", "full_mc_evaluate",
        "cached_mc_evaluate",
    }


def test_runtime_worker_permits_matched_gpu_complete_stages(monkeypatch):
    # Stop after validation: reaching the sentinel proves GPU complete stages
    # are no longer rejected by the stage/backend contract.
    monkeypatch.setattr(np.random, "default_rng",
                        lambda seed: (_ for _ in ()).throw(RuntimeError("sentinel")))
    payload = {
        "config": {"N": 500, "D": 400, "k": 10, "maxDim": 100},
        "stage": "spline_evaluate",
        "backend": "gpu",
        "dtype": "float64",
        "seed": 1,
        "n_jobs": 1,
    }

    with pytest.raises(RuntimeError, match="sentinel"):
        _measure_worker(payload)


def test_nu_toy_description_does_not_claim_a_floor_or_proposition_three():
    text = Path("experiments/run_nu_toy_sweep.py").read_text()

    assert "Proposition 3" not in text
    assert "pi/3 floor" not in text
    assert "conditional mechanism" in text
