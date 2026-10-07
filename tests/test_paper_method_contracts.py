"""Canonical estimator contracts for every paper-facing artifact."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.paper_methods import (
    GRIDE_PROFILED_SPLINE_ID,
    LEGACY_METHOD_IDS,
    METHOD_CONTRACTS,
    PAPER_ARTIFACT_METHODS,
    PAPER_MLE_ID,
    PAPER_TWONN_ID,
    MIND_FULL_SPLINE_ID,
    MIND_PROFILED_SPLINE_ID,
    paper_mle,
    paper_twonn,
    stable_unique_rows,
    method_contract_rows,
    spline_method_ids,
)
from experiments.run_baselines import build_estimators, estimate


def test_stable_unique_rows_removes_exact_duplicates_and_preserves_order():
    X = np.array([
        [3.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0],
    ])

    unique, audit = stable_unique_rows(X)

    np.testing.assert_array_equal(
        unique,
        np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    )
    assert audit == {"n_raw": 5, "n_unique": 3, "n_duplicates": 2}


def test_paper_twonn_and_mle_use_the_same_exact_deduplicated_rows():
    rng = np.random.default_rng(7)
    clean = rng.normal(size=(120, 6))
    duplicated = np.vstack([clean, clean[:8]])

    twonn = paper_twonn(duplicated)
    mle = paper_mle(duplicated, k=10)

    assert np.isfinite(twonn) and twonn > 0
    assert np.isfinite(mle) and mle > 0
    assert twonn == pytest.approx(paper_twonn(clean), rel=1e-12)
    assert mle == pytest.approx(paper_mle(clean, k=10), rel=1e-12)


def test_baseline_twonn_and_mle_delegate_to_the_paper_contracts():
    rng = np.random.default_rng(11)
    clean = rng.normal(size=(120, 6))
    duplicated = np.vstack([clean, clean[:8]])

    estimators = build_estimators(["TwoNN", "MLE"])
    twonn, _, twonn_status = estimate(estimators["TwoNN"], duplicated)
    mle, _, mle_status = estimate(estimators["MLE"], duplicated)

    assert twonn_status == mle_status == "ok"
    assert twonn == pytest.approx(paper_twonn(clean), rel=1e-12)
    assert mle == pytest.approx(paper_mle(clean, k=10), rel=1e-12)


def test_every_manuscript_estimator_artifact_has_only_canonical_contracts():
    required = {
        "table_baselines_main",
        "table_baselines_full",
        "fig_error_vs_d",
        "table_sensitivity",
        "table_images",
        "fig_noise_curves",
        "fig_fusion_ablation_main",
        "fig_kl_diagnosis",
        "fig_cnn_paired",
        "table_cnn_peak_summary",
    }

    assert required <= set(PAPER_ARTIFACT_METHODS)
    for artifact, method_ids in PAPER_ARTIFACT_METHODS.items():
        assert method_ids, artifact
        assert set(method_ids) <= set(METHOD_CONTRACTS), artifact
        assert set(method_ids).isdisjoint(LEGACY_METHOD_IDS), artifact

    assert "table_highid" not in PAPER_ARTIFACT_METHODS
    assert "table_sens_k" not in PAPER_ARTIFACT_METHODS
    assert "table_sens_n" not in PAPER_ARTIFACT_METHODS
    assert PAPER_TWONN_ID in PAPER_ARTIFACT_METHODS["table_sensitivity"]
    assert PAPER_MLE_ID in PAPER_ARTIFACT_METHODS["fig_cnn_paired"]


def test_table_generators_never_select_legacy_twonn_columns():
    from experiments.make_tables import (
        IMAGE_EST_COLS,
        SENSITIVITY_EST_COLS,
    )

    selections = [
        IMAGE_EST_COLS,
        SENSITIVITY_EST_COLS,
    ]
    for columns in selections:
        assert "2NN_ref" in columns
        assert "2NN" not in columns


def test_runner_default_twonn_is_canonical_and_legacy_is_explicit():
    from experiments.common import DEFAULT_ESTIMATORS, ESTIMATORS

    assert ESTIMATORS["twonn"][1] == "2NN_ref"
    assert ESTIMATORS["twonn_legacy"][1] == "2NN"
    assert "twonn" in DEFAULT_ESTIMATORS.split(",")
    assert "twonn_legacy" not in DEFAULT_ESTIMATORS.split(",")


def test_spline_method_ids_follow_the_selected_distance_variant():
    assert spline_method_ids("DANCo_gride_fast")[1] == GRIDE_PROFILED_SPLINE_ID
    assert spline_method_ids("DANCo_fast") == (
        MIND_FULL_SPLINE_ID, MIND_PROFILED_SPLINE_ID,
    )
    with pytest.raises(ValueError, match="no paper spline method IDs"):
        spline_method_ids("DANCo")


def test_cnn_estimators_share_one_stable_exact_deduplicated_sample(monkeypatch):
    import experiments.run_cnn_layers as cnn

    rng = np.random.default_rng(19)
    clean = rng.normal(size=(20, 6))
    duplicated = np.vstack([clean, clean[:4]])
    seen = {}

    class FakeDANCo:
        def __init__(self, *, N, **kwargs):
            seen["danco_n"] = N

        def evaluate_all(self, X, **kwargs):
            seen["danco_rows"] = X.copy()
            return {
                "de_nustrip": 7.0,
                "de_standard": 6.0,
                "dhat": 6.5,
                "mu_nu": 1.0,
                "mu_tau": 10.0,
            }

    def fake_twonn(X, **kwargs):
        seen["twonn_rows"] = X.copy()
        return 5.0

    def fake_mle(X, **kwargs):
        seen["mle_rows"] = X.copy()
        return 4.0

    monkeypatch.setattr(cnn, "DANCo", FakeDANCo)
    monkeypatch.setattr(cnn, "paper_twonn", fake_twonn)
    monkeypatch.setattr(cnn, "paper_mle", fake_mle)

    result = cnn.estimate_layer(
        duplicated,
        backend="cpu",
        maxDim=10,
        n_jobs=1,
        diagnostics=True,
        version="DANCo_gride_fast",
        spline_file=Path("unused.pkl"),
    )

    assert result["n_raw"] == 24
    assert result["n_unique"] == 20
    assert result["n_duplicates"] == 4
    assert seen["danco_n"] == 20
    np.testing.assert_array_equal(seen["danco_rows"], clean)
    np.testing.assert_array_equal(seen["twonn_rows"], clean)
    np.testing.assert_array_equal(seen["mle_rows"], clean)
    assert result["twonn_ref"] == 5.0
    assert result["id_mle"] == 4.0
    assert result["method_id_twonn"] == PAPER_TWONN_ID
    assert result["method_id_mle"] == PAPER_MLE_ID
    assert result["method_id_profiled"] == GRIDE_PROFILED_SPLINE_ID


def test_contract_audit_rows_are_machine_readable_and_classify_exceptions():
    rows = method_contract_rows()

    assert rows
    assert {"artifact", "method_id", "display_name", "status"} <= set(rows[0])
    assert {row["status"] for row in rows} == {
        "UNIFIED", "INTENTIONAL_EXCEPTION",
    }
    assert not ({row["method_id"] for row in rows} & LEGACY_METHOD_IDS)

    checked_in = pd.read_csv(
        "results/diagnostics/paper_method_contract_audit.csv",
        keep_default_na=False,
    )
    expected = pd.DataFrame(rows).fillna("")
    pd.testing.assert_frame_equal(checked_in, expected, check_dtype=False)


def test_paper_table_sources_do_not_carry_ambiguous_legacy_twonn_values():
    wide_sources = [
        Path("results/tables/image_estimates_mnist.csv"),
        Path("results/tables/image_estimates_cifar.csv"),
        Path("results/tables/image_estimates_imagenet.csv"),
    ]
    for path in wide_sources:
        columns = set(pd.read_csv(path, nrows=1).columns)
        assert "2NN_ref" in columns, path
        assert "2NN" not in columns, path

    sensitivity = pd.read_csv("results/baselines/sensitivity.csv")
    assert "2NN_ref" in set(sensitivity["estimator"])
    assert "2NN" not in set(sensitivity["estimator"])


def test_cnn_backend_parity_gate_allows_one_dimension_only():
    from experiments.audit_cnn_backend_parity import (
        parity_within_tolerance,
        documented_parity_exception,
    )

    assert parity_within_tolerance(20, 21, 100, 99, tolerance=1)
    assert not parity_within_tolerance(20, 22, 100, 99, tolerance=1)
    assert not parity_within_tolerance(20, 21, 100, 102, tolerance=1)
    assert documented_parity_exception(
        2, "vgg16", "n02279972", 2, 18, 18, 172, 164,
    )
    assert not documented_parity_exception(
        2, "vgg16", "n02279972", 2, 18, 18, 171, 164,
    )


def test_cnn_result_loader_rejects_incomplete_canonical_sets(tmp_path):
    from experiments.paper_methods import load_canonical_cnn_results

    with pytest.raises(ValueError, match="three canonical CNN CSVs"):
        load_canonical_cnn_results(tmp_path)


def test_checked_in_cnn_result_set_is_complete_and_canonical():
    from experiments.paper_methods import load_canonical_cnn_results

    frame = load_canonical_cnn_results("results/cnn")
    assert len(frame) == 672
    assert frame[["subset_rep", "arch", "category_tag"]].drop_duplicates().shape[0] == 84
