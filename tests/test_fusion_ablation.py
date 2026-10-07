import numpy as np
import pandas as pd

from danco import DANCo
from experiments.fusion_ablation_summary import (
    METHODS,
    aggregate,
    paired_intervention_bootstrap,
)
from experiments.run_fusion_ablation import (
    COLUMNS,
    component_estimates,
    gaussian_scale_mixture,
    gaussian_scale_mixture_components,
    load_checkpoint,
)


def test_component_objectives_rebuild_joint_estimates():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 8))
    estimator = DANCo(
        N=len(X),
        k=5,
        random_state=812,
        fractal=True,
        calibration_seeding="derived",
    )

    stats = estimator.evaluate_all(X, maxDim=8, n_jobs=1)
    estimates = component_estimates(stats, ambient_dim=X.shape[1], fractal=True)

    assert estimates["joint_standard"] == stats["de_standard"]
    assert estimates["joint_profiled"] == stats["de_nustrip"]


def test_latent_descaling_reuses_the_raw_gsm_latent_draw():
    latent, scales, observed = gaussian_scale_mixture_components(
        seed=7, sigma_s=0.5, n=40, d=6,
    )

    np.testing.assert_allclose(observed / scales[:, None], latent)
    np.testing.assert_array_equal(
        observed, gaussian_scale_mixture(seed=7, sigma_s=0.5, n=40, d=6),
    )


def _intervention_frame():
    rows = []
    for rep in range(2):
        for intervention, estimate in (("raw", 5.0), ("latent_descaled", 7.0)):
            for signature in METHODS:
                rows.append({
                    "suite": "gsm",
                    "dataset": "gsm_scale_0.5",
                    "true_d": 10.0,
                    "rep": rep,
                    "noise_sigma": 0.0,
                    "scale_sigma": 0.5,
                    "intervention": intervention,
                    "distance_model": signature[0],
                    "angle_mode": signature[1],
                    "objective": signature[2],
                    "estimate": estimate,
                })
    return pd.DataFrame(rows)


def test_intervention_summary_keeps_regimes_separate_and_pairs_replicates():
    frame = _intervention_frame()

    summary = aggregate(frame)
    contrast = paired_intervention_bootstrap(frame, n_boot=200, seed=3)

    assert set(summary["intervention"]) == {"raw", "latent_descaled"}
    assert len(contrast) == len(METHODS)
    np.testing.assert_allclose(contrast["delta_estimate"], 2.0)
    np.testing.assert_allclose(contrast["delta_mpe"], -20.0)
    assert set(contrast["n_pairs"]) == {2}


def test_checkpoint_rows_without_intervention_are_backfilled_as_raw(tmp_path):
    path = tmp_path / "legacy.csv"
    frame = _intervention_frame()
    raw = frame[frame["intervention"] == "raw"].copy()
    raw = raw[raw["rep"] == 0]
    defaults = {
        "ambient_d": 10, "n": 20, "data_seed": 0, "noise_seed": np.nan,
        "max_dim": 10, "calibration_seed": 1, "integer_argmin": 5,
        "mu_nu": 1.0, "mu_tau": 8.0, "base_dhat": 5.0, "seconds": 0.1,
    }
    for key, value in defaults.items():
        raw[key] = value
    raw[[column for column in COLUMNS if column != "intervention"]].to_csv(
        path, index=False,
    )

    rows, complete = load_checkpoint(path)

    assert len(complete) == 1
    assert {row["intervention"] for row in rows} == {"raw"}
