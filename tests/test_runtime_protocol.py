import pytest

from experiments.run_runtime import effective_config, validate_config


def test_max_dim_sweep_keeps_the_requested_candidate_range():
    config = effective_config("maxDim", 400)

    assert config["maxDim"] == 400
    assert config["D"] >= 400
    assert config["N"] == 500


def test_non_max_dim_sweeps_never_request_more_candidates_than_features():
    for sweep, value in (("N", 625), ("D", 100), ("k", 20)):
        config = effective_config(sweep, value)

        assert config["maxDim"] <= config["D"]


def test_worker_rejects_a_silently_truncated_candidate_range():
    with pytest.raises(ValueError, match="candidate dimensions"):
        validate_config({"N": 80, "D": 20, "k": 5, "maxDim": 40})
