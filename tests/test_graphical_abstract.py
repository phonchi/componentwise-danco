"""The graphical abstract must keep its comparisons and conditions matched."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from experiments.make_graphical_abstract import GSM, NOISE, load_evidence


def test_ga_selects_the_reported_noise_and_raw_gsm_comparisons():
    values = load_evidence()
    assert values == pytest.approx({
        "noise_mind": 27.744788059251178,
        "noise_gride": 17.643343804049284,
        "gsm_full": 22.801761802409924,
        "gsm_profiled": 66.66727832097735,
    })


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_replicates"])
def test_ga_refuses_ambiguous_or_mislabelled_evidence(tmp_path, change):
    frame = pd.read_csv(GSM)
    selected = ((frame["method"] == "profiled_danco")
                & (frame["intervention"] == "raw")
                & (frame["scale_sigma"] == 0.25)
                & (frame["noise_sigma"] == 0))
    if change == "missing":
        frame = frame.loc[~selected]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.loc[selected]], ignore_index=True)
    else:
        frame.loc[selected, "n_reps"] = 29
    altered = tmp_path / "gsm.csv"
    frame.to_csv(altered, index=False)
    with pytest.raises(ValueError):
        load_evidence(NOISE, altered)
