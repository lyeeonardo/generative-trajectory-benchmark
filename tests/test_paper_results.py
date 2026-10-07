"""Keep the revised E2 table's retained recovery arm in the paper checks."""
from scripts.verify_paper_results import table3


def test_table3_includes_recovery_trained_diffusion():
    rows = table3()
    assert len(rows) == 5
    assert rows[-1] == {
        "model": "Diffusion + recovery data",
        "delay_median": 1,
        "delay_range": [1, 2],
        "identified_by_six": "12/12",
        "correct_observation_30": "12/12",
    }
