import numpy as np
import pytest

from evaluation.uncertainty import (
    clopper_pearson,
    fixed_case_trial_interval,
)


def test_exact_binomial_interval_known_values_and_edges():
    result = clopper_pearson(5, 10)
    assert result["estimate"] == 0.5
    assert result["ci95"] == pytest.approx([0.187086, 0.812914], abs=1e-6)
    assert clopper_pearson(0, 10)["ci95"] == pytest.approx([0.0, 0.308497], abs=1e-6)
    assert clopper_pearson(10, 10)["ci95"] == pytest.approx([0.691503, 1.0], abs=1e-6)


def test_fixed_case_trial_interval_is_deterministic_and_keeps_cases_in_trial_blocks():
    values = np.arange(9 * 10, dtype=float).reshape(9, 10)
    a = fixed_case_trial_interval(values, draws=200, seed=7)
    b = fixed_case_trial_interval(values[None], draws=200, seed=7)
    assert a == b
    assert a["mean"] == values.mean()
    assert a["training_seed"] == 13
    assert a["fixed_conditions"] == 9
    assert a["locked_proposal_trials"] == 10
    assert a["retraining_variation_included"] is False
    zero = fixed_case_trial_interval(np.zeros((9, 10)), draws=100)
    assert zero["ci95"] == [0.0, 0.0]


def test_fixed_case_trial_interval_rejects_multiple_training_fits():
    with pytest.raises(ValueError, match="exactly one training fit"):
        fixed_case_trial_interval(np.zeros((2, 9, 10)), draws=10)


def test_paired_success_interval_handles_no_discordances_without_equivalence_claim():
    from evaluation.uncertainty import paired_binary_interval
    result=paired_binary_interval([True]*10,[True]*10)
    assert result['mean']==0 and result['ci95'][0]<0<result['ci95'][1]
    win=paired_binary_interval([True]*10,[False]*10)
    assert win['mean']==1 and 0<win['ci95'][0]<=win['ci95'][1]==1
