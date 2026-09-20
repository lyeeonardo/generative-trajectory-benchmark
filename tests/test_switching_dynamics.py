import numpy as np

from aif.switching_belief import filter_log_likelihoods, prediction_update, symmetric_transition
from environment.task import UphillTask, scene
from evaluation.switching_dynamics import rebuild_task_preserving_public_state


def test_symmetric_transition_and_update():
    transition = symmetric_transition(.01)
    np.testing.assert_allclose(transition.sum(0), 1)
    prior = np.asarray([0, 1, 0.], dtype=float)
    predicted, posterior = prediction_update(prior, np.zeros(3), transition)
    np.testing.assert_allclose(predicted, [.01, .98, .01])
    np.testing.assert_allclose(posterior, predicted)


def test_filter_can_recover_after_confidence_without_reset():
    transition = symmetric_transition(.01)
    flat = np.asarray([-8, 0, -8.])
    minus = np.asarray([0, -8, -8.])
    _, posterior = filter_log_likelihoods(np.stack([flat] * 5 + [minus] * 3), transition)
    assert posterior[5, 1] > .99
    assert posterior[-1, 0] > .99


def test_switch_preserves_public_state_clock_and_dwell():
    env = UphillTask(max_steps=100)
    env.reset(scene(0, 0))
    for _ in range(3):
        env.step(np.asarray([0.0, .2, 0.0], np.float32))
    before = env.current_observation().copy()
    step, elapsed, dwell = env.state.step, env.state.time, env.settle_count
    switched, audit = rebuild_task_preserving_public_state(env, scene(0, -15))
    after = switched.current_observation()
    np.testing.assert_allclose(after[:12], before[:12], atol=2e-6, rtol=0)
    assert switched.state.step == step
    assert switched.state.time == elapsed
    assert switched.settle_count == dwell
    assert after[12] == np.float32(np.deg2rad(-15))
    assert audit["maximum_public_state_jump"] <= 2e-6


def test_total_hazard_matches_direct_bayesian_formula():
    hazard = .06
    transition = np.full((3, 3), hazard / 2)
    np.fill_diagonal(transition, 1 - hazard)
    np.testing.assert_allclose(symmetric_transition(hazard / 2), transition)
    belief = np.asarray([.001, .998, .001])
    likelihood = np.asarray([.8, .1, .1])
    predicted, posterior = prediction_update(belief, np.log(likelihood), transition)
    expected_prior = transition @ belief
    expected = expected_prior * likelihood
    expected /= expected.sum()
    np.testing.assert_allclose(predicted, expected_prior)
    np.testing.assert_allclose(posterior, expected)


def test_zero_hazard_reduces_to_constant_context_update():
    belief = np.asarray([.2, .5, .3])
    likelihood = np.asarray([.1, .6, .3])
    predicted, posterior = prediction_update(belief, np.log(likelihood), symmetric_transition(0))
    np.testing.assert_allclose(predicted, belief)
    np.testing.assert_allclose(posterior, belief * likelihood / (belief * likelihood).sum())


def test_transition_precedes_evidence_without_posterior_mixing():
    transition = symmetric_transition(.03)
    predicted, posterior = prediction_update(np.asarray([0., 1., 0.]), np.asarray([0., -100., -100.]), transition)
    np.testing.assert_allclose(predicted, [.03, .94, .03])
    assert posterior[0] > 1 - 1e-12
    assert posterior[1] < 1e-30  # no uniform floor applied after evidence


def test_positive_switch_uses_ten_flat_then_twenty_changed_transitions(tmp_path, monkeypatch):
    from pathlib import Path
    import evaluation.switching_dynamics as benchmark
    source = benchmark._source_paths('development')[0]
    destination = tmp_path / 'positive_switch.npz'
    monkeypatch.setattr(benchmark, 'ROOT', Path('/'))
    row = benchmark._generate_trial(source, 10, destination, switched_tilt_degrees=15, max_transitions=30)
    with np.load(destination) as actual, np.load(source) as reference:
        np.testing.assert_array_equal(actual['true_tilt_index'][:11], 1)
        np.testing.assert_array_equal(actual['true_tilt_index'][11:], 2)
        np.testing.assert_allclose(actual['public_observations'][:11], reference['observations'][:11, :7], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(actual['actions'], reference['actions'][:len(actual['actions'])])
        assert len(actual['actions']) <= 30
    assert row['public_state_jump_at_switch'] <= 2e-6
    assert row['clock_and_dwell_preserved']
