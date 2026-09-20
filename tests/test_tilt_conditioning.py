import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.tilt_conditioning import label_schedule, supplied_labels, paired_interval
from evaluation.reuse import run_group, TOLERANCE
from tests.fakes import CASE, COLLECTION, FakeEnv, factory
from tests.test_reuse import Proposals


def test_wrong_labels_are_balanced_and_do_not_change_cases():
    protocol = json.loads(Path('datasets/evaluation/test_cases.json').read_text())
    before = copy.deepcopy(protocol)
    schedule = label_schedule(protocol)
    counts = np.zeros((3, 3), int)
    for row in schedule:
        counts[row['true_index'], row['wrong_index']] += 1
    np.testing.assert_array_equal(np.diag(counts), 0)
    assert np.all(counts[~np.eye(3, dtype=bool)] == 15)
    assert len(supplied_labels(schedule, 'wrong')) == 90
    assert protocol == before


def test_supplied_label_changes_model_context_only_and_is_audited(tmp_path, monkeypatch):
    from evaluation import replay
    wrong = np.deg2rad([0, 0]).tolist()
    p = Proposals(); scenes = []
    def env_factory(collection, scene):
        scenes.append(copy.deepcopy(scene)); return factory(FakeEnv(3))(collection, scene)
    kwargs = dict(policy='every_step', device='cpu', env_factory=env_factory,
                  proposal_fn=p, supplied_tilts=[wrong])
    result = run_group(None, [CASE], COLLECTION, tmp_path, **kwargs)
    assert scenes == [CASE['scene']]
    assert not result['oracle_true_tilt']
    assert result['episodes'][0]['true_tilt'] == CASE['tilt_degrees']
    assert not result['episodes'][0]['conditioning_correct']
    for context in p.contexts:
        np.testing.assert_array_equal(context['tilt'], np.zeros((1, 2)))
    monkeypatch.setattr(replay, 'environment', lambda collection, scene: factory(FakeEnv(3))(collection, scene))
    args = (tmp_path, [CASE], COLLECTION, 'every_step', TOLERANCE)
    assert replay.audit_group((*args, [wrong]))['errors'] == []
    assert 'public_context:tilt' in replay.audit_group(args)['errors']
    with pytest.raises(ValueError, match='Group inputs'):
        run_group(None, [CASE], COLLECTION, tmp_path, **{**kwargs, 'supplied_tilts': [CASE['tilt_radians']]})


def test_label_intervention_keeps_same_initial_random_draws_and_resume(tmp_path):
    kwargs = dict(policy='every_step', device='cpu', evaluation_seed=42)
    def run(dest, labels, steps=None):
        return run_group(None, [CASE], COLLECTION, dest, **kwargs,
            env_factory=factory(FakeEnv(4)), proposal_fn=Proposals(), supplied_tilts=labels, stop_after_steps=steps)
    run(tmp_path/'correct', [CASE['tilt_radians']])
    run(tmp_path/'wrong', [[0., 0.]], 2)
    resumed = run(tmp_path/'wrong', [[0., 0.]])
    assert resumed['replay_steps_this_invocation'] == 2
    with np.load(tmp_path/'correct/step_000_proposal.npz') as a, np.load(tmp_path/'wrong/step_000_proposal.npz') as b:
        np.testing.assert_array_equal(a['rng_after'], b['rng_after'])
        np.testing.assert_array_equal(a['raw_actions'], b['raw_actions'])
        for key in a.files:
            if key.startswith('context_') and key != 'context_tilt':
                np.testing.assert_array_equal(a[key], b[key])


def test_paired_interval_preserves_sign_and_layout_clusters():
    report = paired_interval(np.arange(5.), np.arange(5))
    assert report['mean'] == 2
    assert report['bootstrap_draws'] == 3125
    zero = paired_interval(np.zeros(10), np.repeat(np.arange(5), 2))
    assert zero['ci95'] == [0., 0.]
