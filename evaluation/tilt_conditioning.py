"""E1 controlled label intervention and paired capability measurements.

No maintained belief, candidate ranking, information bonus, or simulator model.
Wrong labels are balanced over all five alternatives across layouts and RNGs.
"""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from aif.observation import law_from_samples
from data.banks import load_bank
from evaluation.capabilities import circular_error
from evaluation.common import BANK, CONFIG, atomic_npz


def label_schedule(protocol):
    grid = json.loads((CONFIG / 'experiment1.json').read_text())['tilts_degrees']
    layouts = sorted({c['base_layout_id'] for c in protocol['cases']})
    seeds = sorted(protocol['evaluation_rng_seeds'])
    rows = []
    for case in protocol['cases']:
        true_index = grid.index(case['tilt_degrees'])
        for seed in protocol['evaluation_rng_seeds']:
            offset = 1 + (layouts.index(case['base_layout_id']) * len(seeds) + seeds.index(seed)) % (len(grid) - 1)
            wrong_index = (true_index + offset) % len(grid)
            rows.append(dict(case_id=case['case_id'], base_layout_id=case['base_layout_id'],
                             evaluation_rng_seed=seed, true_index=true_index, wrong_index=wrong_index,
                             correct_radians=case['tilt_radians'],
                             wrong_radians=np.deg2rad(grid[wrong_index]).tolist()))
    return rows


def supplied_labels(schedule, condition):
    if condition not in ('correct', 'wrong'):
        raise ValueError('Condition must be correct or wrong')
    return {f"{r['case_id']}/rng{r['evaluation_rng_seed']}": r[condition + '_radians'] for r in schedule}


def paired_interval(values, layouts):
    """Exact paired cluster bootstrap over the supplied fixed-layout labels."""
    values = np.asarray(values, dtype=np.float64)
    layouts = np.asarray(layouts)
    unique = sorted(set(layouts.tolist()))
    if values.ndim != 1 or values.shape != layouts.shape or not np.isfinite(values).all():
        raise ValueError('Invalid paired observations')
    means = np.asarray([values[layouts == layout].mean() for layout in unique])
    counts = np.asarray([np.sum(layouts == layout) for layout in unique])
    if len(unique) > 6:
        raise ValueError('Exact enumeration intended for at most six layouts')
    indices = np.asarray(list(product(range(len(unique)), repeat=len(unique))))
    draws = np.sum(means[indices] * counts[indices], axis=1) / counts[indices].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                independent_layouts=len(unique), paired_rows=len(values), bootstrap_draws=len(draws))


def behavior_comparison(correct, wrong):
    key = lambda r: (r['case_id'], r['evaluation_rng_seed'])
    a = {key(r): r for r in correct['episodes']}
    b = {key(r): r for r in wrong['episodes']}
    if a.keys() != b.keys() or len(a) != len(correct['episodes']):
        raise ValueError('Unpaired or duplicate behavior rows')
    keys = sorted(a)
    layouts = [a[k]['base_layout_id'] for k in keys]
    if not all(a[k]['conditioning_correct'] and not b[k]['conditioning_correct'] for k in keys):
        raise ValueError('Incorrect intervention labels')
    success = [100 * (int(a[k]['success']) - int(b[k]['success'])) for k in keys]
    cost = [b[k]['actual_operational_cost'] - a[k]['actual_operational_cost'] for k in keys]
    distance = [b[k]['final_goal_distance'] - a[k]['final_goal_distance'] for k in keys]
    return dict(episodes_per_condition=len(keys), correct_successes=correct['successes'],
                wrong_successes=wrong['successes'], correct_mean_cost=correct['mean_actual_cost'],
                wrong_mean_cost=wrong['mean_actual_cost'],
                success_gain_correct_minus_wrong_pp=paired_interval(success, layouts),
                cost_reduction_wrong_minus_correct=paired_interval(cost, layouts),
                final_distance_reduction_wrong_minus_correct_m=paired_interval(distance, layouts),
                executed_steps=correct['executed_steps'] + wrong['executed_steps'],
                paired_success_counts=dict(correct_only=sum(a[k]['success'] and not b[k]['success'] for k in keys),
                    wrong_only=sum(b[k]['success'] and not a[k]['success'] for k in keys),
                    both=sum(a[k]['success'] and b[k]['success'] for k in keys),
                    neither=sum(not a[k]['success'] and not b[k]['success'] for k in keys)),
                by_true_tilt={str(tilt): dict(correct_successes=sum(a[k]['success'] for k in keys if a[k]['true_tilt'] == tilt),
                    wrong_successes=sum(b[k]['success'] for k in keys if b[k]['true_tilt'] == tilt),
                    episodes=sum(a[k]['true_tilt'] == tilt for k in keys))
                    for tilt in json.loads((CONFIG/'experiment1.json').read_text())['tilts_degrees']})


def prediction_comparison(source, output):
    """Reanalyze saved H1 forecasts; every query/action is scored under every configured label.

    The wrong-label loss is the mean of the individual wrong-label conditional losses,
    never the loss of their averaged prediction. Uncertainty clusters by layout.
    """
    source, output = Path(source), Path(output)
    bank, meta = load_bank('test')
    grid = json.loads((CONFIG/'experiment1.json').read_text())['tilts_degrees']
    true = np.repeat([grid.index(m['tilt_degrees']) for m in meta], 16)
    layouts = np.repeat([m['base_layout_id'] for m in meta], 16)
    with np.load(BANK/'truth/test_fixed16.npz') as data:
        truth = data['observations'].reshape(-1, 6, 7)[:, 0]
        valid = data['valid'].reshape(-1, 6)[:, 0]
    extra = json.loads((source/'calibration.json').read_text())['extra_variance']
    with np.load(source/'sensing_log_likelihoods.npz') as data:
        stored_logp = data['log_likelihood'].copy()
    errors, values, means = [], [], []
    maximum_likelihood_error = 0.
    for z in range(len(grid)):
        with np.load(source/f'sensing_hypothesis_{z}.npz') as data:
            samples = data['samples']
        if samples.shape != (len(truth), 16, 7) or not np.isfinite(samples).all():
            raise ValueError('Invalid saved hypothesis predictions')
        law = law_from_samples(torch.from_numpy(samples), extra)
        logp = law.log_prob(torch.from_numpy(truth)).numpy()
        maximum_likelihood_error = max(maximum_likelihood_error, float(np.max(abs(logp - stored_logp[:, z]))))
        if not np.allclose(logp, stored_logp[:, z], rtol=1e-6, atol=1e-5):
            raise ValueError('Stored and recomputed likelihood differ')
        mean = law.mean.numpy()
        errors.append(circular_error(mean, truth)); values.append(logp); means.append(mean)
    errors = np.stack(errors, axis=1).astype(np.float64)
    logp = np.stack(values, axis=1).astype(np.float64)
    indices = np.arange(len(truth)); correct_error = errors[indices, true]
    other = np.arange(len(grid))[None, :] != true[:, None]
    wrong_squared = (errors**2)[other].reshape(len(truth), len(grid)-1, 7).mean(1)
    correct_squared = correct_error**2
    correct_nll = -logp[indices, true]
    wrong_nll = -logp[other].reshape(len(truth), len(grid)-1).mean(1)
    atomic_npz(output/'fixed_action_losses.npz', correct_squared_error=correct_squared,
               wrong_mean_squared_error=wrong_squared, correct_nll=correct_nll, wrong_mean_nll=wrong_nll,
               valid=valid, true_indices=true, layouts=layouts, means=np.stack(means, axis=1),
               log_likelihood=logp, imposed_actions=bank['actions'].reshape(-1, 6, 3)[:, 0])
    report = dict(source=str(source), horizon=1, snapshots=len(meta), fixed_actions_per_snapshot=16,
                  valid_transitions=int(valid.sum()), hypotheses_per_transition=len(grid), samples_per_hypothesis=16,
                  correct_rmse=np.sqrt(correct_squared[valid].mean(0)).tolist(),
                  wrong_rmse=np.sqrt(wrong_squared[valid].mean(0)).tolist(),
                  correct_ball_position_rmse_cm=100*float(np.sqrt(correct_squared[valid, :2].sum(1).mean())),
                  wrong_ball_position_rmse_cm=100*float(np.sqrt(wrong_squared[valid, :2].sum(1).mean())),
                  correct_mean_nll=float(correct_nll[valid].mean()), wrong_mean_nll=float(wrong_nll[valid].mean()),
                  nll_benefit_wrong_minus_correct=paired_interval((wrong_nll-correct_nll)[valid], layouts[valid]),
                  ball_squared_error_benefit_m2=paired_interval((wrong_squared[:, :2]-correct_squared[:, :2]).sum(1)[valid], layouts[valid]),
                  likelihood_recomputation_max_error=maximum_likelihood_error,
                  scope='Fixed public histories and imposed actions, NULL prediction query; no Bayes filter or planner; mean loss over all configured wrong labels.')
    return report
