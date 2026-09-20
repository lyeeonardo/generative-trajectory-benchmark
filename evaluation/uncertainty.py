"""Deterministic uncertainty summaries for the fixed-condition paper study.

The paper E1 analysis
uses ``fixed_case_trial_interval``: it resamples locked proposal-trial seeds as
blocks while retaining all nine fixed physical cases in every sampled trial.
"""
from __future__ import annotations

import math
import numpy as np

def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    m = float(values.max())
    return m + math.log(float(np.exp(values - m).sum()))


def _binomial_range_probability(n, lo, hi, p):
    if lo > hi:
        return 0.0
    if p <= 0:
        return 1.0 if lo <= 0 <= hi else 0.0
    if p >= 1:
        return 1.0 if lo <= n <= hi else 0.0
    q = 1.0 - p
    logs = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log(q)
        for k in range(lo, hi + 1)
    ]
    return float(math.exp(_logsumexp(logs)))


def clopper_pearson(successes, total, alpha=0.05):
    """Exact equal-tailed binomial interval, computed without optional SciPy."""
    successes, total = int(successes), int(total)
    if total <= 0 or not 0 <= successes <= total or not 0 < alpha < 1:
        raise ValueError("Invalid binomial interval inputs")
    target = alpha / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, successes / total
        for _ in range(80):
            mid = (lo + hi) / 2
            if _binomial_range_probability(total, successes, total, mid) < target:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if successes == total:
        upper = 1.0
    else:
        lo, hi = successes / total, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if _binomial_range_probability(total, 0, successes, mid) > target:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return {
        "estimate": successes / total,
        "ci95": [float(lower), float(upper)],
        "successes": successes,
        "episodes": total,
        "method": "Clopper-Pearson exact equal-tailed interval",
    }


def fixed_case_trial_interval(values, *, draws=20000, seed=260916):
    """Bootstrap locked proposal trials while retaining every fixed case.

    Values may have shape (fixed_condition, proposal_trial) or
    (1, fixed_condition, proposal_trial). A paired contrast supplies paired
    row-wise differences. The same sampled trial indices apply to every
    condition, preserving each proposal RNG trial as a nine-case block.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError("Fixed-case trial interval requires exactly one training fit")
        values = values[0]
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite fixed_condition x proposal_trial values")
    draws = int(draws)
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    n_condition, n_trial = values.shape
    trial_ix = rng.integers(0, n_trial, size=(draws, n_trial))
    estimates = values[:, trial_ix].mean(axis=(0, 2))
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_draws": draws,
        "bootstrap_seed": int(seed),
        "training_seed": 13,
        "fixed_conditions": int(n_condition),
        "locked_proposal_trials": int(n_trial),
        "method": "deterministic fixed-case proposal-trial block percentile bootstrap",
        "conditional_on_fixed_conditions": True,
        "retraining_variation_included": False,
    }


def paired_binary_interval(target, reference, alpha=.05):
    """Conservative >=95% paired risk-difference interval from discordant cells.

    Two exact binomial 97.5% intervals and Bonferroni coverage bound the two
    discordant probabilities. This stays nondegenerate with no discordances.
    """
    a=np.asarray(target,bool);b=np.asarray(reference,bool)
    if a.ndim!=1 or a.shape!=b.shape or not len(a):raise ValueError('Expected matched nonempty binary trials')
    wins=int(np.sum(a & ~b));losses=int(np.sum(~a & b));n=len(a)
    plus=clopper_pearson(wins,n,alpha=alpha/2);minus=clopper_pearson(losses,n,alpha=alpha/2)
    return dict(mean=(wins-losses)/n,ci95=[plus['ci95'][0]-minus['ci95'][1],plus['ci95'][1]-minus['ci95'][0]],
        favorable_discordant=wins,unfavorable_discordant=losses,paired_trials=n,
        method='Conservative paired risk-difference interval; exact discordant-cell binomial intervals with Bonferroni coverage')
