"""Categorical prediction-update filtering for a changing hidden tilt.

This module is separate from :mod:`aif.belief`, whose paper controller assumes
a constant hidden tilt.  The switching benchmark uses a fixed transition law,
then the same learned Gaussian observation likelihood as the existing filter.
"""
from __future__ import annotations

import numpy as np

from aif.likelihood import logsumexp


def symmetric_transition(off_diagonal_probability: float, states: int = 3) -> np.ndarray:
    """Return ``T[new_state, previous_state]`` with equal switch probabilities.

    The total hazard is ``h = (states - 1) * off_diagonal_probability``.
    For three tilts each alternative has probability h/2 and staying has 1-h.
    This is a latent-state transition before observation evidence, not posterior
    mixing or likelihood tempering.
    """
    q = float(off_diagonal_probability)
    if states < 2 or not np.isfinite(q) or q < 0 or q * (states - 1) >= 1:
        raise ValueError("Invalid symmetric transition probability")
    matrix = np.full((states, states), q, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0 - q * (states - 1))
    np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-15, rtol=0)
    return matrix


def prediction_update(
    belief: np.ndarray,
    log_likelihood: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one HMM prediction-update step in log space.

    Returns the transition-predicted prior and normalized posterior.  There is
    no uniform posterior mixing and no knowledge of a physical switch time.
    """
    prior = np.asarray(belief, dtype=np.float64).reshape(-1)
    evidence = np.asarray(log_likelihood, dtype=np.float64).reshape(-1)
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.shape != (len(prior), len(prior)):
        raise ValueError("Transition matrix and belief dimensions differ")
    if (
        not np.isfinite(prior).all()
        or not np.isfinite(evidence).all()
        or not np.isfinite(matrix).all()
        or np.any(prior < 0)
        or np.any(matrix < 0)
        or prior.sum() <= 0
    ):
        raise ValueError("Belief update inputs must be finite probabilities and scores")
    prior = prior / prior.sum()
    predicted = matrix @ prior
    if np.any(predicted <= 0):
        raise ValueError("Transition prediction must keep all benchmark hypotheses possible")
    logits = np.log(predicted) + evidence
    posterior = np.exp(logits - logsumexp(logits))
    posterior /= posterior.sum()
    return predicted, posterior


def filter_log_likelihoods(
    log_likelihoods: np.ndarray,
    transition: np.ndarray,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a valid sequence, returning priors and posteriors including t=0."""
    scores = np.asarray(log_likelihoods, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("Expected transition x hypothesis log likelihoods")
    belief = (
        np.full(scores.shape[1], 1.0 / scores.shape[1], dtype=np.float64)
        if prior is None
        else np.asarray(prior, dtype=np.float64).reshape(scores.shape[1])
    )
    belief = belief / belief.sum()
    predicted = [belief.copy()]
    posterior = [belief.copy()]
    for row in scores:
        forecast, belief = prediction_update(belief, row, transition)
        predicted.append(forecast)
        posterior.append(belief.copy())
    return np.asarray(predicted), np.asarray(posterior)
