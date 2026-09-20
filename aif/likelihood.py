"""Shared learned Gaussian likelihood and one-step mutual information."""
from __future__ import annotations
import numpy as np

def logsumexp(values, axis=-1):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values-maximum), axis=axis, keepdims=True)), axis=axis)


def observation_variance(variance):
    """Match the Torch observation law, including its circular variance cap."""
    value = np.asarray(variance, dtype=np.float64).copy()
    if value.ndim < 1 or value.shape[-1] != 7 or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError('Expected finite positive seven-feature variances')
    value[..., 6] = np.minimum(value[..., 6], np.pi**2)
    return value


def wrapped_gaussian_log_likelihood(observation, predictions, variance):
    """Normalized six-Gaussian plus wrapped-yaw log likelihood.

    ``predictions`` and ``variance`` share a trailing seven-feature axis. The
    first six coordinates use independent Gaussians; yaw uses a normalized
    wrapped Gaussian. This law is shared by learned filtering and information.
    """
    y = np.asarray(observation, dtype=np.float64)
    mean = np.asarray(predictions, dtype=np.float64)
    var = observation_variance(variance)
    if mean.shape[-1] != 7 or var.shape[-1] != 7 or np.any(var <= 0):
        raise ValueError("Expected finite positive seven-feature Gaussian moments")
    if not np.isfinite(y).all() or not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise ValueError("Likelihood inputs must be finite")
    delta = y - mean
    normal = -.5 * np.sum(
        delta[..., :6] ** 2 / var[..., :6] + np.log(2 * np.pi * var[..., :6]), axis=-1
    )
    angle = np.arctan2(np.sin(delta[..., 6]), np.cos(delta[..., 6]))
    shifts = np.arange(-8, 9, dtype=np.float64) * (2 * np.pi)
    yaw_terms = -.5 * (
        (angle[..., None] + shifts) ** 2 / var[..., 6, None]
        + np.log(2 * np.pi * var[..., 6, None])
    )
    return normal + logsumexp(yaw_terms, axis=-1)


def learned_gaussian_mixture_information(means, variances, probabilities, *, samples=64, seed=0):
    """One-step MI for candidate-specific learned Gaussian/wrapped laws.

    Inputs have shape ``(candidate, hypothesis, 7)``. Each hypothesis supplies
    ``samples`` draws, and the same standard-normal draws are reused for every
    candidate. Finite Monte Carlo negatives remain visible with their errors.
    """
    mu = np.asarray(means, dtype=np.float64)
    var = observation_variance(variances)
    q = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if mu.ndim != 3 or mu.shape != var.shape or mu.shape[1:] != (len(q), 7):
        raise ValueError("Expected paired (candidate,hypothesis,7) moments")
    if samples < 2 or np.any(q < 0) or not np.isfinite(q).all() or q.sum() <= 0:
        raise ValueError("Invalid MI sample count or categorical probabilities")
    if not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("Invalid learned Gaussian moments")
    q = q / q.sum()
    draws = np.zeros((len(mu), int(samples)), dtype=np.float64)
    if np.count_nonzero(q) == 1:
        return draws.mean(1), draws.std(1), draws
    eps = np.random.default_rng(seed).normal(size=(len(q), int(samples), 7))
    logq = np.full(len(q), -np.inf)
    logq[q > 0] = np.log(q[q > 0])
    for k in range(len(mu)):
        if np.all(mu[k] == mu[k, :1]) and np.all(var[k] == var[k, :1]):
            continue
        y = mu[k, :, None, :] + eps * np.sqrt(var[k, :, None, :])
        y[..., 6] = np.arctan2(np.sin(y[..., 6]), np.cos(y[..., 6]))
        likelihoods = wrapped_gaussian_log_likelihood(
            y[:, :, None, :], mu[k, None, None, :, :], var[k, None, None, :, :]
        )
        own = likelihoods[np.arange(len(q)), :, np.arange(len(q))]
        integrand = own - logsumexp(likelihoods + logq[None, None, :], axis=-1)
        draws[k] = np.sum(q[:, None] * integrand, axis=0)
    return draws.mean(1), draws.std(1, ddof=1) / np.sqrt(samples), draws
