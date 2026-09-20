"""Epistemic value terms for hidden dynamics context."""

from __future__ import annotations

from typing import Any

import numpy as np

from aif.hidden_tilt_belief import HiddenTiltBelief, categorical_entropy, observation_features
from aif.tilt_context import snapshot_with_tilt


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def adaptive_epistemic_beta(
    entropy: float,
    *,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    theta: float = 1.0,
    temperature: float = 0.2,
    mismatch: float = 0.0,
    mismatch_weight: float = 0.0,
) -> float:
    drive = (float(entropy) + float(mismatch_weight) * float(mismatch) - float(theta)) / max(float(temperature), 1e-6)
    return float(beta_min + (float(beta_max) - float(beta_min)) * sigmoid(drive))


def separability_information_gain(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    precision: np.ndarray | None = None,
) -> float:
    """Cheap proxy for expected information gain across tilt predictions."""

    pred = np.asarray(predictions, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    if pred.shape[0] != probs.shape[0]:
        raise ValueError("predictions and probabilities must have the same first dimension.")
    if np.count_nonzero(probs > 1e-8) <= 1:
        return 0.0
    mean = np.sum(pred * probs[:, None], axis=0)
    delta = pred - mean[None, :]
    if precision is None:
        values = np.sum(delta * delta, axis=1)
    else:
        prec = np.asarray(precision, dtype=np.float64)
        values = np.einsum("ij,jk,ik->i", delta, prec, delta)
    return float(np.sum(probs * values))


def epistemic_values_for_candidates(
    *,
    candidates: np.ndarray,
    env_adapter,
    tilt_belief: HiddenTiltBelief,
    snapshot=None,
    approximation: str = "one_step_epistemic",
    horizon_steps: int | None = None,
    feature_indices: tuple[int, ...] | None = None,
    precision: np.ndarray | None = None,
    hypothesis_mode: str = "exact_grid",
    hypothesis_top_m: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute hidden-tilt epistemic values without mutating the live env."""

    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3:
        raise ValueError("candidates must be shaped (K, H, action_dim).")
    live_snapshot = env_adapter.clone_state()
    base_snapshot = live_snapshot if snapshot is None else snapshot.copy()
    approx = str(approximation)
    if approx not in {"one_step_epistemic", "short_horizon_epistemic", "endpoint_epistemic"}:
        raise ValueError("Unknown epistemic approximation.")
    max_horizon = candidates.shape[1]
    if approx == "one_step_epistemic":
        steps = 1
    elif approx == "endpoint_epistemic":
        steps = max_horizon
    else:
        steps = min(max(1, int(horizon_steps or 2)), max_horizon)
    hypothesis_indices = tilt_belief.hypothesis_indices(mode=str(hypothesis_mode), m=hypothesis_top_m, seed=seed)
    hypothesis_probs = np.asarray(tilt_belief.probabilities, dtype=np.float64)[hypothesis_indices]
    hypothesis_probs = hypothesis_probs / max(float(np.sum(hypothesis_probs)), 1e-12)
    values = []
    prediction_tensor = []
    try:
        for candidate in candidates:
            per_tilt = []
            for hypothesis_index in hypothesis_indices:
                tilt = tilt_belief.tilt_grid[int(hypothesis_index)]
                tilted_snapshot = snapshot_with_tilt(base_snapshot, tilt)
                rollout = env_adapter.rollout_from_state(tilted_snapshot, candidate[:steps])
                if approx == "short_horizon_epistemic":
                    features = np.concatenate(
                        [observation_features(obs, feature_indices) for obs in rollout.observations[1 : steps + 1]],
                        axis=0,
                    )
                else:
                    features = observation_features(rollout.observations[-1], feature_indices)
                per_tilt.append(features)
            pred = np.asarray(per_tilt, dtype=np.float32)
            prediction_tensor.append(pred)
            values.append(separability_information_gain(pred, hypothesis_probs, precision=precision))
    finally:
        env_adapter.restore_state(live_snapshot)
    return np.asarray(values, dtype=np.float32), {
        "approximation": approx,
        "horizon_steps": int(steps),
        "hypothesis_mode": str(hypothesis_mode),
        "hypothesis_indices": [int(index) for index in hypothesis_indices.tolist()],
        "hypothesis_probabilities": hypothesis_probs.astype(float).tolist(),
        "tilt_entropy": categorical_entropy(tilt_belief.probabilities),
        "predictions": np.asarray(prediction_tensor, dtype=np.float32).tolist(),
    }


def epistemic_adjusted_scores(
    base_scores: np.ndarray,
    epistemic_values: np.ndarray,
    beta: float,
    *,
    safety_costs: np.ndarray | None = None,
    safety_threshold: float = 1.0,
    allow_unsafe_override: bool = False,
) -> np.ndarray:
    """Subtract epistemic value from G, with a default safety guard."""

    scores = np.asarray(base_scores, dtype=np.float32).reshape(-1)
    values = np.asarray(epistemic_values, dtype=np.float32).reshape(scores.shape)
    bonus = float(beta) * values
    if safety_costs is not None and not bool(allow_unsafe_override):
        safety = np.asarray(safety_costs, dtype=np.float32).reshape(scores.shape)
        bonus = np.where(safety > float(safety_threshold), 0.0, bonus)
    return (scores - bonus).astype(np.float32)


def realized_information_gain(previous_probs: np.ndarray, next_probs: np.ndarray) -> float:
    return float(categorical_entropy(previous_probs) - categorical_entropy(next_probs))
