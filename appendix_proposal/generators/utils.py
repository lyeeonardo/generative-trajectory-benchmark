"""Utilities for enforcing the Generator generator contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from generators.base import ProposalBatch


def context_to_vector(context: Any) -> np.ndarray:
    if hasattr(context, "normalized_vector"):
        return np.asarray(context.normalized_vector, dtype=np.float32).reshape(-1)
    if hasattr(context, "to_context_vector"):
        return np.asarray(context.to_context_vector(), dtype=np.float32).reshape(-1)
    return np.asarray(context, dtype=np.float32).reshape(-1)


def clip_actions(actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    low, high = action_bounds
    return np.clip(
        np.asarray(actions, dtype=np.float32),
        np.asarray(low, dtype=np.float32).reshape(1, 1, -1),
        np.asarray(high, dtype=np.float32).reshape(1, 1, -1),
    ).astype(np.float32)


def validate_proposal_batch(
    batch: ProposalBatch,
    *,
    K: int,
    H: int,
    action_dim: int,
    action_bounds: tuple[np.ndarray, np.ndarray],
) -> ProposalBatch:
    actions = np.asarray(batch.actions, dtype=np.float32)
    expected = (int(K), int(H), int(action_dim))
    if actions.shape != expected:
        raise ValueError(f"Expected actions shaped {expected}, got {actions.shape}.")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Generator returned non-finite actions.")
    low, high = action_bounds
    low = np.asarray(low, dtype=np.float32).reshape(1, 1, -1)
    high = np.asarray(high, dtype=np.float32).reshape(1, 1, -1)
    if np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
        raise ValueError("Generator returned actions outside bounds.")
    if batch.log_prob is not None and np.asarray(batch.log_prob).shape != (int(K),):
        raise ValueError(f"Expected log_prob shaped {(int(K),)}, got {np.asarray(batch.log_prob).shape}.")
    if batch.observations is not None:
        observations = np.asarray(batch.observations, dtype=np.float32)
        if observations.ndim != 3:
            raise ValueError(f"Expected observations shaped (K, H, obs_dim) or (K, H + 1, obs_dim), got {observations.shape}.")
        if observations.shape[0] != int(K) or observations.shape[1] not in {int(H), int(H) + 1}:
            raise ValueError(f"Expected observations first axes {(int(K), int(H))} or {(int(K), int(H) + 1)}, got {observations.shape[:2]}.")
        if observations.shape[2] <= 0:
            raise ValueError("Generated observations must have a positive obs_dim.")
        if not np.all(np.isfinite(observations)):
            raise ValueError("Generator returned non-finite observations.")
    if batch.sample_time_sec < 0.0:
        raise ValueError("sample_time_sec must be non-negative.")
    return batch


def stable_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
