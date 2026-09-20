"""Rollout helpers for batched candidate evaluation."""

from __future__ import annotations

import numpy as np


def rollout_action_batch(env_adapter, state_snapshot, candidates: np.ndarray) -> list[object]:
    actions = np.asarray(candidates, dtype=np.float32)
    if actions.ndim != 3:
        raise ValueError(f"Expected candidates shaped (K, H, A), got {actions.shape}.")
    return [env_adapter.rollout_from_state(state_snapshot, candidate) for candidate in actions]
