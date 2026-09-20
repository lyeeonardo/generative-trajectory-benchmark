"""Mismatch signals used by Experiment reliability gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aif.hidden_tilt_belief import observation_features


@dataclass(frozen=True)
class MismatchSignals:
    delta_AB: float = 0.0
    delta_R: float = 0.0
    Delta_d: float = 0.0
    residual_mismatch: float = 0.0
    ood: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_AB": float(self.delta_AB),
            "delta_R": float(self.delta_R),
            "Delta_d": float(self.Delta_d),
            "residual_mismatch": float(self.residual_mismatch),
            "ood": bool(self.ood),
        }


def one_step_prediction_error(
    *,
    env_adapter,
    previous_snapshot,
    action: np.ndarray,
    observed_next_obs: np.ndarray,
    feature_indices: tuple[int, ...] | None = None,
    scale: float = 1.0,
) -> float:
    """Simulator-grounded one-step prediction error without mutating live env."""

    live_snapshot = env_adapter.clone_state()
    try:
        rollout = env_adapter.rollout_from_state(previous_snapshot, np.asarray(action, dtype=np.float32).reshape(1, -1))
        predicted = rollout.observations[min(1, rollout.observations.shape[0] - 1)]
        delta = observation_features(observed_next_obs, feature_indices) - observation_features(predicted, feature_indices)
        return float(np.sum(delta * delta) / max(float(scale), 1e-12))
    finally:
        env_adapter.restore_state(live_snapshot)


def generator_surprise(log_prob: float | None = None, *, proxy_distance: float | None = None) -> float:
    if log_prob is not None and np.isfinite(log_prob):
        return float(max(0.0, -float(log_prob)))
    if proxy_distance is not None and np.isfinite(proxy_distance):
        return float(max(0.0, float(proxy_distance)))
    return 0.0


def residual_mismatch(delta_ab_z: float, belief_kl_z: float, *, c_d: float = 0.5) -> float:
    return float(max(0.0, float(delta_ab_z) - float(c_d) * float(belief_kl_z)))


def make_mismatch_signals(
    *,
    delta_AB: float = 0.0,
    delta_R: float = 0.0,
    Delta_d: float = 0.0,
    ood: bool = False,
    c_d: float = 0.5,
) -> MismatchSignals:
    residual = residual_mismatch(delta_AB, Delta_d, c_d=c_d)
    return MismatchSignals(delta_AB=float(delta_AB), delta_R=float(delta_R), Delta_d=float(Delta_d), residual_mismatch=residual, ood=bool(ood))
