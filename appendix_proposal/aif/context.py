"""Shared Generator generator context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

OBS_DIM = 14
ACTION_DIM = 3


@dataclass(frozen=True)
class GeneratorContext:
    obs_t: np.ndarray
    obs_history: np.ndarray
    action_history: np.ndarray
    goal: np.ndarray
    obstacle: dict[str, Any]
    tilt: np.ndarray
    belief_summary: np.ndarray
    time_fraction: float
    scene_context: dict[str, Any]
    normalized_vector: np.ndarray
    raw_vector: np.ndarray = field(repr=False)

    @classmethod
    def from_belief(
        cls,
        belief,
        scene_context: dict[str, Any] | None = None,
        *,
        normalizer: object | None = None,
        history_len: int = 4,
        route_cue: float | None = None,
    ) -> "GeneratorContext":
        obs_t = np.asarray(belief.obs, dtype=np.float32).reshape(OBS_DIM).copy()
        scene_context = dict(scene_context or {})
        if bool(getattr(belief, "hide_tilt", False)):
            scene_context["tilt"] = obs_t[12:14].copy()
        obs_history = _obs_history_from_belief(belief, history_len)
        action_history = _action_history_from_belief(belief, history_len)
        uncertainty = belief.uncertainty_summary()
        belief_summary = np.asarray(
            [
                uncertainty.get("physical_state_entropy", 0.0),
                uncertainty.get("goal_entropy", 0.0),
                uncertainty.get("dynamics_context_entropy", 0.0),
                uncertainty.get("regime_entropy", 0.0),
                uncertainty.get("parameter_entropy", 0.0),
            ],
            dtype=np.float32,
        )
        time_fraction = float(np.clip(belief.time_step / max(belief.max_steps, 1), 0.0, 1.0))
        goal = obs_t[7:9].copy()
        obstacle_center = obs_t[9:11].copy()
        obstacle_radius = float(obs_t[11])
        tilt = obs_t[12:14].copy()
        raw_vector = build_generator_context_vector(
            obs_t=obs_t,
            obs_history=obs_history,
            action_history=action_history,
            belief_summary=belief_summary,
            time_fraction=time_fraction,
            route_cue=route_cue,
        )
        if normalizer is not None and hasattr(normalizer, "normalize_context"):
            normalized_vector = normalizer.normalize_context(raw_vector)
        elif normalizer is not None and hasattr(normalizer, "transform"):
            normalized_vector = normalizer.transform(raw_vector)
        else:
            normalized_vector = raw_vector.copy()
        return cls(
            obs_t=obs_t,
            obs_history=obs_history,
            action_history=action_history,
            goal=goal,
            obstacle={"center": obstacle_center, "radius": obstacle_radius},
            tilt=tilt,
            belief_summary=belief_summary,
            time_fraction=time_fraction,
            scene_context=scene_context,
            normalized_vector=np.asarray(normalized_vector, dtype=np.float32).reshape(-1),
            raw_vector=raw_vector,
        )

    @property
    def context_dim(self) -> int:
        return int(self.normalized_vector.shape[0])


def build_generator_context_vector(
    *,
    obs_t: np.ndarray,
    obs_history: np.ndarray,
    action_history: np.ndarray,
    belief_summary: np.ndarray,
    time_fraction: float,
    route_cue: float | None = None,
) -> np.ndarray:
    obs = np.asarray(obs_t, dtype=np.float32).reshape(OBS_DIM)
    obs_hist = np.asarray(obs_history, dtype=np.float32).reshape(-1, OBS_DIM)
    act_hist = np.asarray(action_history, dtype=np.float32).reshape(obs_hist.shape[0], ACTION_DIM)
    ball_velocity_history = obs_hist[:, 2:4].reshape(-1)
    parts = [
        obs,
        obs[7:9],
        obs[9:12],
        obs[12:14],
        act_hist.reshape(-1),
        ball_velocity_history,
        np.asarray([float(time_fraction)], dtype=np.float32),
        np.asarray(belief_summary, dtype=np.float32).reshape(-1),
    ]
    if route_cue is not None:
        parts.append(np.asarray([float(route_cue)], dtype=np.float32))
    vector = np.concatenate(parts, axis=0).astype(np.float32)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Generator context contains non-finite values.")
    return vector


def _obs_history_from_belief(belief, history_len: int) -> np.ndarray:
    history_len = max(int(history_len), 1)
    previous = [np.asarray(row["obs"], dtype=np.float32).reshape(OBS_DIM) for row in belief.history_buffer[-(history_len - 1) :]]
    previous.append(np.asarray(belief.obs, dtype=np.float32).reshape(OBS_DIM))
    pad_value = previous[0] if previous else np.asarray(belief.obs, dtype=np.float32).reshape(OBS_DIM)
    while len(previous) < history_len:
        previous.insert(0, pad_value.copy())
    return np.asarray(previous[-history_len:], dtype=np.float32)


def _action_history_from_belief(belief, history_len: int) -> np.ndarray:
    history_len = max(int(history_len), 1)
    actions = [np.asarray(row["action"], dtype=np.float32).reshape(ACTION_DIM) for row in belief.history_buffer[-history_len:]]
    while len(actions) < history_len:
        actions.insert(0, np.zeros(ACTION_DIM, dtype=np.float32))
    return np.asarray(actions[-history_len:], dtype=np.float32)


def generator_context_dim(history_len: int = 4, *, include_route_cue: bool = False) -> int:
    base = OBS_DIM + 2 + 3 + 2 + int(history_len) * ACTION_DIM + int(history_len) * 2 + 1 + 5
    return base + (1 if include_route_cue else 0)
