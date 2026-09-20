"""Shared expected-free-energy-style rollout scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


DEFAULT_SCORING_CONFIG: dict[str, float | tuple[float, float]] = {
    "w_goal_terminal": 12.0,
    "w_goal_path": 1.0,
    "w_progress": 5.0,
    "w_collision": 150.0,
    "w_fall_out": 150.0,
    "w_clearance": 15.0,
    "w_boundary": 8.0,
    "w_smooth": 0.05,
    "w_control": 0.08,
    "w_timeout": 25.0,
    "w_dyn": 100.0,
    "w_epistemic": 0.0,
    "clearance_margin": 0.04,
    "workspace_x": (-0.8, 0.8),
    "workspace_y": (-1.0, 1.0),
}


@dataclass(frozen=True)
class ScoreBreakdown:
    G_total: float
    G_preference: float
    G_safety: float
    G_smoothness: float
    G_control: float
    G_dynamics: float
    G_epistemic: float
    G_terminal: float
    G_progress: float
    invalid_penalty: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _cfg(config: dict[str, Any] | None, key: str) -> Any:
    if config is None:
        return DEFAULT_SCORING_CONFIG[key]
    if key in config:
        return config[key]
    scoring = config.get("scoring") if isinstance(config, dict) else None
    if isinstance(scoring, dict) and key in scoring:
        return scoring[key]
    return DEFAULT_SCORING_CONFIG[key]


def policy_posterior(G: np.ndarray, gamma_t: float, log_E: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(G, dtype=np.float64).reshape(-1)
    prior = np.zeros_like(values) if log_E is None else np.asarray(log_E, dtype=np.float64).reshape(values.shape)
    logits = -float(gamma_t) * values + prior
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    denom = np.sum(weights)
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=np.float32)
    return (weights / denom).astype(np.float32)


class AIFScorer:
    """Shared scorer used for every Stage 1 proposal source."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def score_rollout(self, rollout, belief, config: dict[str, Any] | None = None) -> ScoreBreakdown:
        cfg = self.config if config is None else {**self.config, **dict(config)}
        observations = np.asarray(rollout.observations, dtype=np.float32)
        actions = np.asarray(rollout.actions, dtype=np.float32)
        finite = bool(np.all(np.isfinite(observations)) and np.all(np.isfinite(actions)))
        ball = observations[:, :2]
        goal = observations[-1, 7:9]
        terminal_goal_distance = float(np.linalg.norm(ball[-1] - goal))
        path_goal_distance = float(np.mean(np.linalg.norm(ball - goal[None, :], axis=1)))
        start_distance = float(np.linalg.norm(ball[0] - goal))
        final_distance = float(np.linalg.norm(ball[-1] - goal))
        progress = start_distance - final_distance
        negative_progress_penalty = float(max(0.0, -progress))
        collision_penalty = 1.0 if bool(rollout.collision) else 0.0
        fall_out_penalty = 1.0 if bool(getattr(rollout, "fall_out", False)) else 0.0
        timeout_penalty = 1.0 if bool(rollout.timeout) else 0.0
        clearance_shortfall = max(0.0, float(_cfg(cfg, "clearance_margin")) - float(rollout.minimum_obstacle_clearance))
        clearance_penalty = clearance_shortfall * clearance_shortfall
        workspace_x = tuple(_cfg(cfg, "workspace_x"))
        workspace_y = tuple(_cfg(cfg, "workspace_y"))
        boundary_penalty = float(
            np.mean(np.maximum(0.0, workspace_x[0] - ball[:, 0]) ** 2)
            + np.mean(np.maximum(0.0, ball[:, 0] - workspace_x[1]) ** 2)
            + np.mean(np.maximum(0.0, workspace_y[0] - ball[:, 1]) ** 2)
            + np.mean(np.maximum(0.0, ball[:, 1] - workspace_y[1]) ** 2)
        )
        action_smoothness = float(rollout.action_smoothness)
        action_magnitude = float(np.mean(np.sum(actions * actions, axis=-1))) if actions.size else 0.0
        dynamics_validity_penalty = 0.0 if finite else 1.0
        epistemic_value = float(getattr(rollout, "raw_simulator_info", {}).get("epistemic_value", 0.0))

        G_terminal = float(_cfg(cfg, "w_goal_terminal")) * terminal_goal_distance
        G_path = float(_cfg(cfg, "w_goal_path")) * path_goal_distance
        G_progress = float(_cfg(cfg, "w_progress")) * negative_progress_penalty
        G_preference = G_terminal + G_path + G_progress
        G_safety = (
            float(_cfg(cfg, "w_collision")) * collision_penalty
            + float(_cfg(cfg, "w_fall_out")) * fall_out_penalty
            + float(_cfg(cfg, "w_clearance")) * clearance_penalty
            + float(_cfg(cfg, "w_boundary")) * boundary_penalty
            + float(_cfg(cfg, "w_timeout")) * timeout_penalty
        )
        G_smoothness = float(_cfg(cfg, "w_smooth")) * action_smoothness
        G_control = float(_cfg(cfg, "w_control")) * action_magnitude
        G_dynamics = float(_cfg(cfg, "w_dyn")) * dynamics_validity_penalty
        G_epistemic = float(_cfg(cfg, "w_epistemic")) * epistemic_value
        invalid_penalty = 0.0 if finite else float(_cfg(cfg, "w_dyn"))
        G_total = G_preference + G_safety + G_smoothness + G_control + G_dynamics - G_epistemic
        return ScoreBreakdown(
            G_total=float(G_total),
            G_preference=float(G_preference),
            G_safety=float(G_safety),
            G_smoothness=float(G_smoothness),
            G_control=float(G_control),
            G_dynamics=float(G_dynamics),
            G_epistemic=float(G_epistemic),
            G_terminal=float(G_terminal),
            G_progress=float(G_progress),
            invalid_penalty=float(invalid_penalty),
            diagnostics={
                "terminal_goal_distance": terminal_goal_distance,
                "path_goal_distance": path_goal_distance,
                "progress": float(progress),
                "negative_progress_penalty": negative_progress_penalty,
                "collision_penalty": collision_penalty,
                "fall_out_penalty": fall_out_penalty,
                "clearance_penalty": clearance_penalty,
                "boundary_penalty": boundary_penalty,
                "timeout_penalty": timeout_penalty,
                "action_smoothness": action_smoothness,
                "action_magnitude": action_magnitude,
                "minimum_obstacle_clearance": float(rollout.minimum_obstacle_clearance),
                "route_label": rollout.route_label,
            },
        )

    def score_batch(self, rollouts, belief, config: dict[str, Any] | None = None) -> list[ScoreBreakdown]:
        return [self.score_rollout(rollout, belief, config=config) for rollout in rollouts]
