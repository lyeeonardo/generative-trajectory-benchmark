"""Stage 1 belief state: fully observed physics with future hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.archive_dataset import CONTEXT_DIM, build_context_vector


@dataclass
class BeliefState:
    obs: np.ndarray
    physical_state_known: bool
    goal_belief: np.ndarray
    dynamics_context_belief: np.ndarray
    regime_belief: str = "known_nominal"
    parameter_belief: dict[str, float] = field(default_factory=dict)
    history_buffer: list[dict[str, Any]] = field(default_factory=list)
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    time_step: int = 0
    max_steps: int = 1
    hide_tilt: bool = False
    tilt_replacement: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    @classmethod
    def from_observation(cls, obs: np.ndarray, env_info: dict[str, Any], config: dict[str, Any] | None = None) -> "BeliefState":
        cfg = dict(config or {})
        observation = np.asarray(obs, dtype=np.float32).copy()
        hide_tilt = bool(cfg.get("hidden_tilt_enabled", cfg.get("mask_tilt", False))) and not bool(cfg.get("oracle_tilt", False))
        tilt_replacement = np.asarray(cfg.get("tilt_replacement", [0.0, 0.0]), dtype=np.float32).reshape(2)
        if hide_tilt:
            observation[12:14] = tilt_replacement
        max_steps = int(env_info.get("max_steps", 1))
        time_step = int(env_info.get("step", 0))
        return cls(
            obs=observation,
            physical_state_known=True,
            goal_belief=observation[7:9].copy(),
            dynamics_context_belief=observation[12:14].copy(),
            parameter_belief={"psi_fixed": 1.0},
            previous_action=np.zeros(3, dtype=np.float32),
            time_step=time_step,
            max_steps=max(max_steps, 1),
            hide_tilt=hide_tilt,
            tilt_replacement=tilt_replacement,
        )

    def update_after_observation(self, obs: np.ndarray, action: np.ndarray, info: dict[str, Any]) -> None:
        self.history_buffer.append(
            {
                "obs": self.obs.copy(),
                "action": np.asarray(action, dtype=np.float32).copy(),
                "info": dict(info),
            }
        )
        next_obs = np.asarray(obs, dtype=np.float32).copy()
        if self.hide_tilt:
            next_obs[12:14] = self.tilt_replacement
        self.obs = next_obs
        self.previous_action = np.asarray(action, dtype=np.float32).reshape(3).copy()
        self.goal_belief = self.obs[7:9].copy()
        self.dynamics_context_belief = self.obs[12:14].copy()
        self.time_step = int(info.get("step", self.time_step + 1))
        self.max_steps = int(info.get("max_steps", self.max_steps))

    def to_context_vector(self) -> np.ndarray:
        progress = float(self.time_step / max(self.max_steps, 1))
        context = build_context_vector(self.obs, prev_action=self.previous_action, progress=progress)
        if context.shape[0] != CONTEXT_DIM:
            raise ValueError(f"Expected context dim {CONTEXT_DIM}, got {context.shape[0]}.")
        return context


    def to_generator_context(
        self,
        scene_context: dict[str, Any] | None = None,
        *,
        normalizer: object | None = None,
        history_len: int = 4,
        route_cue: float | None = None,
    ):
        from aif.context import GeneratorContext

        return GeneratorContext.from_belief(
            self,
            scene_context,
            normalizer=normalizer,
            history_len=history_len,
            route_cue=route_cue,
        )

    def uncertainty_summary(self) -> dict[str, float]:
        return {
            "physical_state_entropy": 0.0 if self.physical_state_known else 1.0,
            "goal_entropy": 0.0,
            "dynamics_context_entropy": 0.0,
            "regime_entropy": 0.0,
            "parameter_entropy": 0.0,
        }
