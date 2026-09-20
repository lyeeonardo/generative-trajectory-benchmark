"""Categorical belief over hidden tilted-board dynamics context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.tilted_board_adapter import StateSnapshot

from mujoco_task.sim.scene import SceneSpec


DEFAULT_TILT_GRID = np.asarray(
    [(lat, lon) for lat in np.deg2rad([-10.0, 0.0, 10.0]) for lon in np.deg2rad([0.0, 10.0, 20.0, 30.0])],
    dtype=np.float32,
)


def observation_features(obs: np.ndarray, feature_indices: tuple[int, ...] | None = None) -> np.ndarray:
    """Project observations to dynamic features, excluding static goal/obstacle fields."""

    observation = np.asarray(obs, dtype=np.float32).reshape(-1)
    indices = (0, 1, 2, 3, 4, 5, 6) if feature_indices is None else tuple(feature_indices)
    if not indices:
        return np.zeros((0,), dtype=np.float32)
    return observation[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def categorical_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _scene_with_tilt(snapshot: StateSnapshot, tilt: np.ndarray) -> StateSnapshot:
    scene = snapshot.scene
    next_scene = SceneSpec(
        start=scene.start.copy(),
        goal=scene.goal.copy(),
        obstacle_center=scene.obstacle_center.copy(),
        obstacle_radius=float(scene.obstacle_radius),
        lateral_tilt=float(tilt[0]),
        longitudinal_tilt=float(tilt[1]),
        seed=int(scene.seed),
    )
    return StateSnapshot(scene=next_scene, state=snapshot.state.copy(), physics_backend=snapshot.physics_backend)


@dataclass
class HiddenTiltBelief:
    tilt_grid: np.ndarray = field(default_factory=lambda: DEFAULT_TILT_GRID.copy())
    probabilities: np.ndarray = field(default_factory=lambda: np.full((DEFAULT_TILT_GRID.shape[0],), 1.0 / DEFAULT_TILT_GRID.shape[0], dtype=np.float32))
    sigma_d: float = 0.08
    epsilon_d: float = 0.01
    known_tilt: bool = False
    feature_indices: tuple[int, ...] | None = None
    last_predictive_errors: np.ndarray = field(default_factory=lambda: np.zeros((9,), dtype=np.float32))
    last_kl: float = 0.0
    collapse_warning: bool = False

    def __post_init__(self) -> None:
        self.tilt_grid = np.asarray(self.tilt_grid, dtype=np.float32).reshape(-1, 2)
        probs = np.asarray(self.probabilities, dtype=np.float32).reshape(-1)
        if probs.shape[0] != self.tilt_grid.shape[0]:
            raise ValueError("probabilities must match tilt_grid length.")
        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0.0:
            probs = np.full((self.tilt_grid.shape[0],), 1.0 / self.tilt_grid.shape[0], dtype=np.float32)
        else:
            probs = probs / total
        self.probabilities = probs.astype(np.float32)
        self.last_predictive_errors = np.zeros((self.tilt_grid.shape[0],), dtype=np.float32)

    @classmethod
    def known(cls, true_tilt: np.ndarray | list[float] | tuple[float, float], **kwargs: Any) -> "HiddenTiltBelief":
        grid = np.asarray(kwargs.pop("tilt_grid", DEFAULT_TILT_GRID), dtype=np.float32).reshape(-1, 2)
        tilt = np.asarray(true_tilt, dtype=np.float32).reshape(2)
        index = int(np.argmin(np.linalg.norm(grid - tilt[None, :], axis=1)))
        probabilities = np.zeros((grid.shape[0],), dtype=np.float32)
        probabilities[index] = 1.0
        return cls(tilt_grid=grid, probabilities=probabilities, known_tilt=True, epsilon_d=0.0, **kwargs)

    @classmethod
    def hidden_uniform(cls, **kwargs: Any) -> "HiddenTiltBelief":
        grid = np.asarray(kwargs.pop("tilt_grid", DEFAULT_TILT_GRID), dtype=np.float32).reshape(-1, 2)
        return cls(tilt_grid=grid, probabilities=np.full((grid.shape[0],), 1.0 / grid.shape[0], dtype=np.float32), known_tilt=False, **kwargs)

    @property
    def entropy(self) -> float:
        return categorical_entropy(self.probabilities)

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def map_tilt(self) -> np.ndarray:
        return self.tilt_grid[self.map_index].copy()

    def true_tilt_probability(self, true_tilt: np.ndarray | list[float] | tuple[float, float]) -> float:
        tilt = np.asarray(true_tilt, dtype=np.float32).reshape(2)
        index = int(np.argmin(np.linalg.norm(self.tilt_grid - tilt[None, :], axis=1)))
        return float(self.probabilities[index])

    def hypothesis_indices(self, mode: str = "exact_grid", m: int | None = None, seed: int | None = None) -> np.ndarray:
        mode = str(mode)
        count = self.tilt_grid.shape[0]
        if mode == "exact_grid":
            return np.arange(count, dtype=np.int64)
        selected = count if m is None else max(1, min(int(m), count))
        if mode == "top_m":
            return np.argsort(-self.probabilities)[:selected].astype(np.int64)
        if mode == "sample_m":
            rng = np.random.default_rng(seed)
            probs = self.probabilities / max(float(np.sum(self.probabilities)), 1e-12)
            return rng.choice(np.arange(count), size=selected, replace=False, p=probs).astype(np.int64)
        raise ValueError("mode must be exact_grid, top_m, or sample_m.")

    def update(
        self,
        *,
        env_adapter,
        previous_snapshot: StateSnapshot,
        action: np.ndarray,
        observed_next_obs: np.ndarray,
        true_tilt: np.ndarray | list[float] | tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Bayesian one-step tilt update using copied simulator predictions."""

        previous_probs = self.probabilities.astype(np.float64)
        if self.known_tilt:
            return self.diagnostics(true_tilt=true_tilt)
        live_snapshot = env_adapter.clone_state()
        observed_features = observation_features(observed_next_obs, self.feature_indices)
        errors = []
        try:
            for tilt in self.tilt_grid:
                hypothesis_snapshot = _scene_with_tilt(previous_snapshot, tilt)
                rollout = env_adapter.rollout_from_state(hypothesis_snapshot, np.asarray(action, dtype=np.float32).reshape(1, -1))
                predicted = rollout.observations[min(1, rollout.observations.shape[0] - 1)]
                predicted_features = observation_features(predicted, self.feature_indices)
                delta = observed_features - predicted_features
                errors.append(float(np.sum(delta * delta)))
        finally:
            env_adapter.restore_state(live_snapshot)
        errors_array = np.asarray(errors, dtype=np.float64)
        scale = max(float(self.sigma_d) ** 2, 1e-12)
        logits = -0.5 * errors_array / scale
        logits = logits - np.max(logits)
        likelihood = np.exp(logits)
        posterior = previous_probs * likelihood
        if not np.isfinite(np.sum(posterior)) or float(np.sum(posterior)) <= 0.0:
            posterior = previous_probs.copy()
        posterior = posterior / max(float(np.sum(posterior)), 1e-12)
        uniform = np.full_like(posterior, 1.0 / posterior.size)
        posterior = (1.0 - float(self.epsilon_d)) * posterior + float(self.epsilon_d) * uniform
        posterior = posterior / max(float(np.sum(posterior)), 1e-12)
        kl = float(np.sum(posterior * (np.log(np.clip(posterior, 1e-12, 1.0)) - np.log(np.clip(previous_probs, 1e-12, 1.0)))))
        self.probabilities = posterior.astype(np.float32)
        self.last_predictive_errors = errors_array.astype(np.float32)
        self.last_kl = kl
        self.collapse_warning = bool(self.entropy < 0.05 and float(np.min(errors_array)) > scale)
        return self.diagnostics(true_tilt=true_tilt)

    def diagnostics(self, true_tilt: np.ndarray | list[float] | tuple[float, float] | None = None) -> dict[str, Any]:
        true_prob = float("nan") if true_tilt is None else self.true_tilt_probability(true_tilt)
        true_arr = None if true_tilt is None else np.asarray(true_tilt, dtype=np.float32).reshape(2)
        map_accuracy = float("nan") if true_arr is None else float(np.allclose(self.map_tilt, true_arr, atol=1e-5))
        return {
            "tilt_entropy": self.entropy,
            "true_tilt_prob": true_prob,
            "MAP_tilt": self.map_tilt.tolist(),
            "MAP_tilt_accuracy": map_accuracy,
            "predictive_error_by_tilt": self.last_predictive_errors.tolist(),
            "belief_update_kl": float(self.last_kl),
            "posterior_collapse_warning": bool(self.collapse_warning),
            "tilt_probabilities": self.probabilities.tolist(),
        }


def time_to_correct_tilt(trace: list[dict[str, Any]], threshold: float = 0.5) -> int | None:
    for index, row in enumerate(trace):
        if float(row.get("true_tilt_prob", 0.0)) >= float(threshold):
            return int(index)
    return None
