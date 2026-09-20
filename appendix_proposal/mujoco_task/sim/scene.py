"""Scene specifications and deterministic family generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mujoco_task.config import Preset


@dataclass(frozen=True)
class SceneSpec:
    start: np.ndarray
    goal: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: float
    lateral_tilt: float
    longitudinal_tilt: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", np.asarray(self.start, dtype=np.float32))
        object.__setattr__(self, "goal", np.asarray(self.goal, dtype=np.float32))
        object.__setattr__(self, "obstacle_center", np.asarray(self.obstacle_center, dtype=np.float32))
        object.__setattr__(self, "obstacle_radius", float(self.obstacle_radius))
        object.__setattr__(self, "lateral_tilt", float(self.lateral_tilt))
        object.__setattr__(self, "longitudinal_tilt", float(self.longitudinal_tilt))
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class SceneFamily:
    family_id: int
    start: np.ndarray
    goal: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", np.asarray(self.start, dtype=np.float32))
        object.__setattr__(self, "goal", np.asarray(self.goal, dtype=np.float32))
        object.__setattr__(self, "obstacle_center", np.asarray(self.obstacle_center, dtype=np.float32))
        object.__setattr__(self, "obstacle_radius", float(self.obstacle_radius))


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def straight_path_blocked(family: SceneFamily, preset: Preset) -> bool:
    margin = preset.scene.path_block_margin
    distance = _point_segment_distance(family.obstacle_center, family.start, family.goal)
    return bool(distance <= family.obstacle_radius + preset.sim.ball_radius + margin)


def sample_scene_family(rng: np.random.Generator, preset: Preset, family_id: int) -> SceneFamily:
    start_x = rng.uniform(*preset.scene.start_x_range)
    goal_x = rng.uniform(*preset.scene.goal_x_range)
    start_y = rng.uniform(*preset.scene.start_y_range)
    goal_y = rng.uniform(*preset.scene.goal_y_range)
    return SceneFamily(
        family_id=family_id,
        start=np.array([start_x, start_y], dtype=np.float32),
        goal=np.array([goal_x, goal_y], dtype=np.float32),
        obstacle_center=np.array(preset.scene.obstacle_center, dtype=np.float32),
        obstacle_radius=preset.sim.obstacle_radius,
    )


def expand_family(family: SceneFamily, preset: Preset) -> list[SceneSpec]:
    scenes: list[SceneSpec] = []
    for lateral_index, lateral_tilt in enumerate(preset.scene.lateral_tilts):
        for longitudinal_index, longitudinal_tilt in enumerate(preset.scene.longitudinal_tilts):
            seed = family.family_id * 100 + lateral_index * 10 + longitudinal_index
            scenes.append(
                SceneSpec(
                    start=family.start,
                    goal=family.goal,
                    obstacle_center=family.obstacle_center,
                    obstacle_radius=family.obstacle_radius,
                    lateral_tilt=float(lateral_tilt),
                    longitudinal_tilt=float(longitudinal_tilt),
                    seed=seed,
                )
            )
    return scenes


def family_route_midpoint(family: SceneFamily, preset: Preset, side: str) -> np.ndarray:
    sign = -1.0 if side == "left" else 1.0
    mid_x = sign * (family.obstacle_radius + preset.scene.route_midpoint_margin)
    return np.array([mid_x, 0.0], dtype=np.float32)


def quadratic_reference(
    current_ball_pos: np.ndarray,
    scene: SceneSpec,
    preset: Preset,
    side: str,
    horizon: int,
) -> np.ndarray:
    current = np.asarray(current_ball_pos, dtype=np.float32)
    family = SceneFamily(
        family_id=scene.seed,
        start=scene.start,
        goal=scene.goal,
        obstacle_center=scene.obstacle_center,
        obstacle_radius=scene.obstacle_radius,
    )
    side_mid = family_route_midpoint(family, preset, side)
    if current[1] >= 0.15:
        ts = np.linspace(0.0, 1.0, horizon + 1, dtype=np.float32)[1:]
        return np.stack([(1.0 - t) * current + t * scene.goal for t in ts], axis=0)
    lower_y = min(-0.28, 0.5 * (current[1] + scene.obstacle_center[1]))
    upper_y = max(0.28, 0.5 * (scene.goal[1] + scene.obstacle_center[1]))
    return_x = 0.45 * side_mid[0]
    if current[1] >= -0.15:
        anchors = np.stack([current, np.array([return_x, upper_y], dtype=np.float32), scene.goal.astype(np.float32)], axis=0)
    else:
        anchors = np.stack(
            [
                current,
                np.array([side_mid[0], current[1]], dtype=np.float32),
                np.array([side_mid[0], lower_y], dtype=np.float32),
                np.array([return_x, upper_y], dtype=np.float32),
                scene.goal.astype(np.float32),
            ],
            axis=0,
        )
    segment_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=-1)
    total = float(np.sum(segment_lengths))
    if total <= 1e-8:
        return np.repeat(scene.goal[None, :], horizon, axis=0).astype(np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths / total)])
    ts = np.linspace(0.0, 1.0, horizon + 1, dtype=np.float32)[1:]
    points = []
    for t in ts:
        segment = int(np.searchsorted(cumulative[1:], t, side="right"))
        segment = min(segment, len(segment_lengths) - 1)
        local_start = cumulative[segment]
        local_end = cumulative[segment + 1]
        alpha = 0.0 if local_end <= local_start else float((t - local_start) / (local_end - local_start))
        points.append(((1.0 - alpha) * anchors[segment] + alpha * anchors[segment + 1]).astype(np.float32))
    return np.stack(points, axis=0)


def wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))
