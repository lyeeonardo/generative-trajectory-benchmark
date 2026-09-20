"""Hidden-tilt context utilities."""

from __future__ import annotations

import numpy as np

from envs.tilted_board_adapter import StateSnapshot
from mujoco_task.sim.scene import SceneSpec


def snapshot_with_tilt(snapshot: StateSnapshot, tilt: np.ndarray | list[float] | tuple[float, float]) -> StateSnapshot:
    """Return a copied snapshot with a different board tilt hypothesis."""

    scene = snapshot.scene
    tilt_arr = np.asarray(tilt, dtype=np.float32).reshape(2)
    next_scene = SceneSpec(
        start=scene.start.copy(),
        goal=scene.goal.copy(),
        obstacle_center=scene.obstacle_center.copy(),
        obstacle_radius=float(scene.obstacle_radius),
        lateral_tilt=float(tilt_arr[0]),
        longitudinal_tilt=float(tilt_arr[1]),
        seed=int(scene.seed),
    )
    return StateSnapshot(scene=next_scene, state=snapshot.state.copy(), physics_backend=snapshot.physics_backend)


def mask_observed_tilt(obs: np.ndarray, replacement: np.ndarray | list[float] | tuple[float, float] | None = None) -> np.ndarray:
    """Copy an observation and replace its true tilt fields."""

    masked = np.asarray(obs, dtype=np.float32).copy()
    fill = np.zeros((2,), dtype=np.float32) if replacement is None else np.asarray(replacement, dtype=np.float32).reshape(2)
    masked[12:14] = fill
    return masked
