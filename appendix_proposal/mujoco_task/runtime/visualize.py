"""Compact GIF rendering for trajectory artifacts."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from mujoco_task.config import Preset
from mujoco_task.dataset.schema import EpisodeArchive
from mujoco_task.sim.scene import SceneSpec


def _frame_image(scene: SceneSpec, trajectory: np.ndarray, step_index: int, preset: Preset, *, title: str, subtitle: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.2, 5.2), dpi=100)
    ax.set_xlim(*preset.sim.workspace_x)
    ax.set_ylim(*preset.sim.workspace_y)
    ax.set_aspect("equal")
    ax.set_facecolor("#f6f2e8")
    ax.grid(True, color="#d9d1c2", linewidth=0.6, alpha=0.6)
    ax.add_patch(Circle(scene.obstacle_center, scene.obstacle_radius, facecolor="#625b55", edgecolor="#3c3835", linewidth=2.0))
    ax.add_patch(Circle(scene.goal, preset.sim.goal_radius, facecolor="none", edgecolor="#2c7a4b", linewidth=2.5))
    ax.scatter(scene.start[0], scene.start[1], s=70, c="#1f6feb", marker="o", label="start")
    ax.scatter(scene.goal[0], scene.goal[1], s=70, c="#2c7a4b", marker="*", label="goal")
    current = trajectory[min(step_index, trajectory.shape[0] - 1)]
    past = trajectory[: step_index + 1]
    ax.plot(past[:, 0], past[:, 1], color="#d97706", linewidth=2.2)
    ax.scatter(current[0], current[1], s=55, c="#b42318", zorder=5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(
        0.02,
        0.98,
        subtitle,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d0c7b8", "alpha": 0.88, "pad": 4},
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image


def _downsample_indices(length: int, max_frames: int = 48) -> np.ndarray:
    if length <= max_frames:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_frames, dtype=np.int64)


def render_trajectory_gif(
    scene: SceneSpec,
    trajectory: np.ndarray,
    preset: Preset,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    points = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    frames = [_frame_image(scene, points, int(index), preset, title=title, subtitle=subtitle) for index in _downsample_indices(points.shape[0])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.08, loop=0)


def select_best_training_episode(archive: EpisodeArchive) -> int:
    indices = np.arange(archive.num_episodes, dtype=np.int64)
    if np.any(archive.success):
        indices = indices[archive.success]
    return int(
        min(
            indices.tolist(),
            key=lambda index: (
                float(np.linalg.norm(archive.next_obs[archive.episode_slice(index)][-1, :2] - archive.goal[index])),
                int(archive.length[index]),
            ),
        )
    )


def archive_episode_trajectory(archive: EpisodeArchive, episode_index: int) -> tuple[SceneSpec, np.ndarray, str]:
    sl = archive.episode_slice(episode_index)
    trajectory = np.concatenate([archive.obs[sl.start : sl.start + 1, :2], archive.next_obs[sl, :2]], axis=0).astype(np.float32)
    scene = SceneSpec(
        start=archive.obs[sl.start, :2],
        goal=archive.goal[episode_index],
        obstacle_center=archive.obstacle_center[episode_index],
        obstacle_radius=float(archive.obstacle_radius[episode_index]),
        lateral_tilt=float(archive.lateral_tilt[episode_index]),
        longitudinal_tilt=float(archive.longitudinal_tilt[episode_index]),
        seed=int(archive.scene_id[episode_index]),
    )
    outcome = "success" if bool(archive.success[episode_index]) else "collision" if bool(archive.collision[episode_index]) else "timeout"
    return scene, trajectory, outcome
