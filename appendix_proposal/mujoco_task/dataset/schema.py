"""Dataset archive schema and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from mujoco_task.config import ACTION_DIM, OBS_DIM


@dataclass(frozen=True)
class ChunkBatch:
    obs_now: np.ndarray
    future_actions: np.ndarray
    future_next_obs: np.ndarray
    mask: np.ndarray
    tilt_cond: np.ndarray


@dataclass(frozen=True)
class SceneManifest:
    split: str
    scene_id: np.ndarray
    family_id: np.ndarray
    scene_index: np.ndarray
    seed: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: np.ndarray
    lateral_tilt: np.ndarray
    longitudinal_tilt: np.ndarray
    left_success_count: np.ndarray
    right_success_count: np.ndarray
    left_available: np.ndarray
    right_available: np.ndarray
    expert_episode_ids: np.ndarray
    perturbed_episode_id: np.ndarray

    @property
    def num_scenes(self) -> int:
        return int(self.scene_id.shape[0])

    def validate(self) -> None:
        expected = self.num_scenes
        for name, value in (
            ("family_id", self.family_id),
            ("scene_index", self.scene_index),
            ("seed", self.seed),
            ("lateral_tilt", self.lateral_tilt),
            ("longitudinal_tilt", self.longitudinal_tilt),
            ("left_success_count", self.left_success_count),
            ("right_success_count", self.right_success_count),
            ("left_available", self.left_available),
            ("right_available", self.right_available),
            ("perturbed_episode_id", self.perturbed_episode_id),
        ):
            if value.shape[0] != expected:
                raise ValueError(f"{name} length does not match scene count.")
        if self.start.shape != (expected, 2):
            raise ValueError("start must be shaped (num_scenes, 2).")
        if self.goal.shape != (expected, 2):
            raise ValueError("goal must be shaped (num_scenes, 2).")
        if self.obstacle_center.shape != (expected, 2):
            raise ValueError("obstacle_center must be shaped (num_scenes, 2).")
        if self.expert_episode_ids.shape != (expected, 2):
            raise ValueError("expert_episode_ids must be shaped (num_scenes, 2).")
        total_success = self.left_success_count + self.right_success_count
        if not np.all(total_success >= 2):
            raise ValueError("Every stored scene must have at least two successful expert episodes.")

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "scenes.npz",
            split=np.asarray([self.split] * self.num_scenes),
            scene_id=self.scene_id,
            family_id=self.family_id,
            scene_index=self.scene_index,
            seed=self.seed,
            start=self.start,
            goal=self.goal,
            obstacle_center=self.obstacle_center,
            obstacle_radius=self.obstacle_radius,
            lateral_tilt=self.lateral_tilt,
            longitudinal_tilt=self.longitudinal_tilt,
            left_success_count=self.left_success_count,
            right_success_count=self.right_success_count,
            left_available=self.left_available,
            right_available=self.right_available,
            expert_episode_ids=self.expert_episode_ids,
            perturbed_episode_id=self.perturbed_episode_id,
        )

    @classmethod
    def load(cls, root: Path, split: str) -> "SceneManifest":
        payload = np.load(root / "scenes.npz")
        manifest = cls(
            split=split,
            scene_id=np.asarray(payload["scene_id"], dtype=np.int64),
            family_id=np.asarray(payload["family_id"], dtype=np.int64),
            scene_index=np.asarray(payload["scene_index"], dtype=np.int64),
            seed=np.asarray(payload["seed"], dtype=np.int64),
            start=np.asarray(payload["start"], dtype=np.float32),
            goal=np.asarray(payload["goal"], dtype=np.float32),
            obstacle_center=np.asarray(payload["obstacle_center"], dtype=np.float32),
            obstacle_radius=np.asarray(payload["obstacle_radius"], dtype=np.float32),
            lateral_tilt=np.asarray(payload["lateral_tilt"], dtype=np.float32),
            longitudinal_tilt=np.asarray(payload["longitudinal_tilt"], dtype=np.float32),
            left_success_count=np.asarray(payload["left_success_count"], dtype=np.int64),
            right_success_count=np.asarray(payload["right_success_count"], dtype=np.int64),
            left_available=np.asarray(payload["left_available"], dtype=bool),
            right_available=np.asarray(payload["right_available"], dtype=bool),
            expert_episode_ids=np.asarray(payload["expert_episode_ids"], dtype=np.int64),
            perturbed_episode_id=np.asarray(payload["perturbed_episode_id"], dtype=np.int64),
        )
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class EpisodeArchive:
    split: str
    episode_id: np.ndarray
    scene_id: np.ndarray
    offset: np.ndarray
    length: np.ndarray
    lateral_tilt: np.ndarray
    longitudinal_tilt: np.ndarray
    goal: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: np.ndarray
    success: np.ndarray
    collision: np.ndarray
    timeout: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    meta: dict[str, object]

    @property
    def num_episodes(self) -> int:
        return int(self.episode_id.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.obs.shape[0])

    def episode_slice(self, index: int) -> slice:
        start = int(self.offset[index])
        return slice(start, start + int(self.length[index]))

    def validate(self) -> None:
        if self.obs.ndim != 2 or self.obs.shape[1] != OBS_DIM:
            raise ValueError(f"Expected obs shape (*, {OBS_DIM}), got {self.obs.shape}.")
        if self.next_obs.ndim != 2 or self.next_obs.shape[1] != OBS_DIM:
            raise ValueError(f"Expected next_obs shape (*, {OBS_DIM}), got {self.next_obs.shape}.")
        if self.action.ndim != 2 or self.action.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected action shape (*, {ACTION_DIM}), got {self.action.shape}.")
        if self.done.shape[0] != self.obs.shape[0]:
            raise ValueError("done length does not match step arrays.")
        if not np.all(self.length > 0):
            raise ValueError("Episode lengths must be positive.")
        if int(self.offset[0]) != 0:
            raise ValueError("First episode offset must be zero.")
        if int(np.sum(self.length)) != self.obs.shape[0]:
            raise ValueError("Episode lengths do not sum to total steps.")
        for index in range(self.num_episodes):
            terminal_flags = self.done[self.episode_slice(index)]
            if terminal_flags.shape[0] == 0:
                raise ValueError("Episode may not be empty.")
            if not bool(terminal_flags[-1]):
                raise ValueError("Final step of each episode must be done=True.")
            if np.any(terminal_flags[:-1]):
                raise ValueError("Only the last step of each episode may be done=True.")
        outcomes = np.stack([self.success, self.collision, self.timeout], axis=-1).astype(np.int32)
        if not np.all(np.sum(outcomes, axis=-1) == 1):
            raise ValueError("Each episode must have exactly one outcome among success/collision/timeout.")

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "episodes.npz",
            episode_id=self.episode_id,
            split=np.asarray([self.split] * self.num_episodes),
            scene_id=self.scene_id,
            offset=self.offset,
            length=self.length,
            lateral_tilt=self.lateral_tilt,
            longitudinal_tilt=self.longitudinal_tilt,
            goal=self.goal,
            obstacle_center=self.obstacle_center,
            obstacle_radius=self.obstacle_radius,
            success=self.success,
            collision=self.collision,
            timeout=self.timeout,
        )
        np.savez_compressed(root / "steps.npz", obs=self.obs, action=self.action, next_obs=self.next_obs, done=self.done)
        (root / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, root: Path, split: str) -> "EpisodeArchive":
        episodes = np.load(root / "episodes.npz")
        steps = np.load(root / "steps.npz")
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        archive = cls(
            split=split,
            episode_id=np.asarray(episodes["episode_id"], dtype=np.int64),
            scene_id=np.asarray(episodes["scene_id"], dtype=np.int64),
            offset=np.asarray(episodes["offset"], dtype=np.int64),
            length=np.asarray(episodes["length"], dtype=np.int64),
            lateral_tilt=np.asarray(episodes["lateral_tilt"], dtype=np.float32),
            longitudinal_tilt=np.asarray(episodes["longitudinal_tilt"], dtype=np.float32),
            goal=np.asarray(episodes["goal"], dtype=np.float32),
            obstacle_center=np.asarray(episodes["obstacle_center"], dtype=np.float32),
            obstacle_radius=np.asarray(episodes["obstacle_radius"], dtype=np.float32),
            success=np.asarray(episodes["success"], dtype=bool),
            collision=np.asarray(episodes["collision"], dtype=bool),
            timeout=np.asarray(episodes["timeout"], dtype=bool),
            obs=np.asarray(steps["obs"], dtype=np.float32),
            action=np.asarray(steps["action"], dtype=np.float32),
            next_obs=np.asarray(steps["next_obs"], dtype=np.float32),
            done=np.asarray(steps["done"], dtype=bool),
            meta=meta,
        )
        archive.validate()
        return archive


def subset_episode_archive(
    archive: EpisodeArchive,
    episode_indices: np.ndarray | list[int],
    *,
    split: str | None = None,
    meta_extra: dict[str, object] | None = None,
) -> EpisodeArchive:
    selected = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("episode_indices may not be empty.")
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []
    offsets: list[int] = []
    offset = 0
    for index in selected:
        offsets.append(offset)
        sl = archive.episode_slice(int(index))
        obs_rows.append(archive.obs[sl])
        action_rows.append(archive.action[sl])
        next_obs_rows.append(archive.next_obs[sl])
        done_rows.append(archive.done[sl])
        offset += int(archive.length[int(index)])
    meta = dict(archive.meta)
    if meta_extra is not None:
        meta.update(meta_extra)
    subset = EpisodeArchive(
        split=archive.split if split is None else str(split),
        episode_id=archive.episode_id[selected].copy(),
        scene_id=archive.scene_id[selected].copy(),
        offset=np.asarray(offsets, dtype=np.int64),
        length=archive.length[selected].copy(),
        lateral_tilt=archive.lateral_tilt[selected].copy(),
        longitudinal_tilt=archive.longitudinal_tilt[selected].copy(),
        goal=archive.goal[selected].copy(),
        obstacle_center=archive.obstacle_center[selected].copy(),
        obstacle_radius=archive.obstacle_radius[selected].copy(),
        success=archive.success[selected].copy(),
        collision=archive.collision[selected].copy(),
        timeout=archive.timeout[selected].copy(),
        obs=np.concatenate(obs_rows, axis=0).astype(np.float32),
        action=np.concatenate(action_rows, axis=0).astype(np.float32),
        next_obs=np.concatenate(next_obs_rows, axis=0).astype(np.float32),
        done=np.concatenate(done_rows, axis=0).astype(bool),
        meta=meta,
    )
    subset.validate()
    return subset
