"""Offline archive windows for CVAE-AIF proposal training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_task.config import ACTION_DIM, OBS_DIM, get_preset
from mujoco_task.dataset.schema import EpisodeArchive, SceneManifest


CONTEXT_DIM = OBS_DIM + 2 + 3 + 2 + ACTION_DIM + 1
OUTCOME_LABELS = ("success", "collision", "fall_out", "timeout")


def build_context_vector(
    obs: np.ndarray,
    *,
    prev_action: np.ndarray | None = None,
    progress: float = 0.0,
) -> np.ndarray:
    observation = np.asarray(obs, dtype=np.float32).reshape(OBS_DIM)
    previous = np.zeros(ACTION_DIM, dtype=np.float32) if prev_action is None else np.asarray(prev_action, dtype=np.float32)
    context = np.concatenate(
        [
            observation,
            observation[7:9],
            observation[9:12],
            observation[12:14],
            previous.reshape(ACTION_DIM),
            np.asarray([float(progress)], dtype=np.float32),
        ],
        axis=0,
    )
    if context.shape != (CONTEXT_DIM,):
        raise ValueError(f"Context shape mismatch: expected {(CONTEXT_DIM,)}, got {context.shape}.")
    return context.astype(np.float32)


def _route_label(obs_seq: np.ndarray) -> str:
    xy = np.asarray(obs_seq, dtype=np.float32)[:, :2]
    center_y = float(obs_seq[0, 10])
    band = np.abs(xy[:, 1] - center_y) <= 0.30
    near = xy[band] if np.any(band) else xy
    if np.min(near[:, 0]) < -0.12:
        return "left"
    if np.max(near[:, 0]) > 0.12:
        return "right"
    return "center"


def outcome_one_hot(label: str) -> np.ndarray:
    key = str(label)
    values = np.zeros((len(OUTCOME_LABELS),), dtype=np.float32)
    values[OUTCOME_LABELS.index(key) if key in OUTCOME_LABELS else 0] = 1.0
    return values


@dataclass(frozen=True)
class SplitFamilies:
    split: str
    families: set[int]


class ActionWindowDataset:
    """Overlapping fixed-horizon action windows from a local EpisodeArchive."""

    def __init__(
        self,
        archive_root: str | Path | None = None,
        *,
        preset: str = "smoke",
        split: str = "train",
        horizon: int | None = None,
        successful_only: bool = True,
        include_perturbed: bool = False,
    ) -> None:
        resolved = get_preset(preset)
        self.preset = resolved
        self.split = str(split)
        self.horizon = int(resolved.dataset.horizon if horizon is None else horizon)
        root = Path(archive_root) if archive_root is not None else resolved.paths.data_dir / resolved.name / self.split
        self.archive_root = root
        self.archive = EpisodeArchive.load(root, self.split)
        self.manifest = SceneManifest.load(root, self.split) if (root / "scenes.npz").exists() else None
        self.index: list[tuple[int, int]] = []
        for episode_index in range(self.archive.num_episodes):
            if successful_only and not bool(self.archive.success[episode_index]):
                if not include_perturbed:
                    continue
            length = int(self.archive.length[episode_index])
            for step in range(length):
                self.index.append((episode_index, step))
        if not self.index:
            raise ValueError(f"No action windows available in {root}.")

    def __len__(self) -> int:
        return len(self.index)

    @property
    def context_dim(self) -> int:
        return CONTEXT_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def split_families(self) -> SplitFamilies:
        if self.manifest is not None:
            families = set()
            for index in range(self.manifest.num_scenes):
                families.add(
                    _scene_family_signature(
                        self.manifest.start[index],
                        self.manifest.goal[index],
                        self.manifest.obstacle_center[index],
                        float(self.manifest.obstacle_radius[index]),
                    )
                )
        else:
            families = set(int(x) for x in np.asarray(self.archive.scene_id).tolist())
        return SplitFamilies(split=self.split, families=families)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_index, step_index = self.index[int(idx)]
        sl = self.archive.episode_slice(episode_index)
        absolute = int(sl.start + step_index)
        length = int(self.archive.length[episode_index])
        remaining = max(length - step_index, 0)
        future_len = min(self.horizon, remaining)

        action_seq = np.zeros((self.horizon, ACTION_DIM), dtype=np.float32)
        obs_seq = np.zeros((self.horizon + 1, OBS_DIM), dtype=np.float32)
        mask = np.zeros((self.horizon,), dtype=bool)
        obs_now = self.archive.obs[absolute].astype(np.float32)
        obs_seq[0] = obs_now
        if future_len > 0:
            action_seq[:future_len] = self.archive.action[absolute : absolute + future_len]
            obs_seq[1 : future_len + 1] = self.archive.next_obs[absolute : absolute + future_len]
            mask[:future_len] = True
        if future_len < self.horizon:
            fill = obs_seq[future_len] if future_len > 0 else obs_now
            obs_seq[future_len + 1 :] = fill
        prev_action = (
            np.zeros(ACTION_DIM, dtype=np.float32)
            if step_index == 0
            else self.archive.action[absolute - 1].astype(np.float32)
        )
        progress = float(step_index / max(length - 1, 1))
        context = build_context_vector(obs_now, prev_action=prev_action, progress=progress)
        scene_id = int(self.archive.scene_id[episode_index])
        family_id = self._family_for_scene(scene_id)
        item = {
            "context": context,
            "action_seq": action_seq,
            "obs_seq": obs_seq,
            "mask": mask,
            "scene_id": scene_id,
            "family_id": family_id,
            "tilt": obs_now[12:14].astype(np.float32),
            "route_label": _route_label(obs_seq[: future_len + 1]) if future_len > 0 else "center",
            "success_label": bool(self.archive.success[episode_index]),
        }
        for key in ("context", "action_seq", "obs_seq", "tilt"):
            if not np.all(np.isfinite(item[key])):
                raise ValueError(f"Non-finite values in {key}.")
        return item

    def _family_for_scene(self, scene_id: int) -> int:
        if self.manifest is None:
            return int(scene_id)
        matches = np.flatnonzero(np.asarray(self.manifest.scene_id, dtype=np.int64) == int(scene_id))
        if matches.size == 0:
            return int(scene_id)
        return int(self.manifest.family_id[int(matches[0])])


def assert_scene_family_disjoint(*datasets: ActionWindowDataset) -> None:
    seen: dict[int, str] = {}
    for dataset in datasets:
        split_families = dataset.split_families()
        for family in split_families.families:
            owner = seen.get(family)
            if owner is not None and owner != split_families.split:
                raise AssertionError(f"Scene family {family} appears in both {owner} and {split_families.split}.")
            seen[family] = split_families.split


def _scene_family_signature(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> int:
    values = np.concatenate(
        [
            np.asarray(start, dtype=np.float32).reshape(2),
            np.asarray(goal, dtype=np.float32).reshape(2),
            np.asarray(obstacle_center, dtype=np.float32).reshape(2),
            np.asarray([float(obstacle_radius)], dtype=np.float32),
        ]
    )
    rounded = tuple(float(x) for x in np.round(values, 5))
    return hash(rounded)



def _canonical_split(split: str) -> str:
    key = str(split)
    return {"validation": "val", "nominal_test": "test", "heldout_test": "test"}.get(key, key)


class GeneratorActionWindowDataset:
    """Generator shared action-window dataset for every trained proposal model."""

    def __init__(
        self,
        archive_root: str | Path | None = None,
        *,
        preset: str = "smoke",
        split: str = "train",
        horizon: int | None = None,
        history_len: int = 4,
        successful_only: bool = True,
        include_failures: bool = False,
        route_balance: bool = False,
        target_mode: str = "action",
    ) -> None:
        from aif.context import build_generator_context_vector

        del build_generator_context_vector
        resolved = get_preset(preset)
        self.preset = resolved
        self.requested_split = str(split)
        self.split = _canonical_split(split)
        self.horizon = int(resolved.dataset.horizon if horizon is None else horizon)
        self.history_len = int(history_len)
        self.target_mode = str(target_mode)
        root = Path(archive_root) if archive_root is not None else resolved.paths.data_dir / resolved.name / self.split
        self.archive_root = root
        self.archive = EpisodeArchive.load(root, self.split)
        self.manifest = SceneManifest.load(root, self.split) if (root / "scenes.npz").exists() else None
        base_index: list[tuple[int, int]] = []
        for episode_index in range(self.archive.num_episodes):
            if successful_only and not bool(self.archive.success[episode_index]) and not include_failures:
                continue
            length = int(self.archive.length[episode_index])
            for step in range(length):
                base_index.append((episode_index, step))
        if route_balance:
            base_index = self._route_balanced_index(base_index)
        self.index = base_index
        if not self.index:
            raise ValueError(f"No Generator action windows available in {root}.")
        first = self[0]
        self.context_dim = int(first["context"].shape[0])

    def __len__(self) -> int:
        return len(self.index)

    def window_keys(self) -> list[tuple[int, int]]:
        return [(int(self.archive.episode_id[episode_index]), int(step)) for episode_index, step in self.index]

    def split_families(self) -> SplitFamilies:
        if self.manifest is not None:
            families = set()
            for index in range(self.manifest.num_scenes):
                families.add(
                    _scene_family_signature(
                        self.manifest.start[index],
                        self.manifest.goal[index],
                        self.manifest.obstacle_center[index],
                        float(self.manifest.obstacle_radius[index]),
                    )
                )
        else:
            families = set(int(x) for x in np.asarray(self.archive.scene_id).tolist())
        return SplitFamilies(split=self.requested_split, families=families)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_index, step_index = self.index[int(idx)]
        sl = self.archive.episode_slice(episode_index)
        absolute = int(sl.start + step_index)
        length = int(self.archive.length[episode_index])
        remaining = max(length - step_index, 0)
        future_len = min(self.horizon, remaining)
        obs_now = self.archive.obs[absolute].astype(np.float32)
        action_seq = np.zeros((self.horizon, ACTION_DIM), dtype=np.float32)
        obs_seq = np.zeros((self.horizon + 1, OBS_DIM), dtype=np.float32)
        mask = np.zeros((self.horizon,), dtype=bool)
        obs_seq[0] = obs_now
        if future_len > 0:
            action_seq[:future_len] = self.archive.action[absolute : absolute + future_len]
            obs_seq[1 : future_len + 1] = self.archive.next_obs[absolute : absolute + future_len]
            mask[:future_len] = True
        if future_len < self.horizon:
            fill = obs_seq[future_len] if future_len > 0 else obs_now
            obs_seq[future_len + 1 :] = fill
        obs_history = self._obs_history(sl, step_index)
        action_history = self._action_history(sl, step_index)
        progress = float(step_index / max(length - 1, 1))
        context = self._context_from_arrays(obs_now, obs_history, action_history, progress)
        scene_id = int(self.archive.scene_id[episode_index])
        item: dict[str, Any] = {
            "context": context,
            "action_seq": action_seq,
            "obs_seq": obs_seq,
            "state_seq": None,
            "goal": obs_now[7:9].astype(np.float32),
            "tilt": obs_now[12:14].astype(np.float32),
            "route_label": _route_label(obs_seq[: future_len + 1]) if future_len > 0 else "center",
            "success_label": bool(self.archive.success[episode_index]),
            "scene_id": scene_id,
            "family_id": self._family_for_scene(scene_id),
            "episode_id": int(self.archive.episode_id[episode_index]),
            "timestep": int(step_index),
            "mask": mask,
        }
        if self.target_mode == "state_action":
            item["state_action_seq"] = np.concatenate([obs_seq[1:], action_seq], axis=-1).astype(np.float32)
        for key in ("context", "action_seq", "obs_seq", "goal", "tilt"):
            if not np.all(np.isfinite(item[key])):
                raise ValueError(f"Non-finite values in {key}.")
        return item

    def _context_from_arrays(
        self,
        obs_now: np.ndarray,
        obs_history: np.ndarray,
        action_history: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        from aif.context import build_generator_context_vector

        belief_summary = np.zeros(5, dtype=np.float32)
        return build_generator_context_vector(
            obs_t=obs_now,
            obs_history=obs_history,
            action_history=action_history,
            belief_summary=belief_summary,
            time_fraction=progress,
        )

    def _obs_history(self, episode_slice: slice, step_index: int) -> np.ndarray:
        rows = []
        for offset in range(self.history_len - 1, -1, -1):
            local = max(step_index - offset, 0)
            rows.append(self.archive.obs[episode_slice.start + local])
        return np.asarray(rows, dtype=np.float32)

    def _action_history(self, episode_slice: slice, step_index: int) -> np.ndarray:
        rows = []
        for offset in range(self.history_len, 0, -1):
            local = step_index - offset
            if local < 0:
                rows.append(np.zeros(ACTION_DIM, dtype=np.float32))
            else:
                rows.append(self.archive.action[episode_slice.start + local])
        return np.asarray(rows, dtype=np.float32)

    def _family_for_scene(self, scene_id: int) -> int:
        if self.manifest is None:
            return int(scene_id)
        matches = np.flatnonzero(np.asarray(self.manifest.scene_id, dtype=np.int64) == int(scene_id))
        if matches.size == 0:
            return int(scene_id)
        return int(self.manifest.family_id[int(matches[0])])

    def route_distribution(self) -> dict[str, int]:
        counts = {"left": 0, "right": 0, "center": 0, "invalid": 0}
        for index in range(len(self)):
            label = str(self[index]["route_label"])
            counts[label if label in counts else "invalid"] += 1
        return counts

    def tilt_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index in range(len(self)):
            tilt = self[index]["tilt"]
            key = f"{float(tilt[0]):+.2f}|{float(tilt[1]):.2f}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _route_balanced_index(self, base_index: list[tuple[int, int]]) -> list[tuple[int, int]]:
        buckets = {"left": [], "right": [], "center": [], "invalid": []}
        for pair in base_index:
            episode_index, step_index = pair
            sl = self.archive.episode_slice(episode_index)
            absolute = int(sl.start + step_index)
            future_len = min(self.horizon, int(self.archive.length[episode_index] - step_index))
            obs_seq = np.zeros((future_len + 1, OBS_DIM), dtype=np.float32)
            obs_seq[0] = self.archive.obs[absolute]
            if future_len > 0:
                obs_seq[1:] = self.archive.next_obs[absolute : absolute + future_len]
            label = _route_label(obs_seq) if future_len > 0 else "center"
            buckets[label if label in buckets else "invalid"].append(pair)
        nonempty = [rows for rows in buckets.values() if rows]
        if not nonempty:
            return base_index
        target = max(len(rows) for rows in nonempty)
        balanced = []
        rng = np.random.default_rng(0)
        for rows in nonempty:
            if len(rows) < target:
                extra = [rows[int(i)] for i in rng.integers(0, len(rows), size=target - len(rows))]
                balanced.extend(rows + extra)
            else:
                balanced.extend(rows)
        return balanced


def assert_generator_scene_family_disjoint(*datasets: GeneratorActionWindowDataset) -> None:
    seen: dict[int, str] = {}
    for dataset in datasets:
        split_families = dataset.split_families()
        for family in split_families.families:
            owner = seen.get(family)
            if owner is not None and owner != split_families.split:
                raise AssertionError(f"Scene family {family} appears in both {owner} and {split_families.split}.")
            seen[family] = split_families.split


class FailureConditionedGeneratorActionWindowDataset:
    """Generator windows with explicit success/failure outcome conditioning.

    Successful windows come from the normal ``EpisodeArchive``. Failure windows
    come from sidecar ``failure_trajectories.npz`` files emitted by the MuJoCo
    specialist dataset generator. Context vectors append a one-hot terminal
    outcome label so models can be queried for successful proposals while still
    learning what failure-conditioned behavior looks like.
    """

    def __init__(
        self,
        archive_root: str | Path | None = None,
        *,
        preset: str = "smoke",
        split: str = "train",
        horizon: int | None = None,
        history_len: int = 4,
        successful_only: bool = True,
        include_failures: bool = False,
        failure_sidecar_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        max_failure_windows: int | None = None,
        route_balance: bool = False,
        target_mode: str = "action",
        append_outcome_to_context: bool = True,
    ) -> None:
        del successful_only, include_failures
        self.success_dataset = GeneratorActionWindowDataset(
            archive_root=archive_root,
            preset=preset,
            split=split,
            horizon=horizon,
            history_len=history_len,
            successful_only=True,
            include_failures=False,
            route_balance=route_balance,
            target_mode=target_mode,
        )
        self.preset = self.success_dataset.preset
        self.requested_split = self.success_dataset.requested_split
        self.split = self.success_dataset.split
        self.horizon = int(self.success_dataset.horizon)
        self.history_len = int(history_len)
        self.target_mode = str(target_mode)
        self.append_outcome_to_context = bool(append_outcome_to_context)
        self.failure_sidecar_paths = [Path(path) for path in (failure_sidecar_paths or [])]
        self.failure_archives = [self._load_failure_sidecar(path) for path in self.failure_sidecar_paths]
        self.failure_index: list[tuple[int, int, int]] = []
        for archive_index, archive in enumerate(self.failure_archives):
            for episode_index in range(int(archive["length"].shape[0])):
                length = int(archive["length"][episode_index])
                for step_index in range(length):
                    self.failure_index.append((archive_index, episode_index, step_index))
        if max_failure_windows is not None:
            self.failure_index = self.failure_index[: max(0, int(max_failure_windows))]
        first = self[0]
        self.context_dim = int(first["context"].shape[0])
        self.archive_root = self.success_dataset.archive_root
        self.archive = self.success_dataset.archive
        self.manifest = self.success_dataset.manifest

    def __len__(self) -> int:
        return len(self.success_dataset) + len(self.failure_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        index = int(idx)
        if index < len(self.success_dataset):
            item = dict(self.success_dataset[index])
            return self._with_outcome(item, "success")
        archive_index, episode_index, step_index = self.failure_index[index - len(self.success_dataset)]
        return self._failure_item(archive_index, episode_index, step_index)

    def window_keys(self) -> list[tuple[int, int]]:
        success_keys = self.success_dataset.window_keys()
        failure_keys = []
        for archive_index, episode_index, step_index in self.failure_index:
            archive = self.failure_archives[archive_index]
            episode_id = int(archive["episode_id"][episode_index])
            failure_keys.append((-(episode_id + 1), int(step_index)))
        return success_keys + failure_keys

    def split_families(self) -> SplitFamilies:
        return self.success_dataset.split_families()

    def route_distribution(self) -> dict[str, int]:
        counts = {"left": 0, "right": 0, "center": 0, "invalid": 0}
        for index in range(len(self)):
            label = str(self[index]["route_label"])
            counts[label if label in counts else "invalid"] += 1
        return counts

    def tilt_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index in range(len(self)):
            tilt = self[index]["tilt"]
            key = f"{float(tilt[0]):+.2f}|{float(tilt[1]):.2f}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _with_outcome(self, item: dict[str, Any], outcome: str) -> dict[str, Any]:
        one_hot = outcome_one_hot(outcome)
        payload = dict(item)
        payload["outcome_label"] = str(outcome)
        payload["outcome_id"] = int(np.argmax(one_hot))
        payload["terminal_reason"] = str(outcome)
        payload["outcome_one_hot"] = one_hot
        if self.append_outcome_to_context:
            payload["context"] = np.concatenate([np.asarray(payload["context"], dtype=np.float32), one_hot], axis=0).astype(np.float32)
        return payload

    def _failure_item(self, archive_index: int, episode_index: int, step_index: int) -> dict[str, Any]:
        archive = self.failure_archives[int(archive_index)]
        start = int(archive["offset"][episode_index])
        length = int(archive["length"][episode_index])
        absolute = start + int(step_index)
        remaining = max(length - int(step_index), 0)
        future_len = min(self.horizon, remaining)
        obs_now = archive["obs"][absolute].astype(np.float32)
        action_seq = np.zeros((self.horizon, ACTION_DIM), dtype=np.float32)
        obs_seq = np.zeros((self.horizon + 1, OBS_DIM), dtype=np.float32)
        mask = np.zeros((self.horizon,), dtype=bool)
        obs_seq[0] = obs_now
        if future_len > 0:
            action_seq[:future_len] = archive["action"][absolute : absolute + future_len]
            obs_seq[1 : future_len + 1] = archive["next_obs"][absolute : absolute + future_len]
            mask[:future_len] = True
        if future_len < self.horizon:
            fill = obs_seq[future_len] if future_len > 0 else obs_now
            obs_seq[future_len + 1 :] = fill
        obs_history = self._failure_obs_history(archive, start, step_index)
        action_history = self._failure_action_history(archive, start, step_index)
        progress = float(step_index / max(length - 1, 1))
        context = self.success_dataset._context_from_arrays(obs_now, obs_history, action_history, progress)
        terminal_reason = str(archive["terminal_reason"][episode_index])
        outcome = terminal_reason if terminal_reason in OUTCOME_LABELS else "timeout"
        item: dict[str, Any] = {
            "context": context,
            "action_seq": action_seq,
            "obs_seq": obs_seq,
            "state_seq": None,
            "goal": obs_now[7:9].astype(np.float32),
            "tilt": obs_now[12:14].astype(np.float32),
            "route_label": _route_label(obs_seq[: future_len + 1]) if future_len > 0 else "invalid",
            "success_label": False,
            "scene_id": int(archive["scene_id"][episode_index]),
            "family_id": int(archive["scene_id"][episode_index]),
            "episode_id": int(archive["episode_id"][episode_index]),
            "timestep": int(step_index),
            "mask": mask,
        }
        if self.target_mode == "state_action":
            item["state_action_seq"] = np.concatenate([obs_seq[1:], action_seq], axis=-1).astype(np.float32)
        return self._with_outcome(item, outcome)

    def _failure_obs_history(self, archive: dict[str, np.ndarray], start: int, step_index: int) -> np.ndarray:
        rows = []
        for offset in range(self.history_len - 1, -1, -1):
            local = max(int(step_index) - offset, 0)
            rows.append(archive["obs"][start + local])
        return np.asarray(rows, dtype=np.float32)

    def _failure_action_history(self, archive: dict[str, np.ndarray], start: int, step_index: int) -> np.ndarray:
        rows = []
        for offset in range(self.history_len, 0, -1):
            local = int(step_index) - offset
            if local < 0:
                rows.append(np.zeros(ACTION_DIM, dtype=np.float32))
            else:
                rows.append(archive["action"][start + local])
        return np.asarray(rows, dtype=np.float32)

    def _load_failure_sidecar(self, path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            required = (
                "episode_id",
                "scene_id",
                "offset",
                "length",
                "terminal_reason",
                "obs",
                "action",
                "next_obs",
                "done",
            )
            missing = [name for name in required if name not in payload.files]
            if missing:
                raise ValueError(f"Failure sidecar {path} is missing fields: {missing}")
            return {name: np.asarray(payload[name]) for name in payload.files}
