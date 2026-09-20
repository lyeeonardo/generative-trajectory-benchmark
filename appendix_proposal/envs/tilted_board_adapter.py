"""Stable adapter around the MuJoCo rigid tilted-board simulator."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_task.config import ACTION_DIM, OBS_DIM, Preset, SimulatorConfig, get_preset
from mujoco_task.dataset.schema import EpisodeArchive, SceneManifest
from mujoco_task.sim.scene import SceneFamily, SceneSpec, expand_family, sample_scene_family, straight_path_blocked


@dataclass(frozen=True)
class StateSnapshot:
    """Copyable MuJoCo simulator state used by planner rollouts."""

    scene: SceneSpec
    state: Any
    physics_backend: str = "mujoco_rigid"

    def copy(self) -> "StateSnapshot":
        return StateSnapshot(
            scene=self.scene,
            state=self.state.copy(),
            physics_backend=str(self.physics_backend),
        )


@dataclass(frozen=True)
class RolloutResult:
    observations: np.ndarray
    states: list[StateSnapshot]
    actions: np.ndarray
    collision: bool
    success: bool
    timeout: bool
    final_distance_to_goal: float
    minimum_obstacle_clearance: float
    path_length: float
    action_smoothness: float
    route_label: str
    fall_out: bool = False
    raw_simulator_info: dict[str, Any] = field(default_factory=dict)


def _copy_scene(scene: SceneSpec) -> SceneSpec:
    return SceneSpec(
        start=scene.start.copy(),
        goal=scene.goal.copy(),
        obstacle_center=scene.obstacle_center.copy(),
        obstacle_radius=float(scene.obstacle_radius),
        lateral_tilt=float(scene.lateral_tilt),
        longitudinal_tilt=float(scene.longitudinal_tilt),
        seed=int(scene.seed),
    )


class TiltedBoardEnvAdapter:
    """Planner-facing interface over the MuJoCo rigid rod-push backend."""

    def __init__(
        self,
        preset: Preset | str = "smoke",
        *,
        split: str = "test",
        physics_backend: str = "mujoco_rigid",
        physics_params: dict[str, Any] | None = None,
        sim_params: dict[str, Any] | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        base_preset = get_preset(preset) if isinstance(preset, str) else preset
        self.sim_params = self._validated_sim_params(sim_params)
        self.preset = (
            base_preset
            if not self.sim_params
            else replace(base_preset, sim=replace(base_preset.sim, **self.sim_params))
        )
        self.split = str(split)
        self.physics_backend = self._normalize_backend(physics_backend)
        self.physics_params = dict(physics_params or {})
        self.dataset_root = None if dataset_root is None else Path(dataset_root)
        self.env = self._new_backend_env()
        self.scene_id: int | None = None
        self.family_id: int | None = None
        self._manifest_cache: SceneManifest | None = None
        self._archive_cache: EpisodeArchive | None = None
        self._rollout_env_cache: dict[tuple[Any, ...], Any] = {}
        self._live_scene_key: tuple[Any, ...] | None = None
        self._live_initial_state: Any | None = None

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def _manifest(self) -> SceneManifest | None:
        roots = []
        if self.dataset_root is not None:
            roots.append(self.dataset_root / self.split)
        roots.append(self.preset.paths.data_dir / self.preset.name / self.split)
        root = next((candidate for candidate in roots if (candidate / "scenes.npz").exists()), roots[-1])
        if not (root / "scenes.npz").exists():
            return None
        if self._manifest_cache is None:
            self._manifest_cache = SceneManifest.load(root, self.split)
        return self._manifest_cache

    def _archive(self) -> EpisodeArchive | None:
        if self.dataset_root is None:
            return None
        root = self.dataset_root / self.split
        if not (root / "episodes.npz").exists():
            return None
        if self._archive_cache is None:
            self._archive_cache = EpisodeArchive.load(root, self.split)
        return self._archive_cache

    def _scene_from_manifest(self, scene_id: int, tilt: tuple[float, float] | None) -> SceneSpec:
        manifest = self._manifest()
        if manifest is None:
            return self._sample_scene(int(scene_id), tilt)
        ids = np.asarray(manifest.scene_id, dtype=np.int64)
        matches = np.flatnonzero(ids == int(scene_id))
        if matches.size == 0:
            index = int(scene_id) % manifest.num_scenes
        else:
            index = int(matches[0])
        lateral = float(manifest.lateral_tilt[index])
        longitudinal = float(manifest.longitudinal_tilt[index])
        if tilt is not None:
            lateral, longitudinal = float(tilt[0]), float(tilt[1])
        self.scene_id = int(manifest.scene_id[index])
        self.family_id = int(manifest.family_id[index])
        return SceneSpec(
            start=manifest.start[index],
            goal=manifest.goal[index],
            obstacle_center=manifest.obstacle_center[index],
            obstacle_radius=float(manifest.obstacle_radius[index]),
            lateral_tilt=lateral,
            longitudinal_tilt=longitudinal,
            seed=int(manifest.seed[index]),
        )

    def _scene_from_archive(self, scene_id: int, tilt: tuple[float, float] | None) -> SceneSpec | None:
        archive = self._archive()
        if archive is None:
            return None
        ids = np.asarray(archive.scene_id, dtype=np.int64)
        matches = np.flatnonzero(ids == int(scene_id))
        if matches.size == 0:
            return None
        index = int(matches[0])
        episode_slice = archive.episode_slice(index)
        start = np.asarray(archive.obs[episode_slice.start, :2], dtype=np.float32)
        lateral = float(archive.lateral_tilt[index])
        longitudinal = float(archive.longitudinal_tilt[index])
        if tilt is not None:
            lateral, longitudinal = float(tilt[0]), float(tilt[1])
        modulus = int(archive.meta.get("scene_id_modulus", 1_000_000))
        self.scene_id = int(archive.scene_id[index])
        self.family_id = int(self.scene_id % modulus) if modulus > 0 else int(self.scene_id)
        return SceneSpec(
            start=start,
            goal=np.asarray(archive.goal[index], dtype=np.float32),
            obstacle_center=np.asarray(archive.obstacle_center[index], dtype=np.float32),
            obstacle_radius=float(archive.obstacle_radius[index]),
            lateral_tilt=lateral,
            longitudinal_tilt=longitudinal,
            seed=int(self.family_id),
        )

    def _sample_scene(self, seed: int, tilt: tuple[float, float] | None) -> SceneSpec:
        rng = np.random.default_rng(int(seed))
        family_id = int(seed)
        family = None
        for offset in range(512):
            candidate = sample_scene_family(rng, self.preset, family_id + offset)
            if straight_path_blocked(candidate, self.preset):
                family = candidate
                break
        if family is None:
            family = SceneFamily(
                family_id=family_id,
                start=np.array([0.0, -0.78], dtype=np.float32),
                goal=np.array([0.0, 0.78], dtype=np.float32),
                obstacle_center=np.asarray(self.preset.scene.obstacle_center, dtype=np.float32),
                obstacle_radius=float(self.preset.sim.obstacle_radius),
            )
        scenes = expand_family(family, self.preset)
        if tilt is None:
            lateral = 0.0
            longitudinal = self.preset.scene.longitudinal_tilts[len(self.preset.scene.longitudinal_tilts) // 2]
        else:
            lateral, longitudinal = float(tilt[0]), float(tilt[1])
        scene = min(
            scenes,
            key=lambda item: abs(item.lateral_tilt - lateral) + abs(item.longitudinal_tilt - longitudinal),
        )
        if tilt is not None and (
            not np.isclose(scene.lateral_tilt, lateral) or not np.isclose(scene.longitudinal_tilt, longitudinal)
        ):
            scene = SceneSpec(
                start=scene.start,
                goal=scene.goal,
                obstacle_center=scene.obstacle_center,
                obstacle_radius=scene.obstacle_radius,
                lateral_tilt=lateral,
                longitudinal_tilt=longitudinal,
                seed=scene.seed,
            )
        self.scene_id = int(seed)
        self.family_id = int(family.family_id)
        return scene

    def reset(
        self,
        seed: int | None = None,
        scene_id: int | None = None,
        tilt: tuple[float, float] | list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        tilt_tuple = None if tilt is None else (float(tilt[0]), float(tilt[1]))
        if scene_id is not None:
            scene = self._scene_from_archive(int(scene_id), tilt_tuple)
            if scene is None:
                scene = self._scene_from_manifest(int(scene_id), tilt_tuple)
        else:
            scene = self._sample_scene(0 if seed is None else int(seed), tilt_tuple)
        return self._reset_live_scene(scene)

    def reset_scene(
        self,
        scene: SceneSpec,
        *,
        scene_id: int | None = None,
        family_id: int | None = None,
    ) -> np.ndarray:
        """Reset from an explicit immutable scene specification.

        Campaign evaluation uses this entry point so geometry and dynamics are
        read from a hashed manifest instead of being re-sampled by the adapter.
        Existing dataset- and seed-based reset behavior is unchanged.
        """

        self.scene_id = int(scene.seed if scene_id is None else scene_id)
        self.family_id = int(scene.seed if family_id is None else family_id)
        return self._reset_live_scene(scene)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool | str], bool, dict[str, Any]]:
        result = self.env.step(self._clip_action(np.asarray(action, dtype=np.float32)))
        reward_like = {"reward": float(result.reward), **result.info}
        info = dict(result.info)
        info.update(
            {
                "success": bool(result.success),
                "collision": bool(result.collision),
                "timeout": bool(result.timeout),
            }
        )
        return result.observation.copy(), reward_like, bool(result.done), info

    def clone_state(self) -> StateSnapshot:
        if self.env.scene is None or self.env.state is None:
            raise RuntimeError("Cannot clone before reset.")
        return StateSnapshot(
            scene=_copy_scene(self.env.scene),
            state=self.env.state.copy(),
            physics_backend=self.physics_backend,
        )

    def restore_state(self, state_snapshot: StateSnapshot) -> None:
        self._assert_snapshot_backend(state_snapshot)
        if self._live_scene_key != self._scene_cache_key(state_snapshot.scene):
            self._live_scene_key = None
            self._live_initial_state = None
        self._restore_backend_env(self.env, state_snapshot)

    def rollout_from_state(self, state_snapshot: StateSnapshot, action_sequence: np.ndarray) -> RolloutResult:
        self._assert_snapshot_backend(state_snapshot)
        actions = self._clip_actions(np.asarray(action_sequence, dtype=np.float32))
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected action_sequence shaped (H, {ACTION_DIM}), got {actions.shape}.")
        trial = self._rollout_env(state_snapshot)
        observations = [trial.current_observation().copy()]
        states = [
            StateSnapshot(
                scene=_copy_scene(trial.scene),
                state=trial.state.copy(),
                physics_backend=self.physics_backend,
            )
        ]
        infos: list[dict[str, Any]] = []
        success = False
        collision = False
        fall_out = False
        timeout = False
        for action in actions:
            result = trial.step(action)
            observations.append(result.observation.copy())
            states.append(
                StateSnapshot(
                    scene=_copy_scene(trial.scene),
                    state=trial.state.copy(),
                    physics_backend=self.physics_backend,
                )
            )
            infos.append(dict(result.info))
            success = success or bool(result.success)
            collision = collision or bool(result.collision)
            fall_out = fall_out or bool(result.info.get("fall_out", False))
            timeout = timeout or bool(result.timeout)
            if result.done:
                break
        while len(observations) < actions.shape[0] + 1:
            observations.append(observations[-1].copy())
            states.append(states[-1].copy())
        obs_array = np.asarray(observations[: actions.shape[0] + 1], dtype=np.float32)
        ball_path = obs_array[:, :2]
        final_distance = float(np.linalg.norm(ball_path[-1] - state_snapshot.scene.goal))
        center = state_snapshot.scene.obstacle_center
        clearance = np.linalg.norm(ball_path - center[None, :], axis=1) - (
            float(state_snapshot.scene.obstacle_radius) + float(self.preset.sim.ball_radius)
        )
        path_length = float(np.sum(np.linalg.norm(np.diff(ball_path, axis=0), axis=1))) if len(ball_path) > 1 else 0.0
        smoothness = (
            float(np.mean(np.sum(np.diff(actions, axis=0) ** 2, axis=1)))
            if actions.shape[0] > 1
            else 0.0
        )
        if self.is_timeout(trial.state.step) and not success and not collision and not fall_out:
            timeout = True
        return RolloutResult(
            observations=obs_array,
            states=states[: actions.shape[0] + 1],
            actions=actions,
            collision=bool(collision),
            success=bool(success),
            timeout=bool(timeout),
            fall_out=bool(fall_out),
            final_distance_to_goal=final_distance,
            minimum_obstacle_clearance=float(np.min(clearance)),
            path_length=path_length,
            action_smoothness=smoothness,
            route_label=self._route_label(obs_array, bool(collision or fall_out)),
            raw_simulator_info={
                "steps_executed": int(min(len(infos), actions.shape[0])),
                "infos": infos,
                "scene_id": self.scene_id,
                "family_id": self.family_id,
                "physics_backend": self.physics_backend,
                "physics_params": self._physics_metadata(),
            },
        )

    def get_obs(self) -> np.ndarray:
        return self.env.current_observation().copy()

    def get_scene_context(self) -> dict[str, Any]:
        if self.env.scene is None or self.env.state is None:
            raise RuntimeError("Environment is not reset.")
        scene = self.env.scene
        return {
            "scene_id": self.scene_id,
            "family_id": self.family_id,
            "start": scene.start.copy(),
            "goal": scene.goal.copy(),
            "obstacle_center": scene.obstacle_center.copy(),
            "obstacle_radius": float(scene.obstacle_radius),
            "tilt": np.asarray([scene.lateral_tilt, scene.longitudinal_tilt], dtype=np.float32),
            "step": int(self.env.state.step),
            "max_steps": int(self.preset.sim.max_steps),
            "workspace_x": tuple(self.preset.sim.workspace_x),
            "workspace_y": tuple(self.preset.sim.workspace_y),
            "ball_radius": float(self.preset.sim.ball_radius),
            "physics_backend": self.physics_backend,
            "physics_params": self._physics_metadata(),
            "sim_params": dict(self.sim_params),
        }

    def get_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.asarray(
            [-self.preset.sim.action_max_speed, -self.preset.sim.action_max_speed, -self.preset.sim.action_max_omega],
            dtype=np.float32,
        )
        high = np.asarray(
            [self.preset.sim.action_max_speed, self.preset.sim.action_max_speed, self.preset.sim.action_max_omega],
            dtype=np.float32,
        )
        return low, high

    def is_success(self, obs_or_state: np.ndarray | StateSnapshot) -> bool:
        obs = self._obs_from_any(obs_or_state)
        return bool(np.linalg.norm(obs[:2] - obs[7:9]) <= self.preset.sim.goal_radius)

    def is_collision(self, obs_or_state: np.ndarray | StateSnapshot) -> bool:
        obs = self._obs_from_any(obs_or_state)
        clearance = np.linalg.norm(obs[:2] - obs[9:11]) - (float(obs[11]) + self.preset.sim.ball_radius)
        return bool(clearance <= 0.0)

    def is_timeout(self, t: int | float) -> bool:
        return bool(int(t) >= int(self.preset.sim.max_steps))

    def _obs_from_any(self, obs_or_state: np.ndarray | StateSnapshot) -> np.ndarray:
        if isinstance(obs_or_state, StateSnapshot):
            from envs.mujoco_tilted_board import compose_mujoco_observation

            return compose_mujoco_observation(obs_or_state.state, obs_or_state.scene)
        return np.asarray(obs_or_state, dtype=np.float32).reshape(OBS_DIM)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
        low, high = self.get_action_bounds()
        return np.clip(action, low, high).astype(np.float32)

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        low, high = self.get_action_bounds()
        return np.clip(actions, low[None, :], high[None, :]).astype(np.float32)

    def _route_label(self, observations: np.ndarray, collision: bool) -> str:
        if collision or not np.all(np.isfinite(observations)):
            return "invalid"
        xy = observations[:, :2]
        center_y = float(observations[0, 10])
        band = np.abs(xy[:, 1] - center_y) <= 0.30
        near = xy[band] if np.any(band) else xy
        if np.min(near[:, 0]) < -0.12:
            return "left"
        if np.max(near[:, 0]) > 0.12:
            return "right"
        return "center"

    def _normalize_backend(self, physics_backend: str) -> str:
        backend = str(physics_backend).strip().lower()
        aliases = {
            "default": "mujoco_rigid",
            "mujoco": "mujoco_rigid",
            "mujoco_rigid": "mujoco_rigid",
        }
        if backend not in aliases:
            raise ValueError("physics_backend must be 'mujoco_rigid'.")
        return aliases[backend]

    @staticmethod
    def _validated_sim_params(sim_params: dict[str, Any] | None) -> dict[str, Any]:
        if not sim_params:
            return {}
        valid = {item.name for item in fields(SimulatorConfig)}
        unknown = sorted(set(sim_params) - valid)
        if unknown:
            raise ValueError(f"Unknown simulator parameter(s): {unknown}")
        return dict(sim_params)

    def _new_backend_env(self):
        if self.physics_backend == "mujoco_rigid":
            from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv

            return MujocoRigidTiltPushEnv(self.preset.sim, self.physics_params)
        raise ValueError(f"Unsupported physics_backend={self.physics_backend!r}.")

    @staticmethod
    def _scene_cache_key(scene: SceneSpec) -> tuple[Any, ...]:
        return (
            *(float(value) for value in np.asarray(scene.start, dtype=np.float64).reshape(-1)),
            *(float(value) for value in np.asarray(scene.goal, dtype=np.float64).reshape(-1)),
            *(float(value) for value in np.asarray(scene.obstacle_center, dtype=np.float64).reshape(-1)),
            float(scene.obstacle_radius),
            float(scene.lateral_tilt),
            float(scene.longitudinal_tilt),
            int(scene.seed),
        )

    def _rollout_env(self, state_snapshot: StateSnapshot):
        """Restore a cached trial simulator without recompiling its scene XML."""

        key = self._scene_cache_key(state_snapshot.scene)
        trial = self._rollout_env_cache.get(key)
        if trial is None:
            trial = self._new_backend_env()
            trial.reset(_copy_scene(state_snapshot.scene))
            self._rollout_env_cache[key] = trial
        trial.set_state(state_snapshot.state.copy())
        return trial

    def _reset_live_scene(self, scene: SceneSpec) -> np.ndarray:
        key = self._scene_cache_key(scene)
        if self._live_scene_key == key and self._live_initial_state is not None:
            self.env.set_state(self._live_initial_state.copy())
            return self.env.current_observation().copy()
        self._rollout_env_cache.clear()
        observation = self.env.reset(_copy_scene(scene)).copy()
        self._live_scene_key = key
        self._live_initial_state = self.env.state.copy()
        return observation

    def _restore_backend_env(self, env, state_snapshot: StateSnapshot) -> None:
        scene = _copy_scene(state_snapshot.scene)
        if self.physics_backend == "mujoco_rigid":
            if env.scene is None or self._scene_cache_key(env.scene) != self._scene_cache_key(scene):
                env.reset(scene)
            env.set_state(state_snapshot.state.copy())
            return
        raise ValueError(f"Unsupported physics_backend={self.physics_backend!r}.")

    def _assert_snapshot_backend(self, state_snapshot: StateSnapshot) -> None:
        if str(state_snapshot.physics_backend) != self.physics_backend:
            raise ValueError(
                f"Snapshot backend {state_snapshot.physics_backend!r} does not match adapter backend "
                f"{self.physics_backend!r}."
            )

    def _physics_metadata(self) -> dict[str, Any]:
        if self.physics_backend == "mujoco_rigid":
            physics_config = getattr(self.env, "physics_config", None)
            if physics_config is not None and hasattr(physics_config, "as_dict"):
                return physics_config.as_dict()
        return dict(self.physics_params)
