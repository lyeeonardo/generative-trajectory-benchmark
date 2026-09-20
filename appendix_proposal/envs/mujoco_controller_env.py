"""Gymnasium wrapper for training a MuJoCo rigid rod-push controller."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only outside RL env.
    raise RuntimeError("MujocoRodPushGymEnv requires gymnasium. Use the RL conda environment.") from exc

from envs.tilted_board_adapter import TiltedBoardEnvAdapter

from mujoco_task.config import OBS_DIM, get_preset
from mujoco_task.sim.scene import SceneSpec


SUCCESS_ROUTE_ACTIONS_30 = np.asarray(
    [
        [-0.0598373451, 0.4986910358, 0.3562963917],
        [0.0822556015, 0.7876584883, -0.7834532351],
        [-0.1855836380, 0.4469501764, 0.4735564233],
        [0.0896529197, 0.5384181594, 0.2418880156],
        [-0.0488970886, 0.6521488314, -0.3975889678],
        [0.0244415333, 0.6612738852, 0.4595303137],
        [0.4668084393, 0.2263113438, 0.3731034529],
        [-0.1800483004, 0.4490665223, 0.9694241890],
        [-0.0743774935, 0.7624165540, 0.8533095203],
    ],
    dtype=np.float32,
)

SUCCESS_ROUTE_ACTIONS_10 = np.asarray(
    [
        [0.0256183218, 0.5754329960, 0.1851760305],
        [-0.1718260011, 0.5242435212, 0.4710971961],
        [-0.1653622651, 0.6108847616, -0.6563024410],
        [0.0085341291, 0.5408280226, 0.3113350265],
        [-0.2358126930, 0.4024160091, -0.7030735825],
        [-0.0227603889, 0.4631454899, -0.7581031224],
        [0.0930508070, 0.5998334947, 0.5965614651],
        [0.3526022448, 0.3569768584, -0.4634963991],
        [-0.3756252565, 0.5653102077, -0.9923656598],
    ],
    dtype=np.float32,
)


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


class MujocoRodPushGymEnv(gym.Env):
    """Continuous-control environment for the opt-in MuJoCo rigid backend.

    The policy action is the rod-push action vector ``[vx, vy, omega]``.
    Observations keep the existing 14D layout and are reported in board-local
    coordinates by the MuJoCo backend.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        preset: str = "run",
        split: str = "test",
        scene_seed: int = 3,
        scene_id: int | None = None,
        tilt_radians: float = math.pi / 6.0,
        lateral_tilt: float = 0.0,
        max_steps: int | None = 90,
        physics_params: dict[str, Any] | None = None,
        route_side: int = -1,
        coach_weight: float = 0.15,
        use_demo_coach: bool = True,
        start: np.ndarray | tuple[float, float] | list[float] | None = None,
        goal: np.ndarray | tuple[float, float] | list[float] | None = None,
        pair_id: int | None = None,
        task_pairs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        task_seed: int | None = None,
        obstacle_center: np.ndarray | tuple[float, float] | list[float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        base_preset = get_preset(preset)
        if max_steps is not None:
            base_preset = replace(base_preset, sim=replace(base_preset.sim, max_steps=int(max_steps)))
        merged_physics_params = {"push_offset": 0.0}
        merged_physics_params.update(dict(physics_params or {}))
        self.adapter = TiltedBoardEnvAdapter(
            base_preset,
            split=split,
            physics_backend="mujoco_rigid",
            physics_params=merged_physics_params,
        )
        low, high = self.adapter.get_action_bounds()
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.full(OBS_DIM, -np.inf, dtype=np.float32),
            high=np.full(OBS_DIM, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.scene_seed = int(scene_seed)
        self.scene_id = None if scene_id is None else int(scene_id)
        self.tilt_radians = float(tilt_radians)
        self.lateral_tilt = float(lateral_tilt)
        self.route_side = -1 if int(route_side) < 0 else 1
        self.coach_weight = float(coach_weight)
        self.use_demo_coach = bool(use_demo_coach)
        self.max_steps = int(base_preset.sim.max_steps)
        self.fixed_start = None if start is None else np.asarray(start, dtype=np.float32).reshape(2)
        self.fixed_goal = None if goal is None else np.asarray(goal, dtype=np.float32).reshape(2)
        if (self.fixed_start is None) != (self.fixed_goal is None):
            raise ValueError("start and goal must be provided together.")
        self.fixed_pair_id = None if pair_id is None else int(pair_id)
        self.obstacle_center = np.asarray(obstacle_center, dtype=np.float32).reshape(2)
        self.task_pairs = self._normalize_task_pairs(task_pairs)
        self._task_rng = np.random.default_rng(self.scene_seed if task_seed is None else int(task_seed))
        self._current_pair_id = self.fixed_pair_id
        self._current_start = None if self.fixed_start is None else self.fixed_start.copy()
        self._current_goal = None if self.fixed_goal is None else self.fixed_goal.copy()
        self._previous_distance = 0.0
        self._episode_step = 0
        self._contact_steps = 0
        self._obstacle_contact_steps = 0
        self._last_obs: np.ndarray | None = None

    @property
    def unwrapped_backend(self):
        return self.adapter.env

    def set_tilt_radians(self, tilt_radians: float) -> None:
        self.tilt_radians = float(tilt_radians)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        if "tilt_radians" in options:
            self.tilt_radians = float(options["tilt_radians"])
        if "lateral_tilt" in options:
            self.lateral_tilt = float(options["lateral_tilt"])
        scene_seed = int(options.get("scene_seed", self.scene_seed if seed is None else seed))
        scene_id = options.get("scene_id", self.scene_id)
        task = self._select_task(options)
        if task is None:
            self._current_pair_id = None
            self._current_start = None
            self._current_goal = None
            obs = self.adapter.reset(
                seed=scene_seed,
                scene_id=None if scene_id is None else int(scene_id),
                tilt=(self.lateral_tilt, self.tilt_radians),
            ).astype(np.float32)
        else:
            start_xy, goal_xy, selected_pair_id = task
            scene = SceneSpec(
                start=start_xy.copy(),
                goal=goal_xy.copy(),
                obstacle_center=self.obstacle_center.copy(),
                obstacle_radius=float(self.adapter.preset.sim.obstacle_radius),
                lateral_tilt=float(self.lateral_tilt),
                longitudinal_tilt=float(self.tilt_radians),
                seed=scene_seed,
            )
            self.adapter.scene_id = int(scene_seed)
            self.adapter.family_id = None
            self._current_pair_id = selected_pair_id
            self._current_start = start_xy.copy()
            self._current_goal = goal_xy.copy()
            obs = self.adapter.env.reset(scene).astype(np.float32)
        self._episode_step = 0
        self._contact_steps = 0
        self._obstacle_contact_steps = 0
        self._previous_distance = self._goal_distance(obs)
        self._last_obs = obs.copy()
        return obs, self._info(obs, terminal_reason="running")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        obs, _, backend_done, backend_info = self.adapter.step(clipped)
        obs = obs.astype(np.float32)
        self._episode_step += 1
        if bool(backend_info.get("rod_ball_contact", False)):
            self._contact_steps += 1
        if bool(backend_info.get("ball_obstacle_collision", False)):
            self._obstacle_contact_steps += 1

        success = bool(backend_info.get("success", False))
        collision = bool(backend_info.get("collision", False))
        fall_out = bool(backend_info.get("fall_out", False))
        timeout = bool(backend_info.get("timeout", False)) or (
            self._episode_step >= self.max_steps and not success and not collision and not fall_out
        )
        terminal_reason = self._terminal_reason(success, collision, fall_out, timeout)
        terminated = terminal_reason in {"success", "collision", "fall_out"}
        truncated = terminal_reason == "timeout"
        reward = self._reward(obs, clipped, backend_info, terminal_reason)
        self._previous_distance = self._goal_distance(obs)
        self._last_obs = obs.copy()
        info = self._info(obs, backend_info=backend_info, terminal_reason=terminal_reason)
        info["backend_done"] = bool(backend_done)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def scripted_action(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Feedback rod target used only for reward shaping and diagnostics."""

        observation = self._last_obs if obs is None else np.asarray(obs, dtype=np.float32)
        if observation is None:
            return np.zeros(self.action_space.shape, dtype=np.float32)
        if self.use_demo_coach:
            return self.demo_action()
        ball = observation[:2]
        rod = observation[4:6]
        direction = self._route_direction(observation)
        ideal_rod = self._ideal_rod_position(observation, direction)
        velocity = 8.0 * (ideal_rod - rod)
        speed = float(np.linalg.norm(velocity))
        max_speed = float(self.action_space.high[0])
        if speed > max_speed > 0.0:
            velocity = velocity / speed * max_speed
        target_yaw = math.atan2(float(direction[1]), float(direction[0])) + 0.5 * math.pi
        omega = float(np.clip(6.0 * _wrap_angle(target_yaw - float(observation[6])), self.action_space.low[2], self.action_space.high[2]))
        return np.asarray([velocity[0], velocity[1], omega], dtype=np.float32)

    def demo_action(self) -> np.ndarray:
        route_actions = SUCCESS_ROUTE_ACTIONS_10 if self.tilt_radians < math.radians(15.0) else SUCCESS_ROUTE_ACTIONS_30
        segment = min(int(self._episode_step) // 10, len(route_actions) - 1)
        return np.clip(
            route_actions[segment],
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

    def _reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        backend_info: dict[str, Any],
        terminal_reason: str,
    ) -> float:
        distance = self._goal_distance(obs)
        progress = self._previous_distance - distance
        direction = self._route_direction(obs)
        ball_velocity = obs[2:4]
        clearance = self._obstacle_clearance(obs)
        ideal_rod = self._ideal_rod_position(obs, direction)
        rod_error = float(np.linalg.norm(obs[4:6] - ideal_rod))
        target_yaw = math.atan2(float(direction[1]), float(direction[0])) + 0.5 * math.pi
        yaw_error = _wrap_angle(target_yaw - float(obs[6]))
        action_energy = float(np.mean((action / np.maximum(np.abs(self.action_space.high), 1e-6)) ** 2))
        coach_action = self.scripted_action(obs)
        coach_error = float(np.mean(((action - coach_action) / np.maximum(np.abs(self.action_space.high), 1e-6)) ** 2))

        reward = 14.0 * progress - 1.15 * distance
        reward += 0.30 * float(np.dot(ball_velocity, direction))
        reward -= 0.45 * rod_error + 0.025 * yaw_error * yaw_error + 0.025 * action_energy
        reward -= self.coach_weight * coach_error
        if clearance < 0.26:
            reward -= 2.2 * float(0.26 - clearance)
        if float(obs[1]) < -0.92:
            reward -= 3.0 * float(-0.92 - obs[1])
        if bool(backend_info.get("rod_ball_contact", False)):
            reward += 0.16
        if terminal_reason == "success":
            reward += 60.0
        elif terminal_reason in {"collision", "fall_out"}:
            reward -= 35.0
        elif terminal_reason == "timeout":
            reward -= 4.0 + 2.0 * distance
        return float(reward)

    def _info(
        self,
        obs: np.ndarray,
        *,
        backend_info: dict[str, Any] | None = None,
        terminal_reason: str,
    ) -> dict[str, Any]:
        backend_info = dict(backend_info or {})
        success = terminal_reason == "success"
        collision = terminal_reason == "collision"
        fall_out = terminal_reason == "fall_out"
        timeout = terminal_reason == "timeout"
        return {
            **backend_info,
            "terminal_reason": terminal_reason,
            "success": bool(success),
            "collision": bool(collision),
            "fall_out": bool(fall_out),
            "timeout": bool(timeout),
            "final_distance": self._goal_distance(obs),
            "episode_length": int(self._episode_step),
            "rod_ball_contact_steps": int(self._contact_steps),
            "obstacle_contact_steps": int(self._obstacle_contact_steps),
            "tilt_radians": float(self.tilt_radians),
            "tilt_degrees": float(math.degrees(self.tilt_radians)),
            "lateral_tilt_radians": float(self.lateral_tilt),
            "lateral_tilt_degrees": float(math.degrees(self.lateral_tilt)),
            "pair_id": None if self._current_pair_id is None else int(self._current_pair_id),
            "start": None if self._current_start is None else self._current_start.astype(float).tolist(),
            "goal": None if self._current_goal is None else self._current_goal.astype(float).tolist(),
            "physics_backend": "mujoco_rigid",
        }

    def _terminal_reason(self, success: bool, collision: bool, fall_out: bool, timeout: bool) -> str:
        if success:
            return "success"
        if collision:
            return "collision"
        if fall_out:
            return "fall_out"
        if timeout:
            return "timeout"
        return "running"

    def _goal_distance(self, obs: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(obs[:2], dtype=np.float32) - np.asarray(obs[7:9], dtype=np.float32)))

    def _obstacle_clearance(self, obs: np.ndarray) -> float:
        obstacle_distance = float(np.linalg.norm(obs[:2] - obs[9:11]))
        ball_radius = float(self.adapter.preset.sim.ball_radius)
        return float(obstacle_distance - (float(obs[11]) + ball_radius))

    def _route_direction(self, obs: np.ndarray) -> np.ndarray:
        ball = np.asarray(obs[:2], dtype=np.float32)
        goal = np.asarray(obs[7:9], dtype=np.float32)
        obstacle = np.asarray(obs[9:11], dtype=np.float32)
        side = float(self.route_side)
        if ball[1] < obstacle[1] - 0.18:
            waypoint = obstacle + np.asarray([side * 0.32, -0.22], dtype=np.float32)
        elif ball[1] < obstacle[1] + 0.28 and abs(float(ball[0] - obstacle[0])) < 0.34:
            waypoint = obstacle + np.asarray([side * 0.34, 0.28], dtype=np.float32)
        else:
            waypoint = goal
        direction = waypoint - ball
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            direction = goal - ball
            norm = max(float(np.linalg.norm(direction)), 1e-6)
        return (direction / norm).astype(np.float32)

    def _ideal_rod_position(self, obs: np.ndarray, direction: np.ndarray) -> np.ndarray:
        ball = np.asarray(obs[:2], dtype=np.float32)
        gap = (
            float(self.adapter.preset.sim.ball_radius)
            + float(self.adapter.env.physics_config.rod_half_width)
            + 0.012
        )
        return (ball - gap * np.asarray(direction, dtype=np.float32)).astype(np.float32)

    def _select_task(self, options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int | None] | None:
        if "start" in options or "goal" in options:
            if "start" not in options or "goal" not in options:
                raise ValueError("reset options must provide both start and goal.")
            start = np.asarray(options["start"], dtype=np.float32).reshape(2)
            goal = np.asarray(options["goal"], dtype=np.float32).reshape(2)
            pair_id = options.get("pair_id", None)
            return start, goal, None if pair_id is None else int(pair_id)
        if self.fixed_start is not None and self.fixed_goal is not None:
            return self.fixed_start.copy(), self.fixed_goal.copy(), self.fixed_pair_id
        if self.task_pairs:
            index = int(self._task_rng.integers(0, len(self.task_pairs)))
            task = self.task_pairs[index]
            return task["start"].copy(), task["goal"].copy(), int(task["pair_id"])
        return None

    def _normalize_task_pairs(
        self,
        task_pairs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    ) -> list[dict[str, Any]]:
        normalized = []
        for index, item in enumerate(task_pairs or []):
            start = np.asarray(item["start"], dtype=np.float32).reshape(2)
            goal = np.asarray(item["goal"], dtype=np.float32).reshape(2)
            pair_id = int(item.get("pair_id", index))
            normalized.append({"pair_id": pair_id, "start": start, "goal": goal})
        return normalized


def make_mujoco_30deg_env(**kwargs: Any) -> MujocoRodPushGymEnv:
    return MujocoRodPushGymEnv(tilt_radians=math.pi / 6.0, **kwargs)
