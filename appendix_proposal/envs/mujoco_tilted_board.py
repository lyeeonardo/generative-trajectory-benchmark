"""MuJoCo rigid-body backend for the rod-push tilted-board task.

The backend keeps the 14D observation and 3D action layout used by the planners,
but it is now the only environment backend exposed through ``TiltedBoardEnvAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any

import numpy as np

from mujoco_task.config import ACTION_DIM, SimulatorConfig
from mujoco_task.sim.scene import SceneSpec, wrap_angle


@dataclass(frozen=True)
class MujocoRigidConfig:
    """Local physics parameters for the opt-in rigid-body backend."""

    substeps: int = 5
    rod_half_length: float = 0.08
    rod_half_width: float = 0.03
    rod_half_height: float = 0.04
    push_offset: float | None = None
    board_thickness: float = 0.03
    rod_workspace_margin: float = 0.30
    obstacle_height: float = 0.16
    ball_mass: float = 0.05
    gravity_z: float = -9.81
    floor_friction: tuple[float, float, float] = (0.8, 0.02, 0.001)
    ball_friction: tuple[float, float, float] = (1.0, 0.02, 0.001)
    rod_friction: tuple[float, float, float] = (1.5, 0.05, 0.001)
    obstacle_friction: tuple[float, float, float] = (1.0, 0.02, 0.001)
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    max_ball_speed: float = 2.5
    fall_out_z_margin: float = 0.25

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None = None) -> "MujocoRigidConfig":
        if not overrides:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ValueError(f"Unknown MuJoCo rigid physics parameter(s): {unknown}")
        return cls(**dict(overrides))

    def as_dict(self) -> dict[str, Any]:
        return {
            "substeps": int(self.substeps),
            "rod_half_length": float(self.rod_half_length),
            "rod_half_width": float(self.rod_half_width),
            "rod_half_height": float(self.rod_half_height),
            "push_offset": None if self.push_offset is None else float(self.push_offset),
            "board_thickness": float(self.board_thickness),
            "rod_workspace_margin": float(self.rod_workspace_margin),
            "obstacle_height": float(self.obstacle_height),
            "ball_mass": float(self.ball_mass),
            "gravity_z": float(self.gravity_z),
            "floor_friction": tuple(float(v) for v in self.floor_friction),
            "ball_friction": tuple(float(v) for v in self.ball_friction),
            "rod_friction": tuple(float(v) for v in self.rod_friction),
            "obstacle_friction": tuple(float(v) for v in self.obstacle_friction),
            "solref": tuple(float(v) for v in self.solref),
            "solimp": tuple(float(v) for v in self.solimp),
            "max_ball_speed": float(self.max_ball_speed),
            "fall_out_z_margin": float(self.fall_out_z_margin),
            "walls_enabled": False,
            "physical_board_tilt": True,
        }


@dataclass
class MujocoRigidState:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    rod_yaw: float
    step: int = 0
    time: float = 0.0

    def copy(self) -> "MujocoRigidState":
        return MujocoRigidState(
            qpos=self.qpos.copy(),
            qvel=self.qvel.copy(),
            mocap_pos=self.mocap_pos.copy(),
            mocap_quat=self.mocap_quat.copy(),
            rod_yaw=float(self.rod_yaw),
            step=int(self.step),
            time=float(self.time),
        )


@dataclass(frozen=True)
class MujocoRigidStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    success: bool
    collision: bool
    timeout: bool
    info: dict[str, float | bool | str]


def _load_mujoco():
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "physics_backend='mujoco_rigid' requires the mujoco Python package. "
            "Use the RL conda environment or install mujoco."
        ) from exc
    return mujoco


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def board_rotation(scene: SceneSpec) -> np.ndarray:
    """Board-local to world rotation.

    Positive longitudinal tilt makes positive board-y higher, so gravity rolls
    the ball toward negative y, matching the previous force convention.
    """

    return _rot_y(scene.lateral_tilt) @ _rot_x(scene.longitudinal_tilt)


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _board_quat(scene: SceneSpec) -> np.ndarray:
    return _quat_from_matrix(board_rotation(scene))


def _rod_quat(scene: SceneSpec, yaw: float) -> np.ndarray:
    return _quat_from_matrix(board_rotation(scene) @ _rot_z(yaw))


def _rod_yaw_from_quat(scene: SceneSpec, quat: np.ndarray) -> float:
    local_matrix = board_rotation(scene).T @ _matrix_from_quat(quat)
    return wrap_angle(math.atan2(float(local_matrix[1, 0]), float(local_matrix[0, 0])))


def _xml_for_scene(scene: SceneSpec, sim: SimulatorConfig, cfg: MujocoRigidConfig) -> str:
    x_min, x_max = sim.workspace_x
    y_min, y_max = sim.workspace_y
    x_half = 0.5 * (x_max - x_min)
    y_half = 0.5 * (y_max - y_min)
    obstacle_half_h = 0.5 * cfg.obstacle_height
    timestep = float(sim.dt) / max(int(cfg.substeps), 1)
    board_half_h = 0.5 * cfg.board_thickness
    rod_z = float(cfg.rod_half_height)
    board_quat = _board_quat(scene)
    return f"""
<mujoco model="iwai_rigid_tilted_board">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{timestep:.9g}" gravity="0 0 {float(cfg.gravity_z):.9g}" integrator="Euler" cone="elliptic" iterations="80"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <size nconmax="256" njmax="256"/>
  <default>
    <geom solref="{_fmt(cfg.solref)}" solimp="{_fmt(cfg.solimp)}" margin="0.0005"/>
  </default>
  <worldbody>
    <light name="key_light" pos="-2 -3 4" dir="2 3 -4" diffuse="0.9 0.9 0.9"/>
    <camera name="orbit" mode="fixed" pos="2.3 -2.7 1.8" xyaxes="0.76 0.65 0 -0.32 0.38 0.86"/>
    <body name="board" pos="0 0 0" quat="{_fmt(board_quat)}">
      <geom name="board_geom" type="box" pos="0 0 {-board_half_h:.9g}" size="{_fmt((x_half, y_half, board_half_h))}" contype="1" conaffinity="1" friction="{_fmt(cfg.floor_friction)}" rgba="0.92 0.89 0.82 1"/>
      <geom name="goal_geom" type="cylinder" pos="{_fmt((float(scene.goal[0]), float(scene.goal[1]), 0.006))}" size="{float(sim.goal_radius):.9g} 0.006" contype="0" conaffinity="0" rgba="0.25 0.72 0.35 0.35"/>
      <geom name="obstacle_geom" type="cylinder" pos="{_fmt((float(scene.obstacle_center[0]), float(scene.obstacle_center[1]), obstacle_half_h))}" size="{float(scene.obstacle_radius):.9g} {obstacle_half_h:.9g}" contype="1" conaffinity="1" friction="{_fmt(cfg.obstacle_friction)}" rgba="0.85 0.1 0.15 1"/>
    </body>
    <body name="ball" pos="0 0 0">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{float(sim.ball_radius):.9g}" mass="{float(cfg.ball_mass):.9g}" contype="1" conaffinity="1" friction="{_fmt(cfg.ball_friction)}" rgba="0.1 0.45 0.95 1"/>
    </body>
    <body name="rod" mocap="true" pos="0 0 {rod_z:.9g}">
      <geom name="rod_geom" type="box" size="{_fmt((cfg.rod_half_length, cfg.rod_half_width, cfg.rod_half_height))}" contype="1" conaffinity="1" friction="{_fmt(cfg.rod_friction)}" rgba="0.95 0.45 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def compose_mujoco_observation(state: MujocoRigidState, scene: SceneSpec) -> np.ndarray:
    rotation = board_rotation(scene)
    ball_pos_local = rotation.T @ np.asarray(state.qpos[0:3], dtype=np.float64)
    ball_vel_local = rotation.T @ np.asarray(state.qvel[0:3], dtype=np.float64)
    rod_pos_local = rotation.T @ np.asarray(state.mocap_pos[0], dtype=np.float64)
    return np.asarray(
        [
            ball_pos_local[0],
            ball_pos_local[1],
            ball_vel_local[0],
            ball_vel_local[1],
            rod_pos_local[0],
            rod_pos_local[1],
            float(state.rod_yaw),
            scene.goal[0],
            scene.goal[1],
            scene.obstacle_center[0],
            scene.obstacle_center[1],
            float(scene.obstacle_radius),
            float(scene.lateral_tilt),
            float(scene.longitudinal_tilt),
        ],
        dtype=np.float32,
    )


class MujocoRigidTiltPushEnv:
    """Planar rigid-body rod-push environment backed by MuJoCo."""

    def __init__(self, config: SimulatorConfig, physics_config: dict[str, Any] | MujocoRigidConfig | None = None) -> None:
        self.config = config
        self.physics_config = (
            physics_config
            if isinstance(physics_config, MujocoRigidConfig)
            else MujocoRigidConfig.from_overrides(physics_config)
        )
        self._mujoco = _load_mujoco()
        self.scene: SceneSpec | None = None
        self.model = None
        self.data = None
        self.state: MujocoRigidState | None = None
        self._ball_body_id = -1
        self._board_body_id = -1
        self._rod_body_id = -1
        self._rod_mocap_id = -1
        self._geom_ids: dict[str, int] = {}
        self._rod_yaw = 0.0
        self._step = 0

    @property
    def backend_name(self) -> str:
        return "mujoco_rigid"

    def reset(self, scene: SceneSpec) -> np.ndarray:
        self.scene = scene
        self.model = self._mujoco.MjModel.from_xml_string(_xml_for_scene(scene, self.config, self.physics_config))
        self.data = self._mujoco.MjData(self.model)
        self._cache_ids()
        push_dir = self._goal_direction(scene)
        push_yaw = float(math.atan2(float(push_dir[1]), float(push_dir[0])))
        push_offset = self.config.push_offset if self.physics_config.push_offset is None else self.physics_config.push_offset
        behind = (
            float(self.config.ball_radius)
            + float(self.physics_config.rod_half_width)
            + float(push_offset)
        )
        rod_pos_local = np.asarray(
            [
                float(scene.start[0] - behind * push_dir[0]),
                float(scene.start[1] - behind * push_dir[1]),
                float(self.physics_config.rod_half_height),
            ],
            dtype=np.float64,
        )
        self._rod_yaw = wrap_angle(push_yaw + 0.5 * math.pi)
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = self._local_to_world(
            np.asarray([scene.start[0], scene.start[1], self.config.ball_radius], dtype=np.float64)
        )
        self.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        self._set_rod_pose_local(rod_pos_local, self._rod_yaw)
        self._step = 0
        self.data.time = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self._stabilize_ball_speed()
        self._sync_state()
        return self.current_observation()

    def copy(self) -> "MujocoRigidTiltPushEnv":
        if self.scene is None or self.state is None:
            raise RuntimeError("Cannot copy before reset.")
        env = MujocoRigidTiltPushEnv(self.config, self.physics_config)
        env.reset(self.scene)
        env.set_state(self.state)
        return env

    def set_state(self, state: MujocoRigidState) -> None:
        if self.scene is None:
            raise RuntimeError("Cannot restore MuJoCo state before reset(scene).")
        if self.model is None or self.data is None:
            self.reset(self.scene)
        assert self.model is not None and self.data is not None
        self.data.qpos[:] = np.asarray(state.qpos, dtype=np.float64).reshape(self.model.nq)
        self.data.qvel[:] = np.asarray(state.qvel, dtype=np.float64).reshape(self.model.nv)
        self.data.mocap_pos[:] = np.asarray(state.mocap_pos, dtype=np.float64).reshape(self.model.nmocap, 3)
        self.data.mocap_quat[:] = np.asarray(state.mocap_quat, dtype=np.float64).reshape(self.model.nmocap, 4)
        self._rod_yaw = float(state.rod_yaw)
        self._step = int(state.step)
        self.data.time = float(state.time)
        self._mujoco.mj_forward(self.model, self.data)
        self._stabilize_ball_speed()
        self._sync_state()

    def current_observation(self) -> np.ndarray:
        if self.scene is None or self.state is None:
            raise RuntimeError("Environment is not reset.")
        self._sync_state()
        return compose_mujoco_observation(self.state, self.scene)

    def step(self, action: np.ndarray) -> MujocoRigidStepResult:
        if self.scene is None or self.model is None or self.data is None:
            raise RuntimeError("Environment is not reset.")
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
        xy_speed = float(np.linalg.norm(action[:2]))
        if xy_speed > self.config.action_max_speed:
            action[:2] = action[:2] / xy_speed * self.config.action_max_speed
        action[2] = float(np.clip(action[2], -self.config.action_max_omega, self.config.action_max_omega))

        dt = float(self.config.dt) / max(int(self.physics_config.substeps), 1)
        collision_seen = self._ball_obstacle_collision_now()
        fall_out_seen = self._fall_out_now()
        for _ in range(max(int(self.physics_config.substeps), 1)):
            self._rod_yaw = wrap_angle(self._rod_yaw + dt * float(action[2]))
            rod_pos_local = self._rod_position_local()
            rod_pos_local[:2] += np.asarray(action[:2], dtype=np.float64) * dt
            margin = float(self.physics_config.rod_workspace_margin)
            rod_pos_local[0] = float(np.clip(rod_pos_local[0], self.config.workspace_x[0] - margin, self.config.workspace_x[1] + margin))
            rod_pos_local[1] = float(np.clip(rod_pos_local[1], self.config.workspace_y[0] - margin, self.config.workspace_y[1] + margin))
            rod_pos_local[2] = float(self.physics_config.rod_half_height)
            self._set_rod_pose_local(rod_pos_local, self._rod_yaw)
            self._apply_tilt_and_damping()
            self._mujoco.mj_step(self.model, self.data)
            self._stabilize_ball_speed()
            collision_seen = collision_seen or self._ball_obstacle_collision_now()
            fall_out_seen = fall_out_seen or self._fall_out_now()
        self.data.xfrc_applied[:] = 0.0
        self._step += 1
        self._sync_state()

        obs = self.current_observation()
        goal_distance = float(np.linalg.norm(obs[:2] - self.scene.goal))
        obstacle_distance = float(np.linalg.norm(obs[:2] - self.scene.obstacle_center))
        fall_out = bool(fall_out_seen or self._is_fall_out(obs))
        ball_obstacle_collision = bool(
            obstacle_distance <= float(self.scene.obstacle_radius) + float(self.config.ball_radius)
        ) or self._has_contact("ball_geom", "obstacle_geom") or collision_seen
        collision = bool(ball_obstacle_collision)
        success = bool(goal_distance <= self.config.goal_radius and not collision and not fall_out)
        timeout = bool(self._step >= self.config.max_steps and not success and not collision and not fall_out)
        done = bool(success or collision or fall_out or timeout)
        reward = -goal_distance
        if collision or fall_out:
            reward -= self.config.timeout_distance_penalty
        if success:
            reward += 5.0
        info = {
            "goal_distance": goal_distance,
            "obstacle_distance": obstacle_distance,
            "step": float(self._step),
            "physics_backend": self.backend_name,
            "ball_obstacle_collision": bool(ball_obstacle_collision),
            "ball_wall_collision": False,
            "fall_out": bool(fall_out),
            "rod_ball_contact": bool(self._has_contact("ball_geom", "rod_geom")),
        }
        return MujocoRigidStepResult(
            observation=obs,
            reward=float(reward),
            done=done,
            success=bool(success),
            collision=bool(collision),
            timeout=bool(timeout),
            info=info,
        )

    def _cache_ids(self) -> None:
        assert self.model is not None
        mujoco = self._mujoco
        self._ball_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"))
        self._board_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "board"))
        self._rod_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rod"))
        self._rod_mocap_id = int(self.model.body_mocapid[self._rod_body_id])
        names = ("ball_geom", "rod_geom", "obstacle_geom", "board_geom", "goal_geom")
        self._geom_ids = {
            name: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in names
        }

    def _goal_direction(self, scene: SceneSpec) -> np.ndarray:
        direction = np.asarray(scene.goal - scene.start, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return np.asarray([0.0, 1.0], dtype=np.float64)
        return direction / norm

    def board_rotation(self) -> np.ndarray:
        if self.scene is None:
            raise RuntimeError("Environment is not reset.")
        return board_rotation(self.scene)

    def _local_to_world(self, local_xyz: np.ndarray) -> np.ndarray:
        return self.board_rotation() @ np.asarray(local_xyz, dtype=np.float64).reshape(3)

    def _world_to_local(self, world_xyz: np.ndarray) -> np.ndarray:
        return self.board_rotation().T @ np.asarray(world_xyz, dtype=np.float64).reshape(3)

    def _rod_position_local(self) -> np.ndarray:
        assert self.data is not None
        return self._world_to_local(self.data.mocap_pos[self._rod_mocap_id])

    def _set_rod_pose_local(self, local_xyz: np.ndarray, yaw: float) -> None:
        assert self.data is not None and self.scene is not None
        self.data.mocap_pos[self._rod_mocap_id] = self._local_to_world(local_xyz)
        self.data.mocap_quat[self._rod_mocap_id] = _rod_quat(self.scene, yaw)

    def _apply_tilt_and_damping(self) -> None:
        assert self.model is not None and self.data is not None and self.scene is not None
        self.data.xfrc_applied[:] = 0.0
        ball_vel_local = self.board_rotation().T @ np.asarray(self.data.qvel[0:3], dtype=np.float64)
        damping_local = np.asarray(
            [-self.config.ball_damping * ball_vel_local[0], -self.config.ball_damping * ball_vel_local[1], 0.0],
            dtype=np.float64,
        )
        mass = float(self.model.body_mass[self._ball_body_id])
        self.data.xfrc_applied[self._ball_body_id, 0:3] = mass * (self.board_rotation() @ damping_local)

    def _stabilize_ball_speed(self) -> None:
        assert self.model is not None and self.data is not None
        speed = float(np.linalg.norm(self.data.qvel[0:3]))
        max_speed = float(self.physics_config.max_ball_speed)
        if speed > max_speed > 0.0:
            self.data.qvel[0:3] *= max_speed / speed
        self._mujoco.mj_forward(self.model, self.data)

    def _is_fall_out(self, obs: np.ndarray) -> bool:
        local_z = float(self._world_to_local(self.data.qpos[0:3])[2]) if self.data is not None else 0.0
        return bool(
            float(obs[0]) < float(self.config.workspace_x[0])
            or float(obs[0]) > float(self.config.workspace_x[1])
            or float(obs[1]) < float(self.config.workspace_y[0])
            or float(obs[1]) > float(self.config.workspace_y[1])
            or local_z < -float(self.physics_config.fall_out_z_margin)
        )

    def _fall_out_now(self) -> bool:
        if self.data is None:
            return False
        local = self._world_to_local(self.data.qpos[0:3])
        return bool(
            float(local[0]) < float(self.config.workspace_x[0])
            or float(local[0]) > float(self.config.workspace_x[1])
            or float(local[1]) < float(self.config.workspace_y[0])
            or float(local[1]) > float(self.config.workspace_y[1])
            or float(local[2]) < -float(self.physics_config.fall_out_z_margin)
        )

    def _ball_obstacle_collision_now(self) -> bool:
        if self.data is None or self.scene is None:
            return False
        xy = self._world_to_local(self.data.qpos[0:3])[:2]
        center = np.asarray(self.scene.obstacle_center, dtype=np.float64)
        distance = float(np.linalg.norm(xy - center))
        overlap = distance <= float(self.scene.obstacle_radius) + float(self.config.ball_radius)
        return bool(overlap or self._has_contact("ball_geom", "obstacle_geom"))

    def _sync_state(self) -> None:
        assert self.model is not None and self.data is not None
        if self.scene is not None:
            self._rod_yaw = _rod_yaw_from_quat(self.scene, self.data.mocap_quat[self._rod_mocap_id])
        self.state = MujocoRigidState(
            qpos=np.asarray(self.data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(self.data.qvel, dtype=np.float64).copy(),
            mocap_pos=np.asarray(self.data.mocap_pos, dtype=np.float64).copy(),
            mocap_quat=np.asarray(self.data.mocap_quat, dtype=np.float64).copy(),
            rod_yaw=float(self._rod_yaw),
            step=int(self._step),
            time=float(self.data.time),
        )

    def _has_contact(self, geom_a: str, geom_b: str) -> bool:
        if self.data is None:
            return False
        a = self._geom_ids[geom_a]
        b = self._geom_ids[geom_b]
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if (int(contact.geom1) == a and int(contact.geom2) == b) or (
                int(contact.geom1) == b and int(contact.geom2) == a
            ):
                return True
        return False
