"""Nine-condition uphill task. Settling replaces first-entry termination."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv
from mujoco_task.config import get_preset
from mujoco_task.sim.scene import SceneSpec

ROOT = Path(__file__).resolve().parents[2]
TILTS = (-15, 0, 15)
OBSTACLES = (-.20, 0., .20)

def collection():
    return json.loads((ROOT / 'datasets/training/collection_config.json').read_text())

def scene_for(obstacle, lateral, seed=0):
    return SceneSpec(start=[0, -.65], goal=[0, .84], obstacle_center=[obstacle, 0],
                     obstacle_radius=.10, lateral_tilt=np.deg2rad(lateral),
                     longitudinal_tilt=np.deg2rad(20), seed=seed)

class TaskEnv(MujocoRigidTiltPushEnv):
    def __init__(self, max_steps=100):
        c = collection()
        super().__init__(replace(get_preset('run').sim, **{**c['simulator'], 'goal_radius': .12, 'max_steps': max_steps}), c['physics'])
        self.settle_count = 0
    def reset(self, scene):
        self.settle_count = 0
        return super().reset(scene)
    def snapshot(self):
        return self.state.copy(), self.settle_count
    def restore(self, snapshot):
        self.set_state(snapshot[0]); self.settle_count = snapshot[1]
    def step(self, action):
        result = super().step(action)
        inside = np.linalg.norm(result.observation[:2] - self.scene.goal) <= self.config.goal_radius
        slow = np.linalg.norm(result.observation[2:4]) < .10
        safe = not result.collision and not result.info['fall_out']
        self.settle_count = self.settle_count + 1 if inside and slow and safe else 0
        success = self.settle_count >= 5
        timeout = self._step >= self.config.max_steps and not success and safe
        return replace(result, success=success, timeout=timeout,
                       done=bool(success or not safe or timeout),
                       info={**result.info, 'settle_count': self.settle_count, 'inside_goal': bool(inside)})
