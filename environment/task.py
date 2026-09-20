"""Canonical uphill push–avoid–settle MuJoCo task."""
from dataclasses import replace
import numpy as np
from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv
from mujoco_task.config import SimulatorConfig
from mujoco_task.sim.scene import SceneSpec
from environment.success import CRITERION, settling

LATERAL_DEGREES=(-15,0,15)
OBSTACLE_X=(-.20,0.,.20)
HIDDEN_CONDITIONS=np.deg2rad(np.array([[-15,20],[0,20],[15,20]],np.float32))
GOAL=np.array([0.,.84],np.float32)
START=np.array([0.,-.65],np.float32)
GEOMETRY_BY_OBSTACLE={x:np.array([0.,.84,x,0.,.10],np.float32) for x in OBSTACLE_X}
SIMULATOR=SimulatorConfig(dt=.05,max_steps=100,workspace_x=(-.8,.8),workspace_y=(-1.,1.),ball_radius=.04,
    obstacle_radius=.10,goal_radius=.12,rod_radius=.03,action_max_speed=.8,action_max_omega=4.,
    gravity_lateral_gain=1.55,gravity_longitudinal_gain=1.25,ball_damping=.22,
    ball_velocity_noise_std=.004,contact_push_gain=7.5,contact_carry_gain=1.,
    controller_position_gain=4.5,controller_feedforward_gain=.75,controller_yaw_gain=5.,
    push_offset=.018,success_speed_threshold=1.5,timeout_distance_penalty=2.)
PHYSICS={'max_ball_speed':4.0,'push_offset':0.0,'rod_workspace_margin':1.2}

def scene(obstacle_x=0.,lateral_degrees=0,seed=0):
    if obstacle_x not in OBSTACLE_X or lateral_degrees not in LATERAL_DEGREES:raise ValueError('Unknown task condition')
    return SceneSpec(start=START,goal=GOAL,obstacle_center=[obstacle_x,0.],obstacle_radius=.1,
        lateral_tilt=np.deg2rad(lateral_degrees),longitudinal_tilt=np.deg2rad(20),seed=seed)

class UphillTask(MujocoRigidTiltPushEnv):
    """Stop after two consecutive safe in-goal observations below 0.15 m/s."""
    def __init__(self,max_steps=100):
        super().__init__(replace(SIMULATOR,max_steps=max_steps),PHYSICS)
        self.settle_count=0
    def reset(self,spec):
        self.settle_count=0
        return super().reset(spec)
    def snapshot(self):return self.state.copy(),self.settle_count
    def restore(self,state):
        self.set_state(state[0]);self.settle_count=int(state[1])
    def step(self,action):
        r=super().step(action);safe=not r.collision and not r.info['fall_out']
        self.settle_count=self.settle_count+1 if safe and settling(r.observation,self.scene.goal) else 0
        success=bool(self.settle_count>=CRITERION.steps)
        timeout=self._step>=self.config.max_steps and not success and safe
        return replace(r,success=success,timeout=timeout,done=bool(success or not safe or timeout),
            info={**r.info,'success_criterion_version':CRITERION.version,
                'inside_goal':bool(np.linalg.norm(r.observation[:2]-self.scene.goal)<=CRITERION.goal_radius),
                'ball_speed_m_s':float(np.linalg.norm(r.observation[2:4])),
                'settle_count':self.settle_count,'success':success})
