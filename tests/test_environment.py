"""Checks of the retained physical backend and online simulator boundary."""
from dataclasses import replace
import numpy as np
import pytest
from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv
from mujoco_task.config import get_preset
from mujoco_task.sim.scene import SceneSpec
from evaluation.common import environment

SCENE=dict(start=[-.45,-.5],goal=[.4,.5],obstacle_center=[0.,0.],obstacle_radius=.1,
           lateral_tilt=0.,longitudinal_tilt=np.deg2rad(20),seed=9)

def backend():
    env=MujocoRigidTiltPushEnv(get_preset('run').sim)
    env.reset(SceneSpec(**SCENE))
    return env

def test_offline_restore_reproduces_actual_steps():
    env=backend();state=env.state.copy();actions=np.array([[.1,.2,.1],[0.,.3,-.2]],np.float32)
    first=[env.step(a).observation.copy() for a in actions]
    env.set_state(state)
    second=[env.step(a).observation.copy() for a in actions]
    np.testing.assert_array_equal(first,second)

def test_board_geometry_matches_tilt():
    env=backend();z=env.data.xmat[env._board_body_id].reshape(3,3)[:,2]
    assert np.arccos(np.clip(z[2],-1,1))==pytest.approx(np.deg2rad(20),abs=1e-3)

def test_online_environment_forbids_hypothetical_rollouts():
    env,obs=environment({'simulator':{'max_steps':100},'physics':{}},SCENE)
    assert obs.shape==(14,)
    with pytest.raises(RuntimeError,match='Hypothetical'):env.copy()
    with pytest.raises(RuntimeError,match='Hypothetical'):env.set_state(None)
    assert np.isfinite(env.step(np.zeros(3)).observation).all()

def test_tilt_causes_downhill_motion():
    env=backend();before=env.current_observation().copy()
    for _ in range(10):after=env.step(np.zeros(3)).observation
    assert after[1]<before[1]-.001
