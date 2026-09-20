from types import SimpleNamespace
import numpy as np
from evaluation.common import forbid_hypothetical

CASE={'case_id':'test_tilt','base_layout_id':42,'tilt_degrees':[-15,20],
      'tilt_radians':np.deg2rad([-15,20]).tolist(),
      'scene':{'obstacle_center':[0.0,0.0]}}


COLLECTION={'simulator':{'max_steps':100,'dt':.05}}


class FakeEnv:
    def __init__(self,terminal=4):self.actions=[];self.terminal=terminal;self.obs=np.array([0,0,0,0,0,0,0,1,1,0,.6,.1],np.float32)
    def step(self,a):
        self.actions.append(a.copy());self.obs[:2]+=a[:2]*.05;self.obs[2:4]=a[:2]
        done=len(self.actions)==self.terminal
        return SimpleNamespace(observation=self.obs.copy(),success=done,collision=False,timeout=False,done=done,info={'fall_out':False,'rod_ball_contact':True})
    copy=staticmethod(forbid_hypothetical)
    set_state=staticmethod(forbid_hypothetical)


def factory(env):return lambda collection,scene:(env,env.obs.copy())
