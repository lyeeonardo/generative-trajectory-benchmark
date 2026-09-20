from dataclasses import replace
import numpy as np
import pytest
from environment.task import UphillTask,scene
from environment.success import CRITERION,score_trace
from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv,MujocoRigidStepResult


def fake_step(monkeypatch,env,speeds,*,collision_at=None,fall_at=None,outside_at=None):
    values=iter(speeds)
    def step(self,action):
        self._step+=1;speed=next(values);index=self._step
        obs=np.zeros(14,np.float32);obs[:2]=self.scene.goal if index!=outside_at else [0,.5];obs[2]=speed
        collision=index==collision_at;fall=index==fall_at
        return MujocoRigidStepResult(obs,0,False,False,collision,False,{'fall_out':fall,'rod_ball_contact':True})
    monkeypatch.setattr(MujocoRigidTiltPushEnv,'step',step)


def test_success_requires_two_consecutive_safe_steps(monkeypatch):
    env=UphillTask();env.reset(scene());fake_step(monkeypatch,env,[.149,.149])
    first=env.step(np.zeros(3));assert not first.done and first.info['settle_count']==1
    second=env.step(np.zeros(3));assert second.success and second.done
    assert 'strict_secondary_success' not in second.info


def test_threshold_is_strict_and_departure_resets(monkeypatch):
    env=UphillTask();env.reset(scene());fake_step(monkeypatch,env,[.15,.14,.14],outside_at=2)
    assert env.step(np.zeros(3)).info['settle_count']==0
    assert env.step(np.zeros(3)).info['settle_count']==0
    assert env.step(np.zeros(3)).info['settle_count']==1


@pytest.mark.parametrize('failure',['collision','fall'])
def test_safety_precedes_success(monkeypatch,failure):
    env=UphillTask();env.reset(scene())
    fake_step(monkeypatch,env,[.01,.01],collision_at=2 if failure=='collision' else None,fall_at=2 if failure=='fall' else None)
    assert not env.step(np.zeros(3)).done
    r=env.step(np.zeros(3));assert r.done and not r.success


def test_snapshot_restores_current_dwell_count():
    env=UphillTask();env.reset(scene());env.settle_count=1
    snapshot=env.snapshot();env.settle_count=0;env.restore(snapshot)
    assert env.settle_count==1


def test_trace_stops_at_success_or_first_safety_failure():
    obs=np.zeros((6,7),np.float32);obs[:,1]=.84;obs[1:,2]=[.14,.14,.11,.09,.09]
    result=score_trace(obs,[0,.84]);assert result['success'] and result['success_step']==2
    assert not score_trace(obs,[0,.84],[True,False,True,True,True])['success']


def predicted(positions,speeds,*,past=(),collision=None,fall=None,step=None):
    from aif.scoring import estimate_cost
    from evaluation.common import public_context
    h=len(speeds);y=np.zeros((1,2,h,7),np.float32);y[0,:,:,1]=positions;y[0,:,:,2]=speeds
    history=[np.array([0,.5,0,0,0,0,0],np.float32)]+[np.array([0,y,v,0,0,0,0],np.float32) for y,v in past]
    ctx=public_context(history,[np.zeros(3)]*len(past),[0,.84,0,0,.1],[0,0],len(past) if step is None else step)
    p=np.zeros((1,2,h,5));p[...,0]=1 # Learned success head cannot shortcut the contract.
    if collision is not None:p[:,:,collision,1]=1
    if fall is not None:p[:,:,fall,2]=1
    return estimate_cost(dict(observations=y,event_probabilities=p),ctx,np.zeros((1,h,3)),event_uniforms=np.full(p.shape,.5))


def test_fast_goal_entry_does_not_end_prediction_or_hide_later_risk():
    r=predicted([.84]*6,[.3]*6,collision=4)
    assert not r['events'][...,0].any()
    assert r['valid'][0,0].tolist()==[True]*5+[False]
    assert r['risk'][0]==1


def test_predicted_departure_resets_dwell_and_short_horizon_is_not_padded():
    r=predicted([.84,.6,.84],[.1]*3)
    assert not r['events'][...,0].any() and r['valid'].all()
    assert not predicted([.84],[.1])['events'][...,0].any()


def test_observed_dwell_carries_into_prediction_but_initial_state_does_not():
    r=predicted([.84],[.1],past=[(.84,.1)])
    assert r['events'][0,0,0,0]
    from environment.success import history_count
    assert history_count(np.array([[0,.84,0,0]]),[0,.84],elapsed_steps=0)==0


@pytest.mark.parametrize('failure',['collision','fall'])
def test_predicted_safety_precedes_success(failure):
    r=predicted([.84],[.1],past=[(.84,.1)],**{failure:0})
    assert not r['events'][...,0].any() and r['risk'][0]==1


def test_success_precedes_timeout_and_timeout_never_pads():
    r=predicted([.84],[.1],past=[(.84,.1)],step=99)
    assert r['events'][0,0,0,0] and not r['events'][0,0,0,3]
    r=predicted([.84],[.3],step=99)
    assert not r['events'][0,0,0,0] and r['events'][0,0,0,3]


def test_offline_capability_execution_uses_same_stopping_rule(tmp_path,monkeypatch):
    import json
    import data.banks as banks
    from evaluation.common import public_context
    case='settling';(tmp_path/'private').mkdir()
    (tmp_path/'private'/f'{case}.json').write_text(json.dumps({'case':{'scene':{'goal':[0,.84]}},'states':[{}]}))
    monkeypatch.setattr(banks,'BANK',tmp_path)
    monkeypatch.setattr(banks,'restore',lambda env,state:None)
    from types import SimpleNamespace
    class Env:
        def __init__(self):self.settle_count=0
        def reset(self,scene):pass
        def step(self,action):
            self.settle_count+=1;success=self.settle_count>=2
            return SimpleNamespace(observation=np.array([0,.84,.1,0,0,0,0],np.float32),success=success,done=success,collision=False,timeout=False,info={'fall_out':False,'rod_ball_contact':False})
    import environment.task
    monkeypatch.setattr(environment.task,'UphillTask',Env)
    # SceneSpec has required geometry fields; use the real public scene definition.
    from dataclasses import asdict
    spec=asdict(scene())
    (tmp_path/'private'/f'{case}.json').write_text(json.dumps({'case':{'scene':spec},'states':[{}]},default=lambda x:x.tolist()))
    context=public_context([np.array([0,.84,.1,0,0,0,0],np.float32)],[],[0,.84,0,0,.1],[0,0],0)
    result=banks.offline_actions([{'case_id':case,'anchor':0}],np.zeros((1,1,6,3),np.float32),context,tmp_path/'truth.npz')
    assert result['actual_steps']==2
    assert result['valid'][0,0].tolist()==[True,True,False,False,False,False]
    assert result['events'][0,0,1,0]
