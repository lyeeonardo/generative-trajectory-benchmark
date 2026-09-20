import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from data.joint_training import JointTrainingData,encode_batch
from generators.registry import create_model,available_models
from tests.test_joint_model import batch,NORM

@pytest.mark.parametrize('method',available_models())
def test_future_constraint_affects_earlier_action_without_leaking_unknown_targets(method):
    torch.manual_seed(12)
    cfg=dict(method=method,width=32,layers=1,heads=4)
    cfg.update(posterior_layers=1,latent_dim=4) if method=='cvae' else cfg.update(sampling_steps=2)
    m=create_model(cfg,NORM).eval();b=batch()
    c={k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
    actions=torch.zeros(1,6,3);obs=torch.zeros(1,6,7);ka=torch.zeros_like(actions,dtype=torch.bool);ko=torch.zeros_like(obs,dtype=torch.bool)
    ko[:,5,:2]=True;ka[:,0,0]=True;actions[:,0,0]=.7
    rng=lambda:torch.Generator().manual_seed(77)
    a=m.complete_joint(c,actions,obs,ka,ko,generator=rng())
    changed=obs.clone();changed[:,5,0]=.3
    d=m.complete_joint(c,actions,changed,ka,ko,generator=rng())
    assert not torch.equal(a['actions'][:,:,0],d['actions'][:,:,0])
    assert a['actions'][0,0,0,0]==actions[0,0,0]
    assert torch.linalg.vector_norm(a['actions'][...,:2],dim=-1).max()<=.800001
    torch.testing.assert_close(d['joint_observations'][0,0,5,:2],changed[0,5,:2],atol=1e-6,rtol=0)
    hidden=changed.clone();hidden[~ko]=999
    e=m.complete_joint(c,actions,hidden,ka,ko,generator=rng())
    for k in d:assert torch.equal(d[k],e[k])


def test_settling_sampler_coverage_short_targets_and_resume():
    data=JointTrainingData(Path('datasets/uphill_push_v1/cache'),resident=False)
    for _ in range(16):
        b=data.sample(2048);e=encode_batch(b,data.normalization)
        assert not (e['known'] & ~e['semantic']).any()
        assert not (b['known_observations'] & ~b['valid'][...,None]).any()
        assert not e['loss_mask'][~b['valid'].repeat_interleave(2,1)].any()
    report=data.exposure
    assert report['covered_scenarios_by_role']==[9]*4
    assert (np.asarray(report['role_phase_counts'])>0).all()
    assert (np.asarray(report['condition_phase_counts'])>0).all()
    assert report['high_reference_draws']>0 and report['short_terminal_draws']>0
    assert min(report['completion_masks'])>0
    saved=data.state_dict();a=data.sample(256);data.load_state_dict(saved);b=data.sample(256)
    for k in a:assert torch.equal(a[k],b[k])


def test_projected_action_forecast_uses_actual_actions_not_raw_decoder():
    import types
    from generators.joint_world_model import JointWorldModel,JointConfig
    m=JointWorldModel(JointConfig(width=16,layers=1,heads=2,sampling_steps=1),NORM)
    b=batch();c={k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
    def decoder(self,x,e,t):
        value=torch.zeros_like(x);fixed=e['role']!=0
        value[:,::2,0]=3. # Native decoder proposes illegal planar action.
        value[:,1::2,0]=99.
        value[fixed,1::2,0]=e['x'][fixed,::2,0]
        return value,torch.zeros(len(x),6,5)
    m.forward=types.MethodType(decoder,m)
    out=m.propose_joint(c,generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(out['actions'][0,0,:,0],torch.full((6,),.8))
    now=c['history_observations'][0,-1,0]
    torch.testing.assert_close(out['observations'][0,0,:,0],torch.ones(6)+now)
    torch.testing.assert_close(out['joint_observations'][0,0,:,0],torch.full((6,),99.)+now)
