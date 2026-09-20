"""Training budgets, exact interruption recovery, and scenario coverage."""
import json
from pathlib import Path
import numpy as np
import torch
import pytest
from data.joint_training import JointTrainingData, encode_batch

ROOT=Path(__file__).resolve().parents[1]

def test_warm_start_architecture_guard_ignores_only_non_weight_controls():
    from training.train_joint import weight_architecture_config
    base={'method':'autoregressive','width':384,'layers':8,'heads':8,'mixtures':16,
          'sampling_steps':8,'first_action_loss_weight':10.,'query_loss_weights':[.45,.25,.2,.1]}
    tuned={**base,'sampling_steps':16,'first_action_loss_weight':20.,'query_loss_weights':[.6,.15,.15,.1]}
    assert weight_architecture_config(base)==weight_architecture_config(tuned)
    for field,value in [('width',256),('layers',6),('heads',4),('mixtures',8),('method','diffusion')]:
        changed={**tuned,field:value}
        assert weight_architecture_config(base)!=weight_architecture_config(changed)

def test_extension_and_checkpoint_rules():
    from training.train_joint import should_extend,checkpoint_rank
    h=[{'windows':w,'raw_action_usefulness':u,'fixed_prediction_rmse':v} for w,u,v in [(80,.1,1.),(85,.1,1.),(95,.102,.99),(100,.102,.99)]]
    yes,why=should_extend(h,100,100);assert yes
    flat=[{**m,'raw_action_usefulness':.1,'fixed_prediction_rmse':1.} for m in h]
    assert not should_extend(flat,100,100)[0]
    assert not should_extend([],100,100)[0]
    a={'prediction_qualified':True,'raw_action_usefulness':.1,'one_step_nll':7.,'windows':2}
    b={**a,'prediction_qualified':False,'raw_action_usefulness':1.}
    assert checkpoint_rank(a)>checkpoint_rank(b)

@pytest.mark.parametrize('method',['diffusion','flow_matching','autoregressive'])
def test_actual_sampler_exposure_and_exact_resume(tmp_path,method):
    from training.train_joint import train
    from scripts.experiment1_full import recipe
    config=recipe(method, 13)
    config['model'].update(width=32,layers=1,heads=4,sampling_steps=2)
    config['training'].update(batch_size=8,checkpoint_windows=8,ema_decay=.9)
    train(config,tmp_path/'continuous',stage='smoke',smoke_steps=4,device='cpu')
    train(config,tmp_path/'resumed',stage='smoke',smoke_steps=4,device='cpu',stop_after_updates=2)
    train(config,tmp_path/'resumed',stage='smoke',smoke_steps=4,device='cpu',resume=True)
    a=torch.load(tmp_path/'continuous/latest.pt',weights_only=False,map_location='cpu')
    b=torch.load(tmp_path/'resumed/latest.pt',weights_only=False,map_location='cpu')
    assert a['windows']==b['windows']==32 and a['updates']==b['updates']==4
    assert a['sampler']['rng']==b['sampler']['rng'];assert a['sampler']['exposure']==b['sampler']['exposure']
    assert np.array_equal(a['sampler']['seen'],b['sampler']['seen'])
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    for k in a['model']:assert torch.equal(a['model'][k],b['model'][k]),k
    assert a['ema_decay']==b['ema_decay']==.9
    for k in a['ema_model']:assert torch.equal(a['ema_model'][k],b['ema_model'][k]),k
    for p in a['optimizer']['state']:
        for k in a['optimizer']['state'][p]:assert torch.equal(a['optimizer']['state'][p][k],b['optimizer']['state'][p][k])

def test_all_scenarios_all_roles_HIGH_coverage_and_exact_resume():
 torch.set_num_threads(1)
 d=JointTrainingData(ROOT/'datasets/uphill_push_v1/cache',seed=13)
 for _ in range(64):
  b=d.sample(2048);e=encode_batch(b,d.normalization)
  fixed=(b['role']==1)|(b['role']==2)
  assert not b['quality'][fixed].any()
  assert not e['loss_mask'][fixed,::2].any()
  prefix=b['role']==3
  assert (b['known_steps'][prefix]==0).all()
  assert not (b['known_observations'][prefix] & ~b['valid'][prefix,:,None]).any()
 assert d.exposure['covered_scenarios_by_role']==[9]*4
 assert d.exposure['HIGH_applied_scenarios']==9
 saved=d.state_dict();a=d.sample(2048);expected=d.state_dict();d.load_state_dict(saved);b=d.sample(2048)
 for k in a:assert torch.equal(a[k],b[k]),k
 actual=d.state_dict()
 assert expected['rng']==actual['rng'];assert expected['exposure']==actual['exposure']
 assert np.array_equal(expected['scenario_role_counts'],actual['scenario_role_counts'])
 assert np.array_equal(expected['scenario_HIGH_counts'],actual['scenario_HIGH_counts'])
 assert np.array_equal(expected['seen'],actual['seen'])


def test_frozen_full_contract_keeps_equal_exposure_and_long_progress_intervals():
    from scripts.experiment1_full import recipe
    for method in ("diffusion", "flow_matching", "autoregressive", "cvae"):
        config = recipe(method, 13)
        assert config["training"]["initial_windows"] == 53_705_000
        assert config["training"]["maximum_windows"] == 53_705_000
        assert config["training"]["progress_seconds"] == 600
        assert config["full_training_ready"] is True


def test_measured_branch_shift_moves_only_recorded_prefixes():
    from data.joint_training import MODEL_FIELDS,shift_measured_windows
    data=JointTrainingData(ROOT/'datasets/uphill_push_v1/cache',seed=7)
    raw=data.raw
    candidates=np.flatnonzero((raw['source']==0)&(raw['valid'].sum(1)>=4))
    assert len(candidates)
    row=int(candidates[0])
    batch={k:torch.as_tensor(np.asarray(raw[k][row:row+1]).copy()) for k in MODEL_FIELDS}
    shifted=shift_measured_windows(batch,torch.tensor([3]))
    torch.testing.assert_close(shifted['history_observations'][0,-1],batch['future_observations'][0,2])
    torch.testing.assert_close(shifted['history_actions'][0,-1],batch['actions'][0,2])
    torch.testing.assert_close(shifted['future_observations'][0,0],batch['future_observations'][0,3])
    torch.testing.assert_close(shifted['actions'][0,0],batch['actions'][0,3])
    assert shifted['valid'][0].sum()==batch['valid'][0].sum()-3
    assert not shifted['valid'][0,-3:].any()
    assert not shifted['event_known'][0,-3:].any()
    torch.testing.assert_close(shifted['time_fraction'],batch['time_fraction']+.03)


def test_branch_shift_sampler_is_explicit_logged_and_exactly_resumable():
    data=JointTrainingData(ROOT/'datasets/uphill_push_v1/cache',seed=13,
                           high_success_demo_fraction=1.,branch_shift_fraction=.5)
    first=data.sample(2048);offset=first['branch_shift_offset']
    assert (offset>0).any()
    assert ((offset==0)|(first['role']==0)|(first['role']==3)).all()
    assert first['valid'][:,0].all()
    assert data.exposure['shifted_branch_HIGH_draws']==int((offset>0).sum())
    assert sum(data.exposure['branch_shift_offsets'][1:])==data.exposure['shifted_branch_HIGH_draws']
    saved=data.state_dict();expected=data.sample(128);end=data.state_dict()
    data.load_state_dict(saved);actual=data.sample(128)
    for key in expected:assert torch.equal(expected[key],actual[key]),key
    assert data.state_dict()['rng']==end['rng']
    assert data.state_dict()['exposure']==end['exposure']


def test_canonical_denoiser_retention_preserves_identical_teacher():
    import copy
    from generators.joint_world_model import JointConfig,JointWorldModel
    from tests.test_joint_model import batch,NORM
    from training.train_joint import denoiser_retention
    model=JointWorldModel(JointConfig(width=16,layers=1,heads=2,sampling_steps=2),NORM)
    teacher=copy.deepcopy(model).eval();teacher.requires_grad_(False)
    value=batch();value['geometry'][:,2]=0;value['tilt'][:,0]=0
    torch.manual_seed(11);same=denoiser_retention(model,teacher,value)
    assert same.item()==0
    with torch.no_grad():model.output.weight.add_(.01)
    torch.manual_seed(11);changed=denoiser_retention(model,teacher,value)
    assert changed.item()>0


def test_canonical_autoregressive_retention_preserves_teacher_distribution():
    import copy
    from generators.joint_world_model import JointConfig,JointWorldModel
    from tests.test_joint_model import batch,NORM
    from training.train_joint import denoiser_retention
    model=JointWorldModel(JointConfig(method='autoregressive',width=16,layers=1,heads=2,mixtures=3),NORM)
    teacher=copy.deepcopy(model).eval();teacher.requires_grad_(False)
    value=batch();value['geometry'][:,2]=0;value['tilt'][:,0]=0
    same=denoiser_retention(model,teacher,value)
    assert same.item()==0
    with torch.no_grad():model.output.weight.add_(.01)
    changed=denoiser_retention(model,teacher,value)
    assert changed.item()>0


def test_canonical_membership_is_physical_and_resumable():
    data=JointTrainingData(ROOT/'datasets/uphill_push_v1/cache',seed=13,membership='canonical',
        canonical_condition={'obstacle_x':0.,'lateral_degrees':0.,'longitudinal_degrees':20.})
    assert len(data.allowed_rows)==5879 and data.scenario_ids.tolist()==[4]
    parents=np.unique(data.raw['parent_episode'][data.allowed_rows])
    assert len(parents)==16
    batch=data.sample(2048)
    assert torch.allclose(batch['geometry'][:,2],torch.zeros(2048))
    assert torch.allclose(batch['tilt'][:,0],torch.zeros(2048))
    assert torch.allclose(batch['tilt'][:,1],torch.full((2048,),np.deg2rad(20)),atol=1e-7,rtol=0)
    saved=data.state_dict();expected=data.sample(64);data.load_state_dict(saved);actual=data.sample(64)
    for key in expected:assert torch.equal(expected[key],actual[key]),key


def test_development_milestone_can_resume_to_larger_target(tmp_path):
    from training.train_joint import train
    from scripts.experiment1_full import recipe
    config=recipe("diffusion", 13)
    config['model'].update(width=16,layers=1,heads=2,sampling_steps=2)
    config['training'].update(batch_size=8,validation_windows=100000,checkpoint_windows=8)
    train(config,tmp_path/'fit',stage='development',target_windows=16,device='cpu')
    first=torch.load(tmp_path/'fit/latest.pt',map_location='cpu',weights_only=False)
    assert first['windows']==16 and first['target_windows']==16
    train(config,tmp_path/'fit',stage='development',target_windows=32,device='cpu',resume=True)
    second=torch.load(tmp_path/'fit/latest.pt',map_location='cpu',weights_only=False)
    assert second['windows']==32 and second['target_windows']==32 and second['updates']==4
