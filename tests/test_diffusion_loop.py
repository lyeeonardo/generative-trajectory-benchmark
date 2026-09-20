"""Audit the x0 diffusion loop independently of deleted historical datasets."""
import copy
import math
import types
import pytest
import torch
from data.joint_training import encode_batch
from generators.joint_world_model import JointConfig, JointWorldModel
from tests.test_joint_model import batch, NORM

@pytest.mark.parametrize('steps',[1,8,32])
def test_ddim_x0_oracle_recovers_target_and_clamps_every_iteration(steps):
    b=batch();b['horizon']=torch.tensor([1,3,6,6]);b['known_steps']=torch.tensor([0,3,6,3])
    e=encode_batch(b,NORM)
    m=JointWorldModel(JointConfig(width=16,layers=1,heads=2,sampling_steps=steps),NORM)
    seen=[]
    def oracle(self,x,encoded,t):
        assert torch.equal(x[encoded['known']],encoded['x'][encoded['known']])
        assert not x[~encoded['semantic']].any()
        seen.append(t[0].item())
        return encoded['x'].clone(),torch.zeros(len(x),6,5)
    m.forward=types.MethodType(oracle,m)
    out=m.generate(b,torch.Generator().manual_seed(5))
    assert len(seen)==2*(steps+1) # Joint pass plus bounded-action physical re-prediction.
    assert all(a>b for a,b in zip(seen[:steps],seen[1:steps]))
    for i,h in enumerate(b['horizon']):
        torch.testing.assert_close(out['actions'][i,:h],b['actions'][i,:h],atol=1e-6,rtol=0)
        torch.testing.assert_close(out['observations'][i,:h],b['future_observations'][i,:h],atol=1e-6,rtol=0)


def test_training_corruption_matches_cosine_schedule_and_clean_clamped_actions():
    b=batch();e=encode_batch(b,NORM);m=JointWorldModel(JointConfig(width=16,layers=1,heads=2),NORM)
    torch.manual_seed(303)
    t=torch.rand(len(b['role']))*.998+.001;noise=torch.randn_like(e['x'])
    expected=torch.cos(t[:,None,None]*math.pi/2)*e['x']+torch.sin(t[:,None,None]*math.pi/2)*noise
    expected=torch.where(e['known'],e['x'],expected)*e['semantic']
    captures=[]
    hook=m.register_forward_pre_hook(lambda model,args:captures.append(args))
    torch.manual_seed(303);loss,_=m.loss(b);hook.remove()
    x,encoded,actual_t=captures[0]
    torch.testing.assert_close(x,expected,atol=0,rtol=0)
    assert torch.equal(t,actual_t)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.output.weight.grad.abs().sum()>0


def test_independent_episode_sampler_matches_native_diffusion():
    from evaluation.common import propose_independent
    torch.manual_seed(4)
    m=JointWorldModel(JointConfig(width=32,layers=2,heads=4,sampling_steps=8),NORM).eval()
    b=batch();context={k:b[k] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
    combined=propose_independent(m,context,[torch.Generator().manual_seed(20+i) for i in range(4)])
    for i in range(4):
        native=m.propose_joint({k:v[i:i+1] for k,v in context.items()},K=1,H=6,generator=torch.Generator().manual_seed(20+i))
        for key in native:torch.testing.assert_close(combined[key][i],native[key][0,0],atol=2e-6,rtol=1e-5)


def test_diffusion_optimizer_and_noise_resume_exactly():
    torch.manual_seed(919);b=batch()
    m=JointWorldModel(JointConfig(width=16,layers=1,heads=2),NORM)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    def update():
        opt.zero_grad();loss,_=m.loss(b);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    update()
    state=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
    update();expected=copy.deepcopy(m.state_dict())
    m.load_state_dict(state);opt.load_state_dict(optimizer);torch.set_rng_state(rng)
    update()
    for key,value in expected.items():assert torch.equal(value,m.state_dict()[key]),key


@pytest.mark.parametrize('stage',['pilot','full'])
def test_unqualified_recipe_cannot_start_expensive_training(stage,tmp_path):
    from training.train_joint import train
    with pytest.raises(RuntimeError,match='not ready'):
        train({'execution_ready':False},tmp_path/'must_not_exist',stage=stage,device='cpu')
    assert not (tmp_path/'must_not_exist').exists()


def test_diffusion_temperature_guidance_and_work_accounting():
    torch.manual_seed(7)
    m=JointWorldModel(JointConfig(width=32,layers=2,heads=4,sampling_steps=8),NORM).eval()
    b=batch();context={k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
    m.sampling_temperature=1.;ordinary=m.propose_joint(context,generator=torch.Generator().manual_seed(9))['actions']
    assert m.inference_work('proposal')['backbone_evaluations']==18
    m.sampling_temperature=.25;cool=m.propose_joint(context,generator=torch.Generator().manual_seed(9))['actions']
    assert not torch.equal(ordinary,cool)
    m.guidance_scale=1.1;guided=m.propose_joint(context,generator=torch.Generator().manual_seed(9))['actions']
    assert not torch.equal(cool,guided)
    work=m.inference_work('proposal');assert work['backbone_evaluations']==26
    assert work['sampling_temperature']==.25 and work['guidance_scale']==1.1


@pytest.mark.parametrize('method',['flow_matching','autoregressive'])
def test_sampling_temperature_controls_other_stochastic_families(method):
    torch.manual_seed(17)
    m=JointWorldModel(JointConfig(method=method,width=32,layers=2,heads=4,sampling_steps=4),NORM).eval()
    b=batch();context={k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
    m.sampling_temperature=1.
    ordinary=m.propose_joint(context,generator=torch.Generator().manual_seed(19))['actions']
    m.sampling_temperature=.25
    cool=m.propose_joint(context,generator=torch.Generator().manual_seed(19))['actions']
    assert not torch.equal(ordinary,cool)
    assert m.inference_work('proposal')['sampling_temperature']==.25
