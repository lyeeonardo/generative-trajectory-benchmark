"""Causal joint-CVAE inference, variational training, and checkpoint recovery."""
import copy
import torch
import pytest
from data.joint_training import encode_batch
from generators.cvae import CVAEConfig, JointCVAE
from tests.test_joint_model import batch, NORM
from evaluation.common import load_model


def model():
    return JointCVAE(CVAEConfig(width=32,layers=2,heads=4,posterior_layers=1,latent_dim=4,kl_warmup_windows=100),NORM)


def context(b):
    return {k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}


def test_variational_gradients_and_exposure_based_KL_warmup():
    m=model();b=batch();m.set_training_progress(50)
    loss,metrics=m.loss(b);assert torch.isfinite(loss);loss.backward()
    assert metrics['kl_coefficient']==pytest.approx(.005)
    assert m.posterior.weight.grad.abs().sum()>0
    assert m.output.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.set_training_progress(1000);assert m.loss(b)[1]['kl_coefficient']==pytest.approx(.01)


def test_inference_never_uses_recognition_or_future_targets(monkeypatch):
    m=model().eval();b=batch();c=copy.deepcopy(b)
    c['future_observations']+=900;c['events'][:]=1;c['valid'][:]=False;c['event_known'][:]=False
    monkeypatch.setattr(m,'encode',lambda *args:(_ for _ in ()).throw(AssertionError('posterior during inference')))
    a=m.generate(b,torch.Generator().manual_seed(4));d=m.generate(c,torch.Generator().manual_seed(4))
    for key in a:assert torch.equal(a[key],d[key])


def test_actions_are_clamped_and_future_actions_do_not_change_earlier_outputs():
    m=model().eval();b=batch();c=context(b);actions=b['actions'][:1].clone();later=actions.clone();later[:,1:]*=-1
    one=m.predict(c,actions,samples=3,generator=torch.Generator().manual_seed(7))
    two=m.predict(c,later,samples=3,generator=torch.Generator().manual_seed(7))
    assert torch.equal(one['actions'],actions[:,None].expand_as(one['actions']))
    assert torch.equal(one['observations'][:,:,0],two['observations'][:,:,0])
    assert torch.equal(one['event_probabilities'][:,:,0],two['event_probabilities'][:,:,0])
    changed={k:v.clone() for k,v in c.items()};changed['geometry'][:,:2]+=50
    three=m.predict(changed,actions,samples=3,generator=torch.Generator().manual_seed(7))
    assert torch.equal(one['observations'],three['observations'])
    with pytest.raises(ValueError,match='whitelist'):m.predict({**c,'hidden_state':torch.zeros(1)},actions)


def test_causal_decoder_and_masked_latents():
    m=model();b=batch();e=encode_batch(b,NORM);z=torch.randn(4,12,4);z2=z.clone();z2[:,4:]+=100
    one,events=m(z,e);two,events2=m(z2,e)
    assert torch.equal(one[:,:4],two[:,:4]);assert torch.equal(events[:,:2],events2[:,:2])
    z3=z.clone();z3[1:3,::2]+=100
    three,_=m(z3,e);assert torch.equal(one[1:3],three[1:3])


def test_joint_shapes_diversity_bounds_and_seeded_sampling():
    m=model().eval();c=context(batch())
    for H in (1,3,6):
        a=m.propose_joint(c,K=4,H=H,generator=torch.Generator().manual_seed(3))
        b=m.propose_joint(c,K=4,H=H,generator=torch.Generator().manual_seed(3))
        assert a['actions'].shape==(1,4,H,3);assert a['observations'].shape==(1,4,H,7)
        for key in a:assert torch.equal(a[key],b[key])
        assert not torch.equal(a['actions'][:,0],a['actions'][:,1])
        assert not torch.equal(a['observations'][:,0],a['observations'][:,1])
        assert torch.linalg.vector_norm(a['actions'][...,:2],dim=-1).max()<=.800001
        assert a['actions'][...,2].abs().max()<=4.000001


def test_checkpoint_retains_latent_configuration(tmp_path):
    from dataclasses import asdict
    m=model().eval();path=tmp_path/'model.pt'
    torch.save({'model':m.state_dict(),'model_config':asdict(m.config),'normalization':NORM,'contract':{'config':{'seed':13}}},path)
    other,saved=load_model(path,'cpu');assert other.config.latent_dim==4
    c=context(batch())
    a=m.propose_joint(c,generator=torch.Generator().manual_seed(4));b=other.propose_joint(c,generator=torch.Generator().manual_seed(4))
    for key in a:assert torch.equal(a[key],b[key])


def test_cvae_training_resume_restores_latent_RNG_and_KL_schedule(tmp_path):
    from training.train_joint import train
    from scripts.experiment1_full import recipe
    config=recipe("cvae", 13)
    config['model'].update(width=32,layers=1,heads=4,posterior_layers=1,latent_dim=4,kl_warmup_windows=16)
    config['training'].update(batch_size=8,checkpoint_windows=8)
    train(config,tmp_path/'full',stage='smoke',smoke_steps=4,device='cpu')
    train(config,tmp_path/'resume',stage='smoke',smoke_steps=4,device='cpu',stop_after_updates=2)
    train(config,tmp_path/'resume',stage='smoke',smoke_steps=4,device='cpu',resume=True)
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False);b=torch.load(tmp_path/'resume/latest.pt',weights_only=False)
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    assert torch.equal(a['torch_rng'],b['torch_rng']);assert a['sampler']['rng']==b['sampler']['rng']


def test_cvae_inference_work_counts_physical_reprediction():
    m=model()
    assert m.inference_work('prediction',1)['backbone_evaluations']==1
    proposal=m.inference_work('proposal',6)
    assert proposal['backbone_evaluations']==2
    assert proposal['imposed_action_reprediction_queries']==1


def test_cvae_sampling_temperature_controls_prior_and_is_reported():
    m=model().eval();c=context(batch())
    m.sampling_temperature=1.
    ordinary=m.propose_joint(c,generator=torch.Generator().manual_seed(29))['actions']
    m.sampling_temperature=.25
    cool=m.propose_joint(c,generator=torch.Generator().manual_seed(29))['actions']
    assert not torch.equal(ordinary,cool)
    assert m.inference_work('proposal')['sampling_temperature']==.25
    m.sampling_temperature=0.
    with pytest.raises(ValueError,match='temperature'):m.propose_joint(c)
