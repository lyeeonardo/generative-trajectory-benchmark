"""Shared model behavior, public-information boundaries, and supervision checks."""
import copy
import numpy as np
import pytest
import torch
from data.joint_training import encode_batch
from generators.joint_world_model import JointConfig, JointWorldModel, METHODS, reduce_query_losses
from generators.registry import create_model, available_models

torch.set_num_threads(1)
NORM={k:{'mean':[0.]*d,'std':[1.]*d} for k,d in [('observation',7),('delta',7),('geometry',5),('tilt',2)]}
NORM['action_scale']=[.8,.8,4.]

def batch():
    n=4;torch.manual_seed(12)
    b={'history_observations':torch.randn(n,5,7)*.1,'history_actions':torch.zeros(n,4,3),
       'history_mask':torch.ones(n,5,dtype=torch.bool),'geometry':torch.randn(n,5),
       'tilt':torch.zeros(n,2),'time_fraction':torch.zeros(n),'future_observations':torch.randn(n,6,7)*.1,
       'actions':torch.randn(n,6,3)*.1,'valid':torch.ones(n,6,dtype=torch.bool),
       'events':torch.zeros(n,6,5),'event_known':torch.ones(n,6,5,dtype=torch.bool),
       'horizon':torch.full((n,),6),'quality':torch.tensor([2,0,1,2]),'role':torch.tensor([0,1,2,3]),'known_steps':torch.tensor([0,6,6,3])}
    return b

def model(method):return JointWorldModel(JointConfig(method,width=32,layers=2,heads=4,sampling_steps=2),NORM).eval()


def test_adaptive_conditioning_is_finite_and_context_sensitive():
    m=JointWorldModel(JointConfig('diffusion',width=32,layers=2,heads=4,sampling_steps=2,
                                  conditioning_mode='adaptive'),NORM).eval()
    b=batch();e=encode_batch(b,NORM);x=e['x'];t=torch.full((len(x),),.5)
    out,_=m(x,e,t)
    changed={**e,'context':e['context'].clone()};changed['context'][:,0]+=1
    shifted,_=m(x,changed,t)
    assert torch.isfinite(out).all()
    assert not torch.equal(out,shifted)

def test_targets_padding_returns_not_context():
    b=batch();e=encode_batch(b,NORM);c=copy.deepcopy(b)
    c['future_observations']+=10;c['valid'][:]=False;c['events'][:]=1;c['return']=torch.ones(4)*999
    f=encode_batch(c,NORM)
    assert torch.equal(e['context'],f['context'])
    assert torch.equal(e['known'],f['known'])
    assert torch.equal(e['semantic'],f['semantic'])
    assert not e['loss_mask'][1:3,::2].any()

def test_fixed_quality_and_goal_excluded_every_path():
    b=batch();b['role'][:]=2;b['known_steps'][:]=6;e=encode_batch(b,NORM)
    b['quality'][:]=1;b['geometry'][:,:2]+=100;f=encode_batch(b,NORM)
    assert torch.equal(e['context'],f['context']);assert torch.equal(e['quality'],f['quality'])

@pytest.mark.parametrize('method',METHODS)
def test_causal_prediction_clamping_stochastic_joint_and_backward(method):
    m=model(method);b=batch();loss,metrics=m.loss(b);assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    b['role'][:]=2;b['known_steps'][:]=6
    e=encode_batch(b,NORM);x=torch.randn_like(e['x']);t=torch.full((4,),.5)
    a,_=m(x,e,t);b2=copy.deepcopy(b);b2['actions'][:,3:]*=-1;e2=encode_batch(b2,NORM)
    x2=x.clone();x2[:,6:]+=100
    c,_=m(x2,e2,t);torch.testing.assert_close(a[:,:6],c[:,:6],rtol=0,atol=0)
    one=m.generate(b,torch.Generator().manual_seed(71));two=m.generate(b2,torch.Generator().manual_seed(71))
    torch.testing.assert_close(one['actions'],b['actions'],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(one['observations'][:,:3],two['observations'][:,:3],rtol=0,atol=0)
    b3=copy.deepcopy(b);b3['quality'][:]=2;b3['geometry'][:,:2]+=10;b3['future_observations']+=77
    three=m.generate(b3,torch.Generator().manual_seed(71));assert torch.equal(one['observations'],three['observations'])
    context={k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
    for h in (1,3,6):
        out=m.propose_joint(context,K=4,H=h);assert out['observations'].shape==(1,4,h,7)
        assert not torch.equal(out['actions'][:,0],out['actions'][:,1])
        assert torch.linalg.vector_norm(out['actions'][...,:2],dim=-1).max()<=.800001
    with pytest.raises(ValueError):m.predict({**context,'private_state':torch.zeros(1)},b['actions'][:1])
    with pytest.raises(ValueError):m.predict(context,torch.ones(1,6,3)*5)

def reduced(values,e):
    return reduce_query_losses(values,e,torch.zeros_like(e['events']))

def test_first_executed_action_has_tenfold_gradient_and_clamped_actions_have_none():
    e=encode_batch(batch(),NORM)
    values=torch.ones(4,12,requires_grad=True)
    loss,metrics=reduced(values,e);loss.backward()
    assert values.grad[0,0]/values.grad[0,2]==10
    assert not values.grad[1:3,::2].any()
    assert not values.grad[3,:6:2].any()
    assert values.grad[3,6]==values.grad[3,8]
    torch.testing.assert_close(metrics['action_loss'],torch.tensor(.55))
    torch.testing.assert_close(metrics['observation_loss'],torch.tensor(1.))
    torch.testing.assert_close(loss,torch.tensor(.55+1.+.1*np.log(2)),check_dtype=False)

def test_per_query_mixture_is_independent_of_count_and_horizon():
    b=batch();e=encode_batch(b,NORM)
    values=torch.arange(1.,5.)[:,None].expand(-1,12).clone()
    expected=.45*2*1+.25*2+.20*3+.10*2*4+.1*np.log(2)
    loss,_=reduced(values,e);assert loss.item()==pytest.approx(expected)
    idx=torch.tensor([0,0,0,1,2,3]);duplicate={k:v[idx] for k,v in b.items()}
    other,_=reduced(values[idx],encode_batch(duplicate,NORM))
    torch.testing.assert_close(loss,other)
    b['horizon'][0]=1
    short,_=reduced(values,encode_batch(b,NORM));torch.testing.assert_close(short,loss)

def test_padding_and_unknown_events_have_no_loss_or_gradient():
    b=batch();b['valid'][:,2:]=False;b['event_known'][:]=False
    e=encode_batch(b,NORM);x=torch.ones(4,12,requires_grad=True)
    loss,metrics=reduced(x,e);loss.backward()
    assert not x.grad[:,4:].any()
    assert metrics['event_loss']==0
    x2=x.detach().clone();x2[:,4:]=99999
    other,_=reduced(x2,e);torch.testing.assert_close(loss,other)


def test_registry_and_primary_candidate_budget():
    assert available_models()==('diffusion','flow_matching','autoregressive','cvae')
    for method in available_models():
        m=create_model({'method':method,'width':16,'layers':1,'heads':2,**({'posterior_layers':1,'latent_dim':4} if method=='cvae' else {'sampling_steps':1})},NORM)
        b=batch()
        context={k:b[k][:1] for k in ('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')}
        for bad in (0,33,-1):
            with pytest.raises(ValueError):m.propose_joint(context,K=bad)
    with pytest.raises(ValueError,match='Unknown joint model'):
        create_model({'method':'not_registered'},NORM)
