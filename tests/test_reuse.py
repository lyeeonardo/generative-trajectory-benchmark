import json
import numpy as np
import pytest
import torch
from evaluation.reuse import observation_agreement,discard_reason,run_group,TOLERANCE
from tests.fakes import FakeEnv,CASE,COLLECTION,factory

class Proposals:
    def __init__(self,error=None,bad_action=False):self.calls=0;self.contexts=[];self.error=error;self.bad_action=bad_action
    def __call__(self,model,ctx,rng):
        self.calls+=1;self.contexts.append(ctx)
        actions=[];observations=[]
        for j,g in enumerate(rng):
            a=torch.rand((6,3),generator=g).numpy()*.2
            o=ctx['history_observations'][j,-1].copy();pred=[]
            for k in range(6):
                o[:2]+=a[k,:2]*.05;o[2:4]=a[k,:2];pred.append(o.copy())
            pred=np.asarray(pred)
            if self.error is not None:pred[:,self.error]+=1.
            if self.bad_action:a[0,0]=np.nan
            actions.append(a);observations.append(pred)
        return {'actions':np.asarray(actions),'observations':np.asarray(observations),'event_probabilities':np.zeros((len(rng),6,5))}

def run(path,p=None,policy='agreement',terminal=8,**kw):
    return run_group(None,[CASE],COLLECTION,path,policy=policy,device='cpu',env_factory=factory(FakeEnv(terminal)),proposal_fn=p or Proposals(),**kw)

@pytest.mark.parametrize('policy,plans,ages',[('every_step',8,[8,0,0,0,0,0]),('agreement',2,[2,2,1,1,1,1])])
def test_exact_predictions_reuse_original_trajectory(tmp_path,policy,plans,ages):
    p=Proposals();r=run(tmp_path,p,policy);row=r['episodes'][0]
    assert row['success'] and row['steps']==8 and row['sampled_plans']==p.calls==plans
    assert row['executed_plan_age_counts']==ages and row['accepted_observations']==8
    assert r['internal_simulator_rollouts']==0 and not r['candidate_ranking']
    for ctx in p.contexts:
        assert set(ctx)=={'history_observations','history_actions','history_mask','geometry','tilt','time_fraction'}
        np.testing.assert_allclose(ctx['tilt'][0],CASE['tilt_radians'])
    if policy!='every_step':
        with np.load(tmp_path/'step_000_proposal.npz') as a,np.load(tmp_path/'step_005_proposal.npz') as b:
            for k in ['raw_actions','raw_observations','rng_after']:np.testing.assert_array_equal(a[k],b[k])
            assert b['plan_age'][0]==5 and b['plan_created_step'][0]==0
        with np.load(tmp_path/'step_006_proposal.npz') as b:assert b['replan_reason'][0]=='plan_exhausted'

@pytest.mark.parametrize('component',[0,2,4,6])
def test_each_physical_mismatch_replans_before_next_action(tmp_path,component):
    r=run(tmp_path,Proposals(error=component))
    assert r['episodes'][0]['sampled_plans']==8
    with np.load(tmp_path/'step_000_outcome.npz') as f:assert f['next_decision'][0]=='prediction_mismatch'
    with np.load(tmp_path/'step_001_proposal.npz') as f:assert f['plan_age'][0]==0 and f['replan_reason'][0]=='prediction_mismatch'

def test_wrapping_tolerance_and_nonfinite():
    a=np.zeros(7);b=a.copy();a[-1]=np.pi-.01;b[-1]=-np.pi+.01
    ok,error=observation_agreement(a,b);assert ok and error[-1]==pytest.approx(.02)
    b=a.copy();b[0]+=.02;assert observation_agreement(a,b)[0]
    b[0]+=.00001;assert not observation_agreement(a,b)[0]
    b[0]=np.nan;assert not observation_agreement(a,b)[0]
    with pytest.raises(ValueError):observation_agreement(a,b,[0,1,1,1])
    assert discard_reason('agreement',6,False)=='plan_exhausted'

def test_invalid_action_fails_without_rescue(tmp_path):
    r=run(tmp_path,Proposals(bad_action=True));row=r['episodes'][0]
    assert row['terminal']=='model_output_failure' and row['steps']==0 and row['sampled_plans']==1

@pytest.mark.parametrize('missing_outcome',[False,True])
def test_resume_reconstructs_cached_plan_rng_and_observation_check(tmp_path,missing_outcome):
    a=tmp_path/'a';b=tmp_path/'b';full=run(a)
    assert run(b,stop_after_steps=3)['status']=='INTERRUPTED_RESUMABLE'
    if missing_outcome:(b/'step_002_outcome.npz').unlink()
    p=Proposals();resumed=run(b,p)
    assert p.calls==1 and resumed['episodes']==full['episodes']
    for t in range(8):
        for kind in ['proposal','outcome']:
            with np.load(a/f'step_{t:03d}_{kind}.npz') as x,np.load(b/f'step_{t:03d}_{kind}.npz') as y:
                for key in x.files:
                    if key.endswith('seconds'):continue
                    np.testing.assert_array_equal(x[key],y[key])
    assert run(b)['new_steps_this_invocation']==0

@pytest.mark.parametrize('key',['raw_observations','plan_age','replan_reason'])
def test_replay_rejects_changed_committed_cache(tmp_path,key):
    run(tmp_path,stop_after_steps=3)
    path=tmp_path/'step_002_proposal.npz'
    with np.load(path) as f:record={k:f[k] for k in f.files}
    if key=='replan_reason':record[key]=np.asarray(['initial'])
    else:record[key]=record[key]+1
    np.savez(path,**record)
    with pytest.raises(ValueError):run(tmp_path)

def test_independent_auditor_rejects_wrong_agreement(tmp_path,monkeypatch):
    from evaluation import replay as audit
    monkeypatch.setattr(audit,'environment',lambda collection,scene:factory(FakeEnv(8))(collection,scene))
    run(tmp_path)
    args=(tmp_path,[CASE],COLLECTION,'agreement',TOLERANCE)
    assert audit.audit_group(args)['errors']==[]
    path=tmp_path/'step_002_outcome.npz'
    with np.load(path) as f:r={k:f[k] for k in f.files}
    r['agreement']=~r['agreement'];np.savez(path,**r)
    assert 'agreement' in audit.audit_group(args)['errors']

def test_batch_replans_only_disagreeing_live_cases(tmp_path):
    import copy
    cases=[copy.deepcopy(CASE),copy.deepcopy(CASE)];cases[1]['case_id']='another';cases[1]['scene']={'early':True}
    sizes=[];base=Proposals()
    def proposal(model,ctx,rng):
        sizes.append(len(rng));out=base(model,ctx,rng)
        # The second case has bad predicted ball position and terminates at step3.
        if len(rng)==2:out['observations'][1,:,0]+=1
        elif ctx['time_fraction'][0]<.03:out['observations'][0,:,0]+=1
        return out
    r=run_group(None,cases,COLLECTION,tmp_path,device='cpu',proposal_fn=proposal,
        env_factory=lambda collection,scene:factory(FakeEnv(3 if scene.get('early') else 8))(collection,scene))
    assert sizes==[2,1,1,1] and [x['sampled_plans'] for x in r['episodes']]==[2,3]
    assert [x['steps'] for x in r['episodes']]==[8,3]

def test_resume_rejects_changed_cached_rng(tmp_path):
    run(tmp_path,stop_after_steps=2)
    path=tmp_path/'step_001_proposal.npz'
    with np.load(path) as f:r={k:f[k] for k in f.files}
    r['rng_after'][0,0]^=1;np.savez(path,**r)
    with pytest.raises(ValueError,match='Cached RNG'):run(tmp_path)

def test_completed_group_rejects_new_inputs(tmp_path):
    run(tmp_path)
    with pytest.raises(ValueError,match='Group inputs'):run(tmp_path,evaluation_seed=2)
