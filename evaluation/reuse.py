"""K1 joint-trajectory execution with observation-triggered plan reuse.

The checker reads only the actual observed physical state and the corresponding
observation in the previously committed trajectory. It makes no model or simulator
prediction call. Each new proposal contains exactly one H6 action/observation pair.
"""
from pathlib import Path
import json,time
import numpy as np
import torch
from data.preparation import write_json
from evaluation.common import (PUBLIC,environment,public_context,episode_seed,
    actual_cost,atomic_npz)
from environment.success import CRITERION
from aif.operational_cost import applied_action
from evaluation.common import propose_independent

POLICIES=('every_step','agreement')
TOLERANCE=np.asarray([.02,.20,.01,.10],np.float64)
ERROR_NAMES=('ball_position_m','ball_velocity_m_s','rod_position_m','yaw_rad')


def observation_agreement(actual,predicted,tolerance=TOLERANCE):
    actual=np.asarray(actual,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.float64);tol=np.asarray(tolerance)
    if actual.shape!=(7,) or predicted.shape!=(7,) or tol.shape!=(4,) or not np.isfinite(tol).all() or (tol<=0).any():
        raise ValueError('Expected physical7D observations and four positive finite tolerances')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():return False,np.full(4,np.inf)
    d=actual-predicted;d[6]=np.arctan2(np.sin(d[6]),np.cos(d[6]))
    errors=np.asarray([np.linalg.norm(d[:2]),np.linalg.norm(d[2:4]),np.linalg.norm(d[4:6]),abs(d[6])])
    return bool(np.all(errors<=tol)),errors


def discard_reason(policy,age_after_action,agreement):
    if policy not in POLICIES:raise ValueError('Unknown execution policy')
    if policy=='every_step':return 'every_step'
    if age_after_action>=6:return 'plan_exhausted'
    if policy=='agreement' and not agreement:return 'prediction_mismatch'
    return None


def run_group(model,cases,collection,dest,*,policy='agreement',tolerance=TOLERANCE,
    device='cuda',evaluation_seed=101,stop_after_steps=None,env_factory=environment,proposal_fn=None,
    supplied_tilts=None):
    if policy not in POLICIES:raise ValueError('Unknown execution policy')
    tolerance=np.asarray(tolerance,dtype=np.float64)
    observation_agreement(np.zeros(7),np.zeros(7),tolerance)
    if collection['simulator']['max_steps']!=100:raise ValueError('Expected actual100-step task')
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate cases')
    for c in cases:
        if not np.allclose(c['tilt_radians'],np.deg2rad(c['tilt_degrees']),atol=1e-7,rtol=0):raise ValueError('Oracle tilt mismatch')
    # Supplied labels change only model input; simulator scenes and episode RNG remain fixed.
    supplied = np.asarray([c['tilt_radians'] for c in cases] if supplied_tilts is None else supplied_tilts, dtype=np.float64)
    if supplied.shape != (len(cases), 2) or not np.isfinite(supplied).all():
        raise ValueError('Expected one finite supplied tilt per case')
    conditioning_correct = np.all(np.isclose(supplied, [c['tilt_radians'] for c in cases], atol=1e-7, rtol=0), axis=1)
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    identity={'cases':cases,'collection':collection,'policy':policy,'tolerance':tolerance.tolist(),'evaluation_seed':evaluation_seed,'device':device,'success_criterion_version':CRITERION.version}
    if supplied_tilts is not None:identity['supplied_tilts_radians'] = supplied.tolist()
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=identity:raise ValueError('Group inputs changed')
    write_json(dest/'identity.json',identity)
    if (dest/'report.json').exists():
        r=json.loads((dest/'report.json').read_text())
        if r['policy']!=policy or r['tolerance']!=np.asarray(tolerance).tolist():raise ValueError('Group identity changed')
        return {**r,'new_steps_this_invocation':0,'replay_steps_this_invocation':0,'reused':True}
    if proposal_fn is None:proposal_fn=propose_independent
    n=len(cases);envs=[];history=[];past=[];events=[];geometry=[];rng=[]
    caches=[None]*n;age=[0]*n;created=[-1]*n;reason=['initial']*n;term=[None]*n
    plans=[0]*n;accepted=[0]*n;tested=[0]*n;primary_step=[None]*n;entry_step=[None]*n
    primary_snapshots=[None]*n;age_counts=np.zeros((n,6),int);failure_components=np.zeros((n,4),int)
    replayed=0;newsteps=0;generation=0.;env_seconds=0.;checker_seconds=0.;forward_batches=0
    for c in cases:
        env,obs=env_factory(collection,c['scene']);envs.append(env);history.append([obs[:7].copy()]);past.append([]);events.append([]);geometry.append(obs[7:12].copy())
        rng.append(torch.Generator(device=device).manual_seed(episode_seed(c['case_id'],evaluation_seed)))
    for t in range(100):
        ids=[i for i in range(n) if term[i] is None]
        if not ids:break
        need=[i for i in ids if caches[i] is None]
        contexts=[public_context(history[i],past[i],geometry[i],supplied[i],t) for i in ids]
        ctx={k:np.concatenate([x[k] for x in contexts]) for k in PUBLIC}
        commit=dest/f'step_{t:03d}_proposal.npz';outcome=dest/f'step_{t:03d}_outcome.npz'
        expected_new=np.asarray([i in need for i in ids],bool)
        if commit.exists():
            with np.load(commit) as f:data={k:f[k] for k in f.files}
            if data['active_indices'].tolist()!=ids or not np.array_equal(expected_new,data['new_plan']):raise ValueError('Replay plan decisions changed')
            for k in PUBLIC:
                if not np.array_equal(data['context_'+k],ctx[k]):raise ValueError('Replay public context changed: '+k)
            for j,i in enumerate(ids):
                if i not in need and not np.array_equal(rng[i].get_state().cpu().numpy(),data['rng_after'][j]):raise ValueError('Cached RNG changed')
                rng[i].set_state(torch.as_tensor(data['rng_after'][j],dtype=torch.uint8,device='cpu'))
                if i in need:
                    caches[i]={k:data['raw_'+k][j].copy() for k in ['actions','observations','event_probabilities']};age[i]=0;created[i]=t
                else:
                    for k,v in caches[i].items():
                        if not np.array_equal(v,data['raw_'+k][j],equal_nan=True):raise ValueError('Cached trajectory was altered')
                if age[i]!=data['plan_age'][j] or created[i]!=data['plan_created_step'][j]:raise ValueError('Trajectory time index changed')
                expected_reason=reason[i] if i in need else 'cached'
                if data['replan_reason'][j]!=expected_reason:raise ValueError('Replanning reason changed')
        else:
            elapsed=0.
            if need:
                select=[ids.index(i) for i in need]
                if device.startswith('cuda'):torch.cuda.synchronize()
                start=time.perf_counter();out=proposal_fn(model,{k:v[select] for k,v in ctx.items()},[rng[i] for i in need])
                if device.startswith('cuda'):torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                out={k:v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k,v in out.items()}
                if out['actions'].shape!=(len(need),6,3) or out['observations'].shape!=(len(need),6,7):raise ValueError('Expected one H6 action/observation pair')
                for j,i in enumerate(need):caches[i]={k:out[k][j].copy() for k in ['actions','observations','event_probabilities']};age[i]=0;created[i]=t
            data={'active_indices':np.asarray(ids),'new_plan':expected_new,'plan_age':np.asarray([age[i] for i in ids]),
                'plan_created_step':np.asarray([created[i] for i in ids]),'replan_reason':np.asarray([reason[i] if i in need else 'cached' for i in ids]),
                'generation_seconds':np.asarray(elapsed),'rng_after':np.stack([rng[i].get_state().cpu().numpy() for i in ids]),
                **{'raw_'+k:np.stack([caches[i][k] for i in ids]) for k in ['actions','observations','event_probabilities']},
                **{'context_'+k:v for k,v in ctx.items()}}
            atomic_npz(commit,**data)
        generation+=float(data['generation_seconds']);forward_batches+=bool(need)
        for i in need:plans[i]+=1
        observations=[];event_rows=[];did=[];applied=[];agreements=[];errors=[];discarded=[]
        elapsed_env=0.;elapsed_check=0.
        new_primary=[]
        for j,i in enumerate(ids):
            request=caches[i]['actions'][age[i]]
            if not np.isfinite(request).all():
                term[i]='model_output_failure';observations.append(history[i][-1]);event_rows.append(np.zeros(5,bool));did.append(False);applied.append(np.zeros(3));agreements.append(False);errors.append(np.full(4,np.inf));discarded.append('invalid_action');continue
            action=applied_action(request);start=time.perf_counter();r=envs[i].step(action);elapsed_env+=time.perf_counter()-start
            primary_now=bool(r.success);inside=bool(r.info.get('inside_goal',False))
            if inside and entry_step[i] is None:entry_step[i]=t+1
            if primary_now and primary_step[i] is None:primary_step[i]=t+1;new_primary.append(i)
            ev=np.array([primary_now,r.collision,r.info['fall_out'],r.timeout,r.info['rod_ball_contact']],bool)
            observations.append(r.observation[:7]);event_rows.append(ev);did.append(True);applied.append(action)
            history[i].append(r.observation[:7].copy());past[i].append(action.copy());events[i].append(ev)
            start=time.perf_counter();agree,error=observation_agreement(r.observation[:7],caches[i]['observations'][age[i]],tolerance);elapsed_check+=time.perf_counter()-start
            agreements.append(agree);errors.append(error);tested[i]+=1;accepted[i]+=agree
            failure_components[i]+=(error>tolerance);age_counts[i,age[i]]+=1;age[i]+=1
            if i in new_primary:primary_snapshots[i]=(plans[i],accepted[i],tested[i],age_counts[i].copy(),failure_components[i].copy())
            if r.done:
                term[i]='success' if ev[0] else 'collision' if ev[1] else 'fall' if ev[2] else 'timeout';discard='terminal'
            else:discard=discard_reason(policy,age[i],agree)
            discarded.append(discard or 'continue')
            if discard is not None:caches[i]=None;reason[i]=discard
        current={'active_indices':np.asarray(ids),'observations':np.asarray(observations),'events':np.asarray(event_rows),
            'success':np.asarray([primary_step[i] is not None for i in ids]),'goal_entry':np.asarray([entry_step[i] is not None for i in ids]),
            'executed':np.asarray(did),'applied_actions':np.asarray(applied),'agreement':np.asarray(agreements),
            'agreement_errors':np.asarray(errors),'next_decision':np.asarray(discarded)}
        if outcome.exists():
            with np.load(outcome) as f:
                for k,v in current.items():
                    if not np.array_equal(v,f[k],equal_nan=v.dtype.kind=='f'):raise ValueError('Physical/decision replay mismatch: '+k)
                elapsed_env=float(f['environment_seconds']);elapsed_check=float(f['checker_seconds'])
            replayed+=sum(did)
        else:
            atomic_npz(outcome,**current,environment_seconds=np.asarray(elapsed_env),checker_seconds=np.asarray(elapsed_check));newsteps+=sum(did)
        env_seconds+=elapsed_env;checker_seconds+=elapsed_check
        if stop_after_steps is not None and t+1>=stop_after_steps:
            return {'status':'INTERRUPTED_RESUMABLE','new_steps':newsteps,'replay_steps':replayed}
    if any(x is None for x in term):raise RuntimeError('Actual100-step task did not terminate')
    rows=[]
    for i,c in enumerate(cases):
        cost,_=actual_cost(history[i],past[i],events[i],geometry[i])
        rows.append({'case_id':c['case_id'],'base_layout_id':c['base_layout_id'],'true_tilt':c['tilt_degrees'],
            'evaluation_rng_seed':evaluation_seed,'success_criterion_version':CRITERION.version,'terminal':term[i],
            'success':term[i]=='success','steps':len(past[i]),'success_step':primary_step[i],
            'goal_entry':entry_step[i] is not None,'first_goal_entry_step':entry_step[i],'actual_operational_cost':cost,
            'sampled_plans':plans[i],'accepted_observations':accepted[i],'checked_observations':tested[i],
            'executed_plan_age_counts':age_counts[i].tolist(),'disagreement_component_counts':failure_components[i].tolist()})
    if supplied_tilts is not None:
        for i,row in enumerate(rows):
            row['supplied_tilt_degrees'] = np.rad2deg(supplied[i]).tolist()
            row['conditioning_correct'] = bool(conditioning_correct[i])
    result={'status':'COMPLETE','episodes':rows,'policy':policy,'tolerance':np.asarray(tolerance).tolist(),
        'new_steps_this_invocation':newsteps,'replay_steps_this_invocation':replayed,'generation_seconds':generation,'environment_seconds':env_seconds,'checker_seconds':checker_seconds,
        'forward_batches':forward_batches,
        'success_criterion_version':CRITERION.version,'successes':sum(r['success'] for r in rows),'goal_entries':sum(r['goal_entry'] for r in rows),'K':1,'H':6,'quality':'HIGH','oracle_true_tilt':bool(conditioning_correct.all()),'internal_simulator_rollouts':0,'candidate_ranking':False,'fallback_steps':0}
    write_json(dest/'report.json',result);return result
