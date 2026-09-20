"""Independent replay audit for executed action-observation trajectories."""
import json
import numpy as np
from evaluation.common import environment, public_context, PUBLIC, actual_cost
from environment.success import CRITERION
from aif.operational_cost import applied_action
def audit_group(args):
    directory,cases,collection,policy,tolerances=args[:5]
    supplied_tilts = args[5] if len(args) == 6 else [c['tilt_radians'] for c in cases]
    if np.asarray(supplied_tilts).shape != (len(cases), 2):raise ValueError('Invalid audit conditioning')
    envs=[];hist=[];past=[];events=[];geometry=[];terminal=[None]*len(cases)
    cache=[None]*len(cases);age=[0]*len(cases);created=[-1]*len(cases);reason=['initial']*len(cases);rng=[None]*len(cases)
    plans=np.zeros(len(cases),int);ages=np.zeros((len(cases),6),int);accepted=np.zeros(len(cases),int);failures=np.zeros((len(cases),4),int)
    primary_step=[None]*len(cases);entry_step=[None]*len(cases);primary_snapshots=[None]*len(cases)
    steps=0;maximum=0.;errors=[]
    for c in cases:
        e,o=environment(collection,c['scene']);envs.append(e);hist.append([o[:7].copy()]);past.append([]);events.append([]);geometry.append(o[7:12])
    for t,path in enumerate(sorted(directory.glob('step_*_proposal.npz'))):
        with np.load(path) as f:p={k:f[k] for k in f.files}
        with np.load(path.with_name(path.name.replace('proposal','outcome'))) as f:o={k:f[k] for k in f.files}
        ids=[i for i in range(len(cases)) if terminal[i] is None]
        if p['active_indices'].tolist()!=ids or o['active_indices'].tolist()!=ids:raise ValueError('Active cases differ')
        for j,i in enumerate(ids):
            context=public_context(hist[i],past[i],geometry[i],supplied_tilts[i],t)
            for key in PUBLIC:
                if not np.array_equal(context[key][0],p['context_'+key][j]):errors.append('public_context:'+key)
            fresh=cache[i] is None
            if bool(p['new_plan'][j])!=fresh:errors.append('new_plan')
            if fresh:
                cache[i]={k:p['raw_'+k][j].copy() for k in ['actions','observations','event_probabilities']};age[i]=0;created[i]=t;plans[i]+=1
                if p['replan_reason'][j]!=reason[i]:errors.append('replan_reason')
            else:
                for key,value in cache[i].items():
                    if not np.array_equal(value,p['raw_'+key][j],equal_nan=True):errors.append('changed_cached_'+key)
                if p['replan_reason'][j]!='cached':errors.append('cached_reason')
                if not np.array_equal(rng[i],p['rng_after'][j]):errors.append('cached_rng_changed')
            rng[i]=p['rng_after'][j].copy()
            if p['plan_age'][j]!=age[i] or p['plan_created_step'][j]!=created[i]:errors.append('plan_time_alignment')
            raw=cache[i]['actions'][age[i]]
            if not np.isfinite(raw).all():
                terminal[i]='model_output_failure'
                if o['executed'][j] or o['next_decision'][j]!='invalid_action':errors.append('invalid_action_executed')
                continue
            action=applied_action(raw)
            if not o['executed'][j] or not np.array_equal(action,o['applied_actions'][j]):errors.append('wrong_cached_action')
            result=envs[i].step(action);steps+=1;hist[i].append(result.observation[:7].copy());past[i].append(action)
            primary_now=bool(result.success);inside=bool(result.info.get('inside_goal',False))
            if inside and entry_step[i] is None:entry_step[i]=t+1
            new_primary=primary_now and primary_step[i] is None
            if new_primary:primary_step[i]=t+1
            ev=np.asarray([primary_now,result.collision,result.info['fall_out'],result.timeout,result.info['rod_ball_contact']],bool);events[i].append(ev)
            maximum=max(maximum,float(np.max(np.abs(result.observation[:7]-o['observations'][j]))))
            if not np.array_equal(ev,o['events'][j]):errors.append('events')
            if 'success' in o and bool(o['success'][j])!=(primary_step[i] is not None):errors.append('primary_success')
            if 'goal_entry' in o and bool(o['goal_entry'][j])!=(entry_step[i] is not None):errors.append('goal_entry')
            d=result.observation[:7].astype(float)-cache[i]['observations'][age[i]].astype(float)
            if not np.isfinite(d).all():err=np.full(4,np.inf)
            else:
                yaw=abs((d[6]+np.pi)%(2*np.pi)-np.pi)
                err=np.asarray([np.sqrt(sum(d[:2]**2)),np.sqrt(sum(d[2:4]**2)),np.sqrt(sum(d[4:6]**2)),yaw])
            agree=bool(np.all(err<=tolerances));accepted[i]+=agree;failures[i]+=(err>tolerances);ages[i,age[i]]+=1;age[i]+=1
            if new_primary:primary_snapshots[i]=(int(plans[i]),int(accepted[i]),len(past[i]),ages[i].copy(),failures[i].copy())
            if agree!=o['agreement'][j] or not np.allclose(err,o['agreement_errors'][j],atol=1e-12,rtol=1e-12):errors.append('agreement')
            if result.done:
                terminal[i]='success' if ev[0] else 'collision' if ev[1] else 'fall' if ev[2] else 'timeout';decision='terminal'
            elif policy=='every_step':decision='every_step'
            elif age[i]==6:decision='plan_exhausted'
            elif policy=='agreement' and not agree:decision='prediction_mismatch'
            else:decision='continue'
            if decision!=o['next_decision'][j]:errors.append('next_decision')
            if decision!='continue':cache[i]=None;reason[i]=decision
    rows=json.loads((directory/'report.json').read_text())['episodes']
    for i,row in enumerate(rows):
        cutoff=primary_step[i] if primary_step[i] is not None else len(past[i]);primary_events=[e.copy() for e in events[i][:cutoff]]
        if primary_step[i] is not None and primary_events:primary_events[-1][0]=True
        cost,_=actual_cost(hist[i][:cutoff+1],past[i][:cutoff],primary_events,geometry[i])
        snapshot=primary_snapshots[i] or (int(plans[i]),int(accepted[i]),len(past[i]),ages[i],failures[i])
        primary_terminal='success' if primary_step[i] is not None else ('timeout' if terminal[i]=='success' else terminal[i])
        expected={'case_id':cases[i]['case_id'],'success_criterion_version':CRITERION.version,'terminal':primary_terminal,
            'success':primary_step[i] is not None,'steps':cutoff,'success_step':primary_step[i],
            'goal_entry':entry_step[i] is not None,'first_goal_entry_step':entry_step[i],
            'sampled_plans':snapshot[0],'accepted_observations':snapshot[1],'checked_observations':snapshot[2],
            'executed_plan_age_counts':snapshot[3].tolist(),'disagreement_component_counts':snapshot[4].tolist()}
        for k,v in expected.items():
            if v!=row[k]:errors.append('episode_'+k)
        if 'supplied_tilt_degrees' in row:
            if not np.allclose(row['supplied_tilt_degrees'], np.rad2deg(supplied_tilts[i]), atol=1e-7, rtol=0):errors.append('supplied_tilt')
            correct = bool(np.allclose(supplied_tilts[i], cases[i]['tilt_radians'], atol=1e-7, rtol=0))
            if row.get('conditioning_correct') != correct:errors.append('conditioning_correct')
        if abs(cost-row['actual_operational_cost'])>1e-9:errors.append('cost')
    return {'steps':steps,'maximum_error':maximum,'errors':sorted(set(errors)),'rows':rows}
