"""Frozen development diagnostics; physics used only after proposals are saved."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json,time
import numpy as np
import torch
from aif.observation import calibrate,law_from_samples,moments
from aif.operational_cost import horizon_cost
from aif.scoring import estimate_cost
from data.preparation import write_json

PUBLIC=('history_observations','history_actions','history_mask','geometry','tilt','time_fraction')

def fixed_bank(cache,split):
    cache=Path(cache);m=json.loads((cache/'manifest.json').read_text())
    fields=set(PUBLIC)|{'actions','future_observations','valid','source','source_row','source_ref'}
    a={k:np.load(cache/split/(k+'.npy'),mmap_mode='r') for k in fields}
    indices=np.flatnonzero(np.load(cache/split/'diagnostic.npy'))
    return {k:np.array(v[indices]) for k,v in a.items()},indices

def predict_bank(model,bank,samples,seed,horizon=6):
    device=next(model.parameters()).device;out=[]
    rng=torch.Generator(device=device).manual_seed(seed)
    for start in range(0,len(bank['actions']),32):
        sl=slice(start,start+32);context={k:bank[k][sl] for k in PUBLIC}
        out.append(model.predict(context,bank['actions'][sl,:horizon],samples=samples,generator=rng)['observations'].cpu())
    return torch.cat(out)

def kinematic(bank):
    now=bank['history_observations'][:,-1];p=np.repeat(now[:,None],6,axis=1)
    t=np.arange(1,7)[None,:,None]*.05;p[:,:,:2]+=now[:,None,2:4]*t
    p[:,:,4:6]+=np.cumsum(bank['actions'][:,:,:2],1)*.05
    yaw=now[:,None,6]+np.cumsum(bank['actions'][:,:,2],1)*.05;p[:,:,6]=np.arctan2(np.sin(yaw),np.cos(yaw))
    return p

def error(pred,truth):
    e=pred-truth;e[...,6]=torch.atan2(e[...,6].sin(),e[...,6].cos());return e


def evaluate(model,cache,diagnostic,output):
    started=time.monotonic();cache=Path(cache);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    d=json.loads(Path(diagnostic).read_text());dev,indices=fixed_bank(cache,'development_selection');cal,ci=fixed_bank(cache,'validation_calibration')
    if indices.tolist()!=d['development_rows'] or ci.tolist()!=d['calibration_rows']:raise ValueError('Diagnostic bank changed')
    pred=predict_bank(model,dev,d['prediction_samples'],d['seed'])
    pred1=predict_bank(model,dev,d['prediction_samples'],d['seed'],horizon=1)
    calpred=predict_bank(model,cal,d['prediction_samples'],d['seed']+1,horizon=1)
    truth=torch.tensor(dev['future_observations']);valid=torch.tensor(dev['valid'])
    extra=calibrate(calpred[:,:,0],torch.tensor(cal['future_observations'][:,0]))
    law=law_from_samples(pred1[:,:,0],extra);lp=law.log_prob(truth[:,0]);e1=error(law.mean,truth[:,0])
    mean=torch.stack([moments(pred[:,:,h])[0] for h in range(6)],1);err=error(mean,truth)
    kin=torch.tensor(kinematic(dev));ke=error(kin,truth)
    rmse=torch.sqrt(err[valid].square().mean(0));krmse=torch.sqrt(ke[valid].square().mean(0))
    h1=torch.sqrt(e1.square().mean(0));h6=torch.sqrt(err[valid[:,5],5].square().mean(0))
    coverage=(e1.abs()<=1.96*law.variance.sqrt()).float().mean(0)
    # Actual joint pools saved in full before any privileged reference is opened.
    picks=d['proposal_anchors'];rows=[p['bank_row'] for p in picks]
    context={k:dev[k][rows] for k in PUBLIC}
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(d['seed']+2)
    pools=model.propose_joint(context,K=d['proposal_K'],quality=2,H=6,generator=generator)
    physical=model.predict({k:np.repeat(v,d['proposal_K'],axis=0) for k,v in context.items()},pools['actions'].reshape(-1,6,3),samples=d['prediction_samples'],generator=generator)
    repeated={k:np.repeat(v,d['proposal_K'],axis=0) for k,v in context.items()}
    estimated=estimate_cost(physical,repeated,pools['actions'].reshape(-1,6,3),seed=d['seed']+3)
    predicted_cost=estimated['expected_cost'].reshape(len(picks),d['proposal_K']);selected=predicted_cost.argmin(1)
    np.savez_compressed(output/'committed_predictions.npz',predicted_cost=predicted_cost,selected_index=selected,fixed_predictions=pred.numpy(),one_step_predictions=pred1.numpy(),calibration_predictions=calpred.numpy(),
                        **{'joint_'+k:v.cpu().numpy() for k,v in pools.items()},
                        controlled_proposal_predictions=physical['observations'].cpu().numpy())
    write_json(output/'commit.json',{'status':'PREDICTIONS_COMMITTED_BEFORE_OFFLINE_TRUTH','seed':d['seed'],'proposal_anchors':picks})
    truth_record=offline_truth(cache.parent,picks,{k:v.cpu().numpy() for k,v in pools.items()},context,output)
    with np.load(output/'offline_truth.npz') as raw:actual_cost=raw['cost'].reshape(len(picks),d['proposal_K'])
    regret=actual_cost[np.arange(len(picks)),selected]-actual_cost.min(1)
    rank_corr=[]
    for predicted,actual in zip(predicted_cost,actual_cost):
        if np.ptp(predicted)>1e-8 and np.ptp(actual)>1e-8:
            def average_rank(values):
                _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
                return (np.cumsum(counts)-.5*(counts+1))[inverse]
            rank_corr.append(float(np.corrcoef(average_rank(predicted),average_rank(actual))[0,1]))
    q=d['qualification'];gates={}
    for h,values in [(1,h1),(6,h6)]:
        for label,sl in [('ball_position',slice(0,2)),('ball_velocity',slice(2,4)),('yaw',slice(6,7))]:
            gates[f'H{h}_{label}']=float(values[sl].square().mean().sqrt())<=q[f'H{h}_{label}_rmse_max']
    gates['coverage']=bool((coverage>=q['minimum_each_channel_95_coverage']).all())
    gates['variance_not_unbounded']=bool((law.variance.sqrt().mean(0)<=q['maximum_mean_std_over_kinematic_rmse']*torch.maximum(krmse,torch.tensor(q['physical_error_scale_floor']))).all())
    gates['position_beats_kinematic']=bool((rmse[:2]<=q['maximum_position_rmse_over_kinematic']*krmse[:2].clamp_min(.001)).all())
    report={'one_step_nll':float(-lp.mean()),'rmse':rmse.tolist(),'kinematic_rmse':krmse.tolist(),
            'fixed_prediction_rmse':float((rmse/torch.maximum(krmse,torch.tensor(q['physical_error_scale_floor']))).mean()),
            'mean_selected_cost_regret':float(regret.mean()),'mean_cost_rank_correlation':float(np.mean(rank_corr)) if rank_corr else None,
            'ranking_selection_committed_before_truth':True,'H1_rmse':h1.tolist(),'H6_rmse':h6.tolist(),'coverage_95':coverage.tolist(),
            'calibrated_extra_variance':extra.tolist(),'prediction_gates':gates,'prediction_qualified':all(gates.values()),
            **truth_record,'seconds':time.monotonic()-started,'scope':'Development diagnostic, no primary validation/control episodes'}
    write_json(output/'metrics.json',report);return report

def offline_truth(root,picks,pools,context,output):
    # Imports deliberately local: generation must finish before offline simulator use.
    from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv,MujocoRigidState
    from mujoco_task.config import get_preset
    from mujoco_task.sim.scene import SceneSpec
    config=json.loads((root/'collection_config.json').read_text());K=pools['actions'].shape[1]
    obs=np.zeros((len(picks),K,7,7),np.float32);events=np.zeros((len(picks),K,6,5),bool);valid=np.zeros((len(picks),K,6),bool);costs=[];useful=[];steps=0
    for i,pick in enumerate(picks):
        private=json.loads((root/'private_snapshots'/(pick['case_id']+'.json')).read_text())
        env=MujocoRigidTiltPushEnv(replace(get_preset('run').sim,**config['simulator']),config['physics']);env.reset(SceneSpec(**private['scene']))
        sd=private['states'][pick['anchor']]
        for key in ('qpos','qvel','mocap_pos','mocap_quat','integration_state'):
            if sd[key] is not None:sd[key]=np.asarray(sd[key],np.float64)
        state=MujocoRigidState(**sd)
        for j in range(K):
            env.set_state(state);obs[i,j,0]=context['history_observations'][i,-1]
            for h,a in enumerate(pools['actions'][i,j]):
                result=env.step(a);steps+=1;valid[i,j,h]=True;obs[i,j,h+1]=result.observation[:7]
                events[i,j,h]=[result.success,result.collision,result.info['fall_out'],result.timeout,result.info['rod_ball_contact']]
                if events[i,j,h,:4].any():break
            cost,_=horizon_cost(obs[i,j],pools['actions'][i,j],events[i,j],valid[i,j],context['geometry'][i],context['history_actions'][i,-1]);costs.append(cost)
            goal=context['geometry'][i,:2];n=int(valid[i,j].sum());progress=np.linalg.norm(obs[i,j,0,:2]-goal)-np.linalg.norm(obs[i,j,n,:2]-goal)
            useful.append(bool(not events[i,j,:,1:3].any() and (progress>=.02 or events[i,j,:,0].any())))
    np.savez_compressed(output/'offline_truth.npz',observations=obs,events=events,valid=valid,cost=np.asarray(costs),useful=np.asarray(useful))
    return {'raw_action_usefulness':float(np.mean(useful)),'mean_raw_operational_cost':float(np.mean(costs)),
            'offline_truth_branches':len(picks)*K,'offline_truth_transitions':steps,'raw_pool_size':len(picks)*K}
