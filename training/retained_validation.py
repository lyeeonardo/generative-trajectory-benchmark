"""Development-only diagnostics on retained data; never consume final calibration/test."""
from pathlib import Path
import json,time
import numpy as np
import torch
from data.preparation import write_json
from aif.observation import calibrate,law_from_samples,moments
from aif.operational_cost import horizon_cost
from training.joint_validation import kinematic,error,PUBLIC


def evaluate(model,cache,diagnostic,output):
    started=time.perf_counter();cache=Path(cache);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    contract=json.loads(Path(diagnostic).read_text())
    data={p.stem:np.load(p,mmap_mode='r') for p in (cache/'development').glob('*.npy')}
    select=np.asarray(contract['score_rows']);fit=np.asarray(contract['fit_rows'])
    if set(data['parent_episode'][select]) & set(data['parent_episode'][fit]):raise ValueError('Development fit/score parent leakage')
    bank={k:np.array(v[select]) for k,v in data.items()};cb={k:np.array(v[fit]) for k,v in data.items()}
    device=next(model.parameters()).device
    def predict(b,h,seed):
        with torch.no_grad():return model.predict({k:b[k] for k in PUBLIC},b['actions'][:,:h],samples=contract['prediction_samples'],generator=torch.Generator(device=device).manual_seed(seed))['observations'].cpu()
    preds={h:predict(bank,h,901+h) for h in (1,3,6)};cp=predict(cb,1,811)
    truth=torch.tensor(bank['future_observations']);valid=torch.tensor(bank['valid'])
    extra=calibrate(cp[:,:,0],torch.tensor(cb['future_observations'][:,0]));law=law_from_samples(preds[1][:,:,0],extra)
    mean=torch.stack([moments(preds[6][:,:,h])[0] for h in range(6)],1)
    rmse=error(mean,truth)[valid].square().mean(0).sqrt();kin=error(torch.tensor(kinematic(bank)),truth)[valid].square().mean(0).sqrt()
    h1=error(law.mean,truth[:,0]).square().mean(0).sqrt();h6=error(mean,truth)[valid[:,5],5].square().mean(0).sqrt()
    coverage=(error(law.mean,truth[:,0]).abs()<=1.96*law.variance.sqrt()).float().mean(0)
    with torch.no_grad():pool=model.propose_joint({k:bank[k] for k in PUBLIC},K=contract['proposal_K'],H=6,quality=2,generator=torch.Generator(device=device).manual_seed(991))
    raw={k:v.cpu().numpy() for k,v in pool.items()}
    np.savez_compressed(output/'committed_predictions.npz',**{'joint_'+k:v for k,v in raw.items()},**{f'fixed_H{h}':v.numpy() for h,v in preds.items()})
    write_json(output/'commit.json',dict(score_rows=select.tolist(),fit_rows=fit.tolist(),status='COMMITTED_BEFORE_OFFLINE_EXECUTION'))
    # The offline executor is isolated from learned inference. Restore exactly the
    # retained parent state, including the settle counter, only after commitment.
    from environment.task import UphillTask,scene
    from environment.success import history_count
    from experiments.task_design.build_dataset import state_at,events
    root=cache.parent;parents={}
    for path in (root/'episodes').glob('*/*/episode_*.npz'):
        meta=json.loads(path.with_suffix('.json').read_text());parents[meta['episode_id']]=(path,meta)
    useful=[];costs=[];steps=0;obs=np.zeros((len(select),contract['proposal_K'],6,7),np.float32);ev=np.zeros((len(select),contract['proposal_K'],6,5),bool);vm=np.zeros(ev.shape[:-1],bool)
    for i,parent in enumerate(bank['parent_episode']):
        path,meta=parents[int(parent)]
        with np.load(path) as f:record={k:f[k] for k in f.files}
        env=UphillTask();env.reset(scene(meta['obstacle_x'],meta['lateral_degrees']))
        t=int(bank['anchor_step'][i]);state=state_at(record,t);settle=history_count(record['observations'][:t+1,:7],env.scene.goal,elapsed_steps=t)
        for j,actions in enumerate(raw['actions'][i]):
            env.restore((state,settle))
            for h,a in enumerate(actions):
                r=env.step(a);steps+=1;obs[i,j,h]=r.observation[:7];ev[i,j,h]=events(r);vm[i,j,h]=True
                if r.done:break
            n=int(vm[i,j].sum());now=bank['history_observations'][i,-1];goal=bank['geometry'][i,:2]
            progress=np.linalg.norm(now[:2]-goal)-np.linalg.norm(obs[i,j,n-1,:2]-goal)
            useful.append(bool(not ev[i,j,:,1:3].any() and (progress>=.02 or ev[i,j,:,0].any())))
            costs.append(horizon_cost(np.r_[now[None],obs[i,j]],actions,ev[i,j],vm[i,j],bank['geometry'][i],bank['history_actions'][i,-1])[0])
    np.savez_compressed(output/'offline_truth.npz',observations=obs,events=ev,valid=vm,cost=costs,useful=useful)
    q=contract['qualification'];gates={}
    for h,values in [(1,h1),(6,h6)]:
        for name,sl in [('ball_position',slice(0,2)),('ball_velocity',slice(2,4)),('yaw',slice(6,7))]:gates[f'H{h}_{name}']=float(values[sl].square().mean().sqrt())<=q[f'H{h}_{name}_rmse_max']
    gates['coverage']=bool((coverage>=q['minimum_each_channel_95_coverage']).all())
    gates['position_beats_kinematic']=bool((rmse[:2]<=1.05*kin[:2].clamp_min(.001)).all())
    report=dict(scope='Parent-separated development fit/score diagnostic; not a full-run qualification',prediction_qualified=all(gates.values()),prediction_gates=gates,
        fixed_prediction_rmse=float((rmse/kin.clamp_min(.01)).mean()),one_step_nll=float(-law.log_prob(truth[:,0]).mean()),raw_action_usefulness=float(np.mean(useful)),mean_raw_operational_cost=float(np.mean(costs)),
        rmse=rmse.tolist(),H1_rmse=h1.tolist(),H6_rmse=h6.tolist(),coverage_95=coverage.tolist(),offline_truth_transitions=steps,raw_pool_size=len(useful),seconds=time.perf_counter()-started)
    write_json(output/'metrics.json',report);return report
