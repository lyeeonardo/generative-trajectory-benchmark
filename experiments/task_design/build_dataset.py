"""Create measured nine-condition trajectories and intervention branches, resumably."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.task_design.task import TaskEnv, scene_for, OBSTACLES, TILTS
from experiments.task_design.reference import control
from aif.operational_cost import applied_action, batch_costs, quality_labels
from data.preparation import empty_batch, FIELDS, fit_normalization, sha, write_json
from evaluation.common import atomic_npz
from envs.mujoco_tilted_board import MujocoRigidState

DEFAULT = ROOT / 'datasets/uphill_v1'
SPLITS = {'train': 8, 'development': 2, 'test': 2}
BOUNDS = {'clear':(.24,.4),'duration':(2.8,4.15),'kp':(6,32),'kd':(2.5,8),
          'gravity':(.5,1.4),'gap':(.064,.076),'rodgain':(10,30),'feed':(.02,.12),'yawgain':(5,16)}

def source_contract():
    paths = list(BASE.glob('*.py')) + [ROOT/'envs/mujoco_tilted_board.py', ROOT/'mujoco_task/config.py', ROOT/'mujoco_task/sim/scene.py', ROOT/'aif/operational_cost.py', ROOT/'data/preparation.py', ROOT/'datasets/training/collection_config.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def events(result):
    return [result.success, result.collision, result.info['fall_out'], result.timeout, result.info['rod_ball_contact']]

def measured_side(obs, obstacle):
    # Actual crossing of the obstacle's horizontal centerline, not requested label.
    y = obs[:,1]
    cross = np.flatnonzero((y[:-1]<0)&(y[1:]>=0))
    if not len(cross):return 0
    i=cross[0];w=-y[i]/(y[i+1]-y[i]);x=obs[i,0]+w*(obs[i+1,0]-obs[i,0])
    return -1 if x<obstacle else 1

def rollout(env, scene, p, side):
    obs=env.reset(scene);history=[obs.copy()];actions=[];ev=[];states=[env.state.copy()];settle=[0]
    for t in range(100):
        a=control(obs,scene,t,p,side);r=env.step(a);obs=r.observation
        history.append(obs.copy());actions.append(a);ev.append(events(r));states.append(env.state.copy());settle.append(env.settle_count)
        if r.done:break
    arrays=dict(observations=np.array(history,np.float32),actions=np.array(actions,np.float32),events=np.array(ev,bool),
                settle_counts=np.array(settle,np.int16),integration_state=np.stack([s.integration_state for s in states]),
                integration_spec=np.array(states[0].integration_spec),qpos=np.stack([s.qpos for s in states]),qvel=np.stack([s.qvel for s in states]),
                mocap_pos=np.stack([s.mocap_pos for s in states]),mocap_quat=np.stack([s.mocap_quat for s in states]),
                rod_yaw=np.array([s.rod_yaw for s in states]),step=np.array([s.step for s in states]),time=np.array([s.time for s in states]))
    return arrays

def state_at(d,t):
    return MujocoRigidState(qpos=d['qpos'][t].copy(),qvel=d['qvel'][t].copy(),mocap_pos=d['mocap_pos'][t].copy(),mocap_quat=d['mocap_quat'][t].copy(),
        rod_yaw=float(d['rod_yaw'][t]),step=int(d['step'][t]),time=float(d['time'][t]),integration_state=d['integration_state'][t].copy(),integration_spec=int(d['integration_spec']))

def collect_group(args):
    output,case_id,obstacle,lateral,side,split,quota,seeds=args;output=Path(output)
    folder=output/'episodes'/split/f'c{case_id}_side{side:+d}';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'complete.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        for name,digest in r['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Completed reference changed')
        return r
    rng=np.random.default_rng(np.random.SeedSequence([20260915,case_id,side+1,list(SPLITS).index(split)]))
    env=TaskEnv();scene=scene_for(obstacle,lateral);good=[];attempts=[];pool=[dict(p) for p in seeds]
    for attempt in range(3000):
        if attempt<len(seeds):p=dict(seeds[attempt])
        elif rng.random()<.85:
            template=pool[int(rng.integers(len(pool)))];p={k:float(np.clip(v+rng.normal(0,(BOUNDS[k][1]-BOUNDS[k][0])*.09),*BOUNDS[k])) for k,v in template.items()}
        else:p={k:float(rng.uniform(*v)) for k,v in BOUNDS.items()}
        # Independently seeded parameter variation in all partitions, including first attempts.
        if attempt<len(seeds):p={k:float(np.clip(v+rng.normal(0,(BOUNDS[k][1]-BOUNDS[k][0])*.015),*BOUNDS[k])) for k,v in p.items()}
        data=rollout(env,scene,p,side);success=bool(data['events'][-1,0]);route=measured_side(data['observations'],obstacle)
        record=dict(attempt=attempt,parameters=p,success=success,measured_side=route,steps=len(data['actions']),terminal_events=data['events'][-1,:4].tolist())
        attempts.append(record)
        if success and route==side:
            i=len(good);name=f'episode_{i:02d}'
            atomic_npz(folder/(name+'.npz'),**data)
            meta=dict(case_id=case_id,obstacle_x=obstacle,lateral_degrees=lateral,longitudinal_degrees=20,side=side,split=split,episode_id=case_id*10000+(side+1)*1000+list(SPLITS).index(split)*100+i,parameters=p,attempt=attempt,steps=len(data['actions']),scene={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(scene).items()})
            write_json(folder/(name+'.json'),meta);good.append(name);pool.append(p)
        if len(good)==quota:break
    write_json(folder/'attempts.json',attempts)
    if len(good)!=quota:raise RuntimeError(f'Reference quota failed for {folder}: {len(good)}/{quota}')
    files={p.name:sha(p) for name in good for p in [folder/(name+'.npz'),folder/(name+'.json')]}
    files['attempts.json']=sha(folder/'attempts.json')
    result=dict(case_id=case_id,obstacle_x=obstacle,lateral_degrees=lateral,side=side,split=split,successes=len(good),attempts=len(attempts),files=files)
    write_json(receipt,result);return result

def branches_for(reference,rng):
    n=reference.shape[0];ref=np.zeros((6,3),np.float32);ref[:n]=reference
    if n<6 and n:ref[n:]=ref[n-1]
    # All imposed plans are committed before querying physical outcomes.
    plans=[ref,ref*.8,ref*1.1,ref+rng.normal(0,[.035,.035,.1],(6,3)),ref+rng.normal(0,[.07,.07,.2],(6,3)),ref+rng.normal(0,[.1,.1,.3],(6,3))]
    plans += [np.zeros((6,3)),-ref,ref+rng.normal(0,[.22,.22,.8],(6,3)),np.repeat(rng.uniform([-.8,-.8,-4],[.8,.8,4],(1,3)),6,axis=0),rng.uniform([-.8,-.8,-4],[.8,.8,4],(6,3))]
    for sign in [-1,1]:plans.append(ref+np.array([sign*.25,0,0]))
    return applied_action(np.asarray(plans,np.float32))

def branch_episode(args):
    output,path=args;output=Path(output);path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    dest=output/'windows'/meta['split']/f"{meta['episode_id']}.npz";receipt=dest.with_suffix('.json')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if sha(dest)!=r['sha256']:raise ValueError('Changed windows')
        return r
    with np.load(path) as f:d={k:f[k] for k in f.files}
    env=TaskEnv();scene=scene_for(meta['obstacle_x'],meta['lateral_degrees']);env.reset(scene)
    rows=[];parent=[];anchor_steps=[];branch_index=[];restore_max=0.;physics_steps=0
    L=len(d['actions']);rng=np.random.default_rng([20260916,meta['episode_id']])
    def blank(t,source,j):
        b=empty_batch(1);nh=min(t+1,5);na=min(t,4)
        b['history_observations'][0,-nh:]=d['observations'][t-nh+1:t+1,:7];b['history_mask'][0,-nh:]=True
        if na:b['history_actions'][0,-na:]=d['actions'][t-na:t]
        b['geometry'][0]=d['observations'][t,7:12];b['tilt'][0]=[scene.lateral_tilt,scene.longitudinal_tilt]
        b['time_fraction'][0]=t/100;b['tilt_id'][0]=TILTS.index(meta['lateral_degrees']);b['stratum'][0]=min(2,t*3//L)
        b['base_id'][0]=OBSTACLES.index(meta['obstacle_x']);b['anchor_id'][0]=meta['episode_id']*100+t
        b['source_ref'][0]=meta['episode_id'];b['source_row'][0]=len(rows);b['source'][0]=source
        parent.append(meta['episode_id']);anchor_steps.append(t);branch_index.append(j)
        return b
    for t in range(L):
        b=blank(t,1,-1);n=min(6,L-t)
        b['actions'][0,:n]=d['actions'][t:t+n];b['future_observations'][0,:n]=d['observations'][t+1:t+n+1,:7]
        b['events'][0,:n]=d['events'][t:t+n];b['event_known'][0,:n]=True;b['valid'][0,:n]=True;b['proposal_eligible'][0]=True;rows.append(b)
    groups=[]
    for t in range(0,L,4):
        plans=branches_for(d['actions'][t:t+6],rng);group=[]
        for j,plan in enumerate(plans):
            env.restore((state_at(d,t),int(d['settle_counts'][t])))
            restore_max=max(restore_max,float(np.max(np.abs(env.current_observation()-d['observations'][t]))))
            b=blank(t,0 if j<6 else 2,j)
            for h,a in enumerate(plan):
                r=env.step(a);physics_steps+=1;b['actions'][0,h]=a;b['future_observations'][0,h]=r.observation[:7];b['events'][0,h]=events(r);b['event_known'][0,h]=True;b['valid'][0,h]=True
                if r.done:break
            b['proposal_eligible'][0]=j<6 and not b['events'][0,:,1:3].any();group.append(len(rows));rows.append(b)
        groups.append(group[:6])
    out={k:np.concatenate([r[k] for r in rows]) for k in FIELDS}
    for hi,h in enumerate([1,3,6]):
        out['returns'][:,hi]=-batch_costs(out,h)
        for group in groups:
            q,_=quality_labels(out['returns'][group,hi],out['proposal_eligible'][group],min_spread=.01,minimum_count=4)
            out['quality'][group,hi]=q
    out.update(scenario_id=np.full(len(rows),meta['case_id'],np.int32),parent_episode=np.array(parent,np.int64),anchor_step=np.array(anchor_steps,np.int16),branch_index=np.array(branch_index,np.int16))
    atomic_npz(dest,**out)
    result=dict(episode_id=meta['episode_id'],case_id=meta['case_id'],split=meta['split'],rows=len(rows),physics_steps=physics_steps,max_restore_error=restore_max,sha256=sha(dest),reference_sha256=sha(path))
    write_json(receipt,result);return result

def assemble(output):
    # Split by parent rollout before windowing; remove exact shared context/action windows.
    seen={};audit={};hashes={}
    for split in SPLITS:
        files=sorted((output/'windows'/split).glob('*.npz'));parts=[]
        for p in files:
            with np.load(p) as f:parts.append({k:f[k] for k in f.files})
        data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};keep=[];duplicates=0
        keys=['history_observations','history_actions','history_mask','geometry','tilt','time_fraction','actions','valid']
        for i in range(len(data['valid'])):
            digest=hashlib.sha256(b''.join(np.ascontiguousarray(data[k][i]).tobytes() for k in keys)).digest()
            if digest in seen:duplicates+=1;continue
            seen[digest]=split;keep.append(i)
        data={k:v[keep] for k,v in data.items()};folder=output/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
        for k,v in data.items():np.save(folder/(k+'.npy'),v,allow_pickle=False)
        audit[split]=dict(raw_rows=sum(len(p['valid']) for p in parts),rows=len(keep),exact_duplicates_removed=duplicates,episodes=len(np.unique(data['parent_episode'])),source_counts={str(s):int((data['source']==s).sum()) for s in [0,1,2]},scenario_rows={str(c):int((data['scenario_id']==c).sum()) for c in range(9)},failure_windows=int(data['events'][:,:,1:3].any((1,2)).sum()),proposal_rows=int(data['proposal_eligible'].sum()),completion_rows=int((data['proposal_eligible']&(data['source']==0)&data['valid'][:,5]).sum()))
    write_json(output/'cache/normalization.json',fit_normalization(output/'cache/train'))
    for p in (output/'cache').rglob('*'):
        if p.is_file() and p.name!='hashes.json':hashes[str(p.relative_to(output/'cache'))]=sha(p)
    write_json(output/'cache/hashes.json',hashes);write_json(output/'cache/assembly.json',audit)
    return audit

def run(output,workers,stage):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    seedfile=BASE/'dataset_development/feedback_search.json'
    seeds=[r[1] for r in json.loads(seedfile.read_text()) if r[2]]
    contract=dict(task='uphill_push_avoid_settle_v1',start=[0,-.65],goal=[0,.84],goal_radius=.12,lateral_degrees=list(TILTS),longitudinal_degrees=20,obstacle_x=list(OBSTACLES),obstacle_y=0,obstacle_radius=.1,settle_speed=.1,settle_steps=5,dt=.05,max_steps=100,split_successes_per_condition_per_route=SPLITS,seed=20260915,reference_seed_parameters=seeds,sources=source_contract(),split_unit='independently sampled reference rollout and all its descendants; same nine physical conditions in every split')
    cp=output/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Source/config changed; use a fresh output')
    write_json(cp,contract)
    jobs=[(str(output),oi*3+zi,o,z,side,split,quota,seeds) for oi,o in enumerate(OBSTACLES) for zi,z in enumerate(TILTS) for side in [-1,1] for split,quota in SPLITS.items()]
    if stage in ['references','all']:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=[]
            for r in pool.map(collect_group,jobs):
                results.append(r);print(json.dumps({k:r[k] for k in ['case_id','side','split','successes','attempts']}),flush=True)
        write_json(output/'reference_summary.json',dict(status='COMPLETE',groups=results,successful_episodes=sum(r['successes'] for r in results),attempts=sum(r['attempts'] for r in results)))
    if stage in ['branches','all']:
        if not (output/'reference_summary.json').exists():raise ValueError('Complete references first')
        paths=sorted((output/'episodes').glob('*/*/episode_*.npz'))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=[]
            for i,r in enumerate(pool.map(branch_episode,[(str(output),str(p)) for p in paths])):
                results.append(r)
                if i%20==0:print(json.dumps({'branched_episodes':i+1,'total':len(paths)}),flush=True)
        write_json(output/'branch_summary.json',dict(episodes=len(results),physics_steps=sum(r['physics_steps'] for r in results),max_restore_error=max(r['max_restore_error'] for r in results)))
        print(json.dumps(assemble(output)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=6);p.add_argument('--stage',choices=['references','branches','all'],default='all');args=p.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    run(args.output,args.workers,args.stage)
