"""Regenerate the frozen scenario bank from preserved reference actions, offline."""
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, multiprocessing as mp, shutil, time
import numpy as np
from data.preparation import ROOT, FIELDS, empty_batch, batch_costs, sha, write_json, tilt_ids
from data.reference import plan_branches, pack_state
from evaluation.common import atomic_npz as write_npz
from aif.operational_cost import applied_action, quality_labels
from mujoco_task.config import get_preset
from mujoco_task.sim.scene import SceneSpec
from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv
OUT = None
CONFIG = None
REFERENCE = None
REFERENCE_REQUESTS = None
def collect(case):
 path=OUT/'raw'/(case['case_id']+'.npz');receipt=OUT/'records'/(case['case_id']+'.json')
 if receipt.exists():
  r=json.loads(receipt.read_text())
  if r['raw_sha256']!=sha(path):raise ValueError('Completed case changed')
  return {**r,'reused':True}
 scene=SceneSpec(**case['scene']);env=MujocoRigidTiltPushEnv(replace(get_preset('run').sim,**CONFIG['simulator']),CONFIG['physics'])
 obs=env.reset(scene);history=[obs.copy()];actions=[];events=[];states=[]
 # Requested actions must be projected exactly once, as in original collection.
 reference_actions=REFERENCE_REQUESTS[case['scenario_id']]
 for request in reference_actions:
  states.append(env.state.copy());a=applied_action(request);r=env.step(a)
  actions.append(a);history.append(r.observation.copy());events.append([r.success,r.collision,r.info['fall_out'],r.timeout,r.info['rod_ball_contact']])
  if r.done:break
 history=np.asarray(history,np.float32);actions=np.asarray(actions,np.float32);events=np.asarray(events,bool);L=len(actions)
 if not events[-1,0]:
  write_json(OUT/'failures'/(case['case_id']+'.json'),{'case':case,'reference_steps':L,'terminal':events[-1].tolist()})
  raise ValueError('Reference replay did not succeed; retain failure and investigate: '+case['case_id'])
 anchors=sorted(set(range(0,L,3))|{0,L//3,2*L//3,L-1})
 probes=[anchors.index(0),anchors.index(L//3),anchors.index(2*L//3)]
 rows=[];splits=[];anchor_groups=[];branch_steps=0;max_restore=0.
 def base(t,anchor):
  b=empty_batch(1);n=min(t+1,5);b['history_observations'][0,-n:]=history[t-n+1:t+1,:7];b['history_mask'][0,-n:]=True
  n=min(t,4)
  if n:b['history_actions'][0,-n:]=actions[t-n:t]
  b['geometry'][0]=history[t,7:12];b['tilt'][0]=case['tilt_radians'];b['time_fraction'][0]=t/100
  b['tilt_id'][0]=tilt_ids(np.asarray(case['tilt_radians'])[None])[0]
  b['stratum'][0]=min(2,(t*3)//L);b['base_id'][0]=case['base_layout_id'];b['source_ref'][0]=case['scenario_id']
  b['anchor_id'][0]=case['scenario_id']*1000+t;b['source_row'][0]=len(rows)
  return b
 # Every successful reference time step is available under NULL proposal conditioning.
 for t in range(L):
  b=base(t,t);n=min(6,L-t);b['actions'][0,:n]=actions[t:t+n];b['future_observations'][0,:n]=history[t+1:t+n+1,:7]
  b['valid'][0,:n]=True;b['events'][0,:n]=events[t:t+n];b['event_known'][0,:n]=True
  b['source'][0]=1;b['proposal_eligible'][0]=True;rows.append(b);splits.append(0)
 for ai,t in enumerate(anchors):
  ref=np.zeros((6,3),np.float32);n=min(6,L-t);ref[:n]=actions[t:t+n]
  rng=np.random.default_rng(np.random.SeedSequence([CONFIG['seed'],case['scenario_id'],ai]))
  plans=list(plan_branches(history[t],ref,CONFIG,rng));assign=[0]*len(plans)
  if ai in probes:
   # Two independently seeded sequences per partition, never a copied reference branch.
   for part in [1,2]:
    rr=np.random.default_rng(np.random.SeedSequence([CONFIG['seed'],case['scenario_id'],ai,part,991]))
    plans.extend([ref+rr.normal(0,[.13,.13,.5],size=(6,3)),rr.uniform([-.8,-.8,-4],[.8,.8,4],size=(6,3))]);assign.extend([part,part])
  group=[]
  for j,(request,part) in enumerate(zip(plans,assign)):
   env.set_state(states[t]);max_restore=max(max_restore,float(np.max(np.abs(env.current_observation()[:7]-history[t,:7]))))
   b=base(t,ai)
   for k in range(6):
    a=applied_action(request[k]);r=env.step(a);branch_steps+=1
    b['actions'][0,k]=a;b['future_observations'][0,k]=r.observation[:7];b['valid'][0,k]=True;b['event_known'][0,k]=True
    b['events'][0,k]=[r.success,r.collision,r.info['fall_out'],r.timeout,r.info['rod_ball_contact']]
    if r.done:break
   intentional=(j<8 or 16<=j<20) and part==0
   b['source'][0]=0 if intentional else 2
   b['proposal_eligible'][0]=intentional and not b['events'][0,:,1:3].any()
   group.append(len(rows));rows.append(b);splits.append(part)
  anchor_groups.append(group[:20])
 data={k:np.concatenate([r[k] for r in rows],axis=0) for k in FIELDS}
 for hi,h in enumerate([1,3,6]):
  data['returns'][:,hi]=-batch_costs(data,h)
  for group in anchor_groups:
   q,_=quality_labels(data['returns'][group,hi],data['proposal_eligible'][group],min_spread=CONFIG['label_minimum_spread'])
   data['quality'][group,hi]=q
 data['partition']=np.asarray(splits,np.int8);data['scenario_id']=np.full(len(rows),case['scenario_id'],np.int32)
 data['anchor_steps']=np.asarray(anchors,np.int32)
 write_npz(path,**data)
 write_npz(OUT/'reference_rollins'/(case['case_id']+'.npz'),observations=history[:,:7],actions=actions,events=events)
 write_json(OUT/'private_snapshots'/(case['case_id']+'.json'),{'scene':case['scene'],'states':[pack_state(states[t]) for t in anchors]})
 r={'case_id':case['case_id'],'scenario_id':case['scenario_id'],'base_layout_id':case['base_layout_id'],'reference_success':True,
    'reference_steps':L,'anchors':len(anchors),'train_branches':len(anchors)*20,'diagnostic_branches':12,'branch_steps':branch_steps,
    'rows_by_partition':np.bincount(data['partition'],minlength=3).tolist(),'maximum_restore_error':max_restore,'raw_sha256':sha(path)}
 write_json(receipt,r);return r


def generate(output, reference=ROOT/"datasets/training", workers=12, max_cases=None):
    global OUT, CONFIG, REFERENCE, REFERENCE_REQUESTS
    OUT=Path(output).resolve(); REFERENCE=Path(reference).resolve()
    if OUT.exists(): raise FileExistsError("Use a fresh output directory; existing data is preserved")
    if workers < 1: raise ValueError("workers must be positive")
    manifest=json.loads((REFERENCE/"manifest.json").read_text())
    CONFIG=json.loads((REFERENCE/"collection_config.json").read_text())
    cases=manifest["cases"]
    if max_cases is not None:
        if max_cases < 1: raise ValueError("max_cases must be positive")
        cases=cases[:max_cases]
    request_path=REFERENCE/"reference_requests.npz"
    with np.load(request_path,allow_pickle=False) as saved:
        requests=saved["actions"].copy(); offsets=saved["offset"]; lengths=saved["length"]; scenario_ids=saved["scenario_id"]
        if requests.ndim!=2 or requests.shape[1]!=3 or not np.isfinite(requests).all():
            raise ValueError("Invalid preserved requested actions")
        if not (len(offsets)==len(lengths)==len(scenario_ids)) or len(set(scenario_ids.tolist()))!=len(scenario_ids):
            raise ValueError("Invalid reference request scenario index")
        expected_offsets=np.concatenate(([0],np.cumsum(lengths[:-1])))
        if np.any(lengths<=0) or not np.array_equal(offsets,expected_offsets) or int(lengths.sum())!=len(requests):
            raise ValueError("Invalid reference request offsets or lengths")
        REFERENCE_REQUESTS={int(sid):requests[int(start):int(start+length)]
                            for sid,start,length in zip(scenario_ids,offsets,lengths)}
    if any(case["scenario_id"] not in REFERENCE_REQUESTS for case in cases):
        raise ValueError("Missing preserved requests for a scenario")
    OUT.mkdir(parents=True)
    shutil.copyfile(request_path,OUT/"reference_requests.npz")
    write_json(OUT/"sidecar_hashes.json",{"reference_requests.npz":sha(OUT/"reference_requests.npz")})
    write_json(OUT/"collection_config.json", CONFIG)
    write_json(OUT/"manifest.json", {**manifest,"cases":cases,"config_sha256":sha(OUT/"collection_config.json"),"collection_complete":len(cases)==1200})
    results=[]; started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("fork")) as pool:
        for future in as_completed([pool.submit(collect,case) for case in cases]):
            results.append(future.result())
    report={"status":"COMPLETE" if len(cases)==1200 else "SMOKE_COMPLETE", "cases":len(results),
            "reference_steps":sum(r["reference_steps"] for r in results), "branch_steps":sum(r["branch_steps"] for r in results),
            "seconds":time.monotonic()-started, "source_sha256":sha(Path(__file__)), "training_ready":False}
    write_json(OUT/"generation_status.json", report)
    return report
