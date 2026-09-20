"""Public capability banks and explicitly offline simulator truth."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from data.preparation import ROOT, sha, write_json
from data.constants import TILTS, PUBLIC
from evaluation.common import BANK, identity_file, public_context, atomic_npz
from data.reference import pack_state, plan_branches, route_waypoints, waypoint_action
from aif.operational_cost import applied_action, horizon_cost
COLLECTION = ROOT / "datasets/training/collection_config.json"
def geometry_key(scene):return tuple(np.round(np.r_[scene['start'],scene['goal'],scene['obstacle_center'],scene['obstacle_radius']],5))


def make_env(case):
    from envs.mujoco_tilted_board import MujocoRigidTiltPushEnv
    from mujoco_task.config import get_preset
    from mujoco_task.sim.scene import SceneSpec
    collection=json.loads(COLLECTION.read_text());sim=replace(get_preset('run').sim,**collection['simulator'])
    env=MujocoRigidTiltPushEnv(sim,collection['physics']);obs=env.reset(SceneSpec(**case['scene']));return env,obs,sim


def restore(env,record):
    from envs.mujoco_tilted_board import MujocoRigidState
    state=dict(record)
    # Older frozen bank records carried task-wrapper settling state. H6 branch
    # replay restores the rigid simulator only; all physical state fields remain.
    state.pop('settle_count',None)
    for key in ['qpos','qvel','mocap_pos','mocap_quat','integration_state']:
        if state[key] is not None:state[key]=np.asarray(state[key],np.float64)
    env.set_state(MujocoRigidState(**state))


def mode_label(context,observations,events,valid):
    n=int(valid.sum())
    if not n or events[:n,1:3].any():return 0
    origin=context['history_observations'][-1];goal=context['geometry'][:2]
    delta=goal-origin[:2];direction=delta/max(np.linalg.norm(delta),1e-9);normal=np.array([-direction[1],direction[0]])
    progress=np.linalg.norm(delta)-np.linalg.norm(observations[n-1,:2]-goal)
    if progress<.02 and not events[:n,0].any():return 0
    lateral=float((observations[n-1,:2]-origin[:2])@normal)
    return 1 if lateral>.02 else -1 if lateral<-.02 else 0


def build_case(case):
    if case['role']=='reserved_e2':raise ValueError('Reserved E2 geometry cannot be executed')
    public=BANK/'public'/(case['case_id']+'.npz');private=BANK/'private'/(case['case_id']+'.json');receipt=BANK/'records'/(case['case_id']+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if sha(public)!=r['public_sha256'] or sha(private)!=r['private_sha256']:raise ValueError('Bank artifacts changed')
        return r
    env,obs,sim=make_env(case);scene=case['scene']
    routes=route_waypoints(np.asarray(scene['start']),np.asarray(scene['goal']),obstacle_center=np.zeros(2),obstacle_radius=.1,ball_radius=.04,workspace_x=sim.workspace_x,workspace_y=sim.workspace_y)
    attempts=[];best=None;rollin_steps=0
    for ri,route in enumerate(routes):
        env,obs,sim=make_env(case);hist=[obs.copy()];actions=[];events=[];states=[];way=0
        for t in range(100):
            states.append(pack_state(env.state.copy()))
            if np.linalg.norm(obs[:2]-route[way])<(.15 if way<len(route)-1 else .12):way=min(way+1,len(route)-1)
            request=waypoint_action(obs,route[way],lateral_tilt=scene['lateral_tilt'],longitudinal_tilt=scene['longitudinal_tilt'],
                ball_radius=.04,rod_half_width=.03,action_max_speed=.8,action_max_omega=4.,position_gain=6.5,push_speed=.58,drift_compensation=.12)
            a=applied_action(request);result=env.step(a);rollin_steps+=1;obs=result.observation.copy();hist.append(obs)
            actions.append(a);events.append([result.success,result.collision,result.info['fall_out'],result.timeout,result.info['rod_ball_contact']])
            if result.done:break
        cost,_=horizon_cost(np.asarray(hist)[:,:7],np.asarray(actions),np.asarray(events),np.ones(len(actions),bool),hist[0][7:12],np.zeros(3))
        record={'route':ri,'steps':len(actions),'success':bool(events[-1][0]),'actual_cost':cost};attempts.append(record)
        key=(not record['success'],cost,ri)
        if best is None or key<best[0]:best=(key,np.asarray(hist),np.asarray(actions),np.asarray(events,bool),states)
    _,hist,actions,events,states=best;L=len(actions)
    if L<3:raise RuntimeError('Physical rollin cannot supply3 real anchors; retain geometry and review explicitly')
    approach=int(np.argmin(np.linalg.norm(hist[1:L-1,:2],axis=1)))+1
    approach=min(approach,L-2);remaining=np.arange(approach+1,L)
    placement=int(remaining[np.argmin(abs(np.linalg.norm(hist[remaining,:2]-np.asarray(scene['goal']),axis=1)-.3))])
    anchors=[0,approach,placement];contexts=[];plans=[];availability=[];mode_steps=0;restore_error=0.
    for ai,t in enumerate(anchors):
        ctx={k:v[0] for k,v in public_context(hist[:t+1,:7],actions[:t],hist[t,7:12],case['tilt_radians'],t).items()};contexts.append(ctx)
        ref=np.zeros((6,3),np.float32);n=min(6,L-t);ref[:n]=actions[t:t+n]
        raw=plan_branches(hist[t],ref,{},np.random.default_rng(91212001+case['base_layout_id']+ai));raw=np.asarray([[applied_action(a) for a in seq] for seq in raw]);plans.append(raw[:16])
        modes=set()
        # Only independently defined side-control references certify LOCAL modes.
        for sequence in raw[4:8]:
            restore(env,states[t]);restore_error=max(restore_error,float(np.max(abs(env.current_observation()[:7]-hist[t,:7]))))
            y=[];ev=[]
            for a in sequence:
                result=env.step(a);mode_steps+=1;y.append(result.observation[:7]);ev.append([result.success,result.collision,result.info['fall_out'],result.timeout,result.info['rod_ball_contact']])
                if result.done:break
            label=mode_label(ctx,np.asarray(y),np.asarray(ev,bool),np.ones(len(y),bool))
            if label:modes.add(label)
        availability.append([int(-1 in modes),int(1 in modes)])
    public.parent.mkdir(parents=True,exist_ok=True)
    atomic_npz(public,**{k:np.stack([c[k] for c in contexts]) for k in PUBLIC},actions=np.asarray(plans),anchor_steps=np.asarray(anchors),available_local_modes=np.asarray(availability,bool))
    write_json(private,{'case':case,'states':[states[t] for t in anchors],'rollin_attempts':attempts,'selected_reference_rule':'Success first, then canonical actual cost, then route ID; independent of all learned models',
        'selected_reference':min(attempts,key=lambda r:(not r['success'],r['actual_cost'],r['route']))})
    rollin=BANK/'rollins'/(case['case_id']+'.npz');rollin.parent.mkdir(parents=True,exist_ok=True);atomic_npz(rollin,observations=hist,actions=actions,events=events)
    r={'case_id':case['case_id'],'role':case['role'],'public_sha256':sha(public),'private_sha256':sha(private),'rollin_sha256':sha(rollin),'rollin_attempts':len(attempts),
        'rollin_steps':rollin_steps,'mode_reference_branches':12,'mode_reference_steps':mode_steps,'maximum_restore_error':restore_error,
        'selected_rollin_success':bool(events[-1,0]),'available_mode_counts':np.sum(availability,axis=1).tolist(),'nominal_strata':['at_rest','obstacle_approach','late_goal_approach'],
        'stratum_limit':'Late anchor is selected by actual distance; if no successful reference, retain and report the failed trajectory rather than fabricate placement.'}
    write_json(receipt,r);return r


def load_bank(role,limit=None):
    manifest=json.loads((BANK/'manifest.json').read_text());cases=[c for c in manifest['cases'] if c['role']==role];rows=[];meta=[]
    for case in cases:
        with np.load(BANK/'public'/(case['case_id']+'.npz')) as f:
            for ai in range(3):
                rows.append({k:f[k][ai] for k in f.files});meta.append({'case_id':case['case_id'],'anchor':ai,'base_layout_id':case['base_layout_id'],'tilt_degrees':case['tilt_degrees']})
    if limit:rows=rows[:limit];meta=meta[:limit]
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]},meta


def offline_actions(meta,actions,context,output):
    """Call only after learned outputs/action requests are durably committed."""
    n,K,H,_=actions.shape;obs=np.zeros((n,K,H,7),np.float32);events=np.zeros((n,K,H,5),bool);valid=np.zeros((n,K,H),bool);cost=np.zeros((n,K));modes=np.zeros((n,K),int);steps=0
    for i,row in enumerate(meta):
        from environment.task import UphillTask
        from environment.success import history_count
        from mujoco_task.sim.scene import SceneSpec
        private=json.loads((BANK/'private'/(row['case_id']+'.json')).read_text())
        env=UphillTask();env.reset(SceneSpec(**private['case']['scene']));state=private['states'][row['anchor']]
        dwell=history_count(context['history_observations'][i],context['geometry'][i,:2],mask=context['history_mask'][i],
            elapsed_steps=int(round(float(context['time_fraction'][i])*100)))
        for j in range(K):
            restore(env,state);env.settle_count=dwell
            for h,request in enumerate(actions[i,j]):
                if not np.isfinite(request).all():break
                result=env.step(applied_action(request));steps+=1;obs[i,j,h]=result.observation[:7];valid[i,j,h]=True
                events[i,j,h]=[result.success,result.collision,result.info['fall_out'],result.timeout,result.info['rod_ball_contact']]
                if result.done:break
            if valid[i,j].any():cost[i,j],_=horizon_cost(np.r_[context['history_observations'][i,-1:],obs[i,j]],actions[i,j],events[i,j],valid[i,j],context['geometry'][i],context['history_actions'][i,-1])
            else:cost[i,j]=float('inf')
            modes[i,j]=mode_label({k:v[i] for k,v in context.items()},obs[i,j],events[i,j],valid[i,j])
    atomic_npz(output,observations=obs,events=events,valid=valid,cost=cost,local_mode=modes,actual_steps=np.asarray(steps))
    return {'observations':obs,'events':events,'valid':valid,'cost':cost,'local_mode':modes,'actual_steps':steps}
