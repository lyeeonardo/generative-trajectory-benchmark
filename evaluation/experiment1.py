#!/usr/bin/env python
"""Serial, fully timed K1 behavior and paired agreement reuse for all joint models."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluation.common import source_identity, BASE,CONFIG,METHODS,load_model,configure,identity_file,propose_independent
from evaluation.reuse import run_group,TOLERANCE
from evaluation.common import environment,public_context
from data.preparation import sha,write_json
from environment.success import CRITERION
from data.banks import COLLECTION
from evaluation.replay import audit_group


def evaluate(checkpoint,protocol_path,output,policy='every_step',device='cuda',max_cases=None,max_steps=None,supplied_tilts=None,
             sampling_temperature=1.,guidance_scale=1.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);protocol=json.loads(Path(protocol_path).read_text())
    if not np.isfinite(sampling_temperature) or sampling_temperature<=0:raise ValueError('Sampling temperature must be finite and positive')
    if not np.isfinite(guidance_scale) or guidance_scale<1:raise ValueError('Guidance scale must be finite and at least one')
    if max_cases is not None:
        if max_cases<1:raise ValueError('max_cases must be positive')
        protocol['cases']=protocol['cases'][:max_cases]
        protocol['evaluation_rng_seeds']=protocol['evaluation_rng_seeds'][:1]
        protocol['episodes_per_fit']=len(protocol['cases'])
    if max_steps is not None and not 1<=max_steps<=100:raise ValueError('max_steps must be 1..100')
    if policy not in ['every_step','agreement']:raise ValueError('Main E1 compares exactly two execution rules')
    conditioning_identity = {} if supplied_tilts is None else {'supplied_tilts_radians': supplied_tilts}
    identity_file(output/'run.json',{'checkpoint_sha256':sha(checkpoint),'protocol_sha256':sha(protocol_path),'policy':policy,
        'max_cases':max_cases,'sources':source_identity(),'source_sha256':sha(Path(__file__)),'contract_sha256':sha(CONFIG/'experiment1.json'),'device':device,
        'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,'success_criterion_version':CRITERION.version,
        'success_criterion_sha256':sha(CONFIG/'success_criterion.json'),**conditioning_identity})
    if (output/'report.json').exists():return json.loads((output/'report.json').read_text())
    configure(101,device);model,saved=load_model(checkpoint,device);collection=json.loads(COLLECTION.read_text());rows=[];groups=[]
    if model.config.method!='diffusion' and guidance_scale!=1.:raise ValueError('Guidance override is diffusion-only')
    model.sampling_temperature=float(sampling_temperature);model.guidance_scale=float(guidance_scale)
    first=protocol['cases'][0];first_seed=protocol['evaluation_rng_seeds'][0]
    warmup_tilt=first['tilt_radians'] if supplied_tilts is None else supplied_tilts[f'{first["case_id"]}/rng{first_seed}']
    env,obs=environment(collection,first['scene']);ctx=public_context([obs[:7]],[],obs[7:12],warmup_tilt,0)
    for i in range(3):propose_independent(model,ctx,[torch.Generator(device=device).manual_seed(111+i)])
    if device.startswith('cuda'):torch.cuda.synchronize()
    for case in protocol['cases']:
        for seed in protocol['evaluation_rng_seeds']:
            dest=output/f'{case["case_id"]}_rng{seed}'
            had_committed_steps=any(dest.glob('step_*_proposal.npz'))
            labels=None if supplied_tilts is None else [supplied_tilts[f'{case["case_id"]}/rng{seed}']]
            started=time.perf_counter()
            r=run_group(model,[case],collection,dest,policy=policy,device=device,evaluation_seed=seed,proposal_fn=propose_independent,stop_after_steps=max_steps,supplied_tilts=labels)
            wall=time.perf_counter()-started
            if r['status']=='INTERRUPTED_RESUMABLE':
                return {**r,'status':'SMOKE_COMPLETE','case_id':case['case_id'],'policy':policy,'output':str(output),'full_episode_complete':False}
            row=dict(r['episodes'][0]);durations=[];native=[];final=None
            for path in sorted(dest.glob('step_*_proposal.npz')):
                with np.load(path) as p,np.load(path.with_name(path.name.replace('proposal','outcome'))) as o:
                    if bool(p['new_plan'][0]):durations.append(float(p['generation_seconds']))
                    if o['executed'][0]:
                        final=o['observations'][0];age=int(p['plan_age'][0]);error=p['raw_observations'][0,age]-final
                        error[6]=np.arctan2(np.sin(error[6]),np.cos(error[6]));native.append(error)
            if final is None:final=np.asarray(case['scene']['start'])
            row.update(final_goal_distance=float(np.linalg.norm(final[:2]-case['scene']['goal'])),steps_to_success=row['steps'] if row['success'] else None,
                repredicted_executed_observation_rmse=np.sqrt(np.mean(np.square(native),axis=0)).tolist() if native else None,
                generation_seconds=r['generation_seconds'],environment_seconds=r['environment_seconds'],checker_seconds=r['checker_seconds'],
                median_proposal_seconds=float(np.median(durations)) if durations else None,
                model_checker_seconds_per_action=(r['generation_seconds']+r['checker_seconds'])/max(row['steps'],1),
                original_run_wall_seconds=None if had_committed_steps else wall,
                completion_invocation_wall_seconds=wall,wall_timing_complete=not had_committed_steps)
            # Save timings on first completion; resume never overwrites them.
            if (dest/'episode_summary.json').exists():row=json.loads((dest/'episode_summary.json').read_text())
            else:write_json(dest/'episode_summary.json',row)
            ap=dest/'audit.json'
            if ap.exists():a=json.loads(ap.read_text());assert a['report_sha256']==sha(dest/'report.json')
            else:
                a=audit_group((dest,[case],collection,policy,TOLERANCE,*(() if labels is None else (labels,))));assert not a['errors'] and a['maximum_error']==0
                a={k:v for k,v in a.items() if k!='rows'};a.update(status='PASS',report_sha256=sha(dest/'report.json'));write_json(ap,a)
            rows.append(row);groups.append(str(dest.relative_to(output)))
            write_json(output/'status.json',{'status':'RUNNING','complete_episodes':len(rows),'target':protocol['episodes_per_fit']})
    total_steps=sum(x['steps'] for x in rows);plans=sum(x['sampled_plans'] for x in rows)
    report={'status':'SMOKE_COMPLETE' if max_cases is not None else 'COMPLETE','method':model.config.method,'display_name':METHODS[model.config.method],'policy':policy,'role':protocol['role'],
        'success_criterion_version':CRITERION.version,'episodes':rows,'groups':groups,'started_episodes':len(rows),'unrun_episodes':protocol['episodes_per_fit']-len(rows),'successes':sum(x['success'] for x in rows),
        'goal_entries':sum(x['goal_entry'] for x in rows),
        'success_rate':float(np.mean([x['success'] for x in rows])),'mean_actual_cost':float(np.mean([x['actual_operational_cost'] for x in rows])),
        'terminal_counts':{k:sum(x['terminal']==k for x in rows) for k in ['success','collision','fall','timeout','model_output_failure']},
        'executed_steps':total_steps,'sampled_plans':plans,'cached_step_fraction':1-plans/max(total_steps,1),
        'generation_seconds':sum(x['generation_seconds'] for x in rows),'checker_seconds':sum(x['checker_seconds'] for x in rows),
        'environment_seconds':sum(x['environment_seconds'] for x in rows),
        'model_checker_seconds_per_action':sum(x['generation_seconds']+x['checker_seconds'] for x in rows)/max(total_steps,1),
        'instrumented_wall_seconds':sum(x['original_run_wall_seconds'] for x in rows) if all(x['wall_timing_complete'] for x in rows) else None,
        'episodes_with_complete_wall_timing':sum(x['wall_timing_complete'] for x in rows),
        'complete_episode_instrumented_wall_seconds':sum(x['original_run_wall_seconds'] for x in rows if x['wall_timing_complete']),
        'median_K1_generation_seconds':float(np.median([x['median_proposal_seconds'] for x in rows if x['median_proposal_seconds'] is not None])),
        'independent_replay_steps':total_steps,'unscored_warmup_K1_queries':3,'K':1,'H':6,'internal_simulator_rollouts':0,'candidate_ranking':False,'fallback_steps':0,
        'tilt_conditioning':'oracle' if supplied_tilts is None else 'controlled_supplied_label',
        'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,'inference_work':model.inference_work('proposal',6),
        'timing_scope':'Serial synchronized model timing; environment and common checker instrumentation separately recorded. Original full wall time is unavailable after a partial resume; committed component timings remain complete.',
        'checkpoint_sha256':sha(checkpoint),'training_seed':saved['contract']['config']['seed']}
    write_json(output/'report.json',report);return report
