"""Measured standalone K/H query scaling, with fresh calls and fixed sample budget."""
from pathlib import Path
import json,time
import numpy as np
import torch
from data.preparation import write_json,sha
from evaluation.common import atomic_npz
from evaluation.common import source_identity, BASE,CONFIG,PUBLIC,METHODS,load_model,configure,identity_file
from data.banks import load_bank,offline_actions
from evaluation.capabilities import numpy_dict


def evaluate(checkpoint,output,device='cuda',*,smoke=False,sampling_temperature=1.,guidance_scale=1.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    role='development' if smoke else 'test'
    identity_file(output/'run.json',{'checkpoint_sha256':sha(checkpoint),'sources':source_identity(),'source_sha256':sha(Path(__file__)),'contract_sha256':sha(CONFIG/'experiment1.json'),'role':role,'smoke':smoke,'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale})
    if (output/'report.json').exists():return json.loads((output/'report.json').read_text())
    configure(101,device);model,saved=load_model(checkpoint,device)
    if not np.isfinite(sampling_temperature) or sampling_temperature<=0:raise ValueError('Sampling temperature must be finite and positive')
    if not np.isfinite(guidance_scale) or guidance_scale<1:raise ValueError('Guidance scale must be finite and at least one')
    if model.config.method!='diffusion' and guidance_scale!=1.:raise ValueError('Guidance override is diffusion-only')
    model.sampling_temperature=float(sampling_temperature);model.guidance_scale=float(guidance_scale)
    bank,meta=load_bank(role);tilts=json.loads((CONFIG/'experiment1.json').read_text())['tilts_degrees'];picks=[]
    # Exactly one anchor per tilt/stratum; cycle the five layouts independently of model output.
    for z,tilt in enumerate(tilts):
        for phase in range(3):
            options=[i for i,m in enumerate(meta) if m['tilt_degrees']==tilt and m['anchor']==phase];picks.append(options[(z+phase)%len(options)])
    if smoke:picks=picks[:2]
    rows=[];truth_steps=0
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
    for i in picks:
        context={k:bank[k][i:i+1] for k in PUBLIC}
        for K in [1,8,16,32]:
            for H in [1,3,6]:
                # Warmup per distinct shape; its separate RNG never advances measured queries.
                wg=torch.Generator(device=device).manual_seed(77)
                warm=model.propose_joint(context,K=K,H=H,quality=2,generator=wg)
                model.predict({k:np.repeat(v,K,0) for k,v in context.items()},warm['actions'].reshape(K,H,3),samples=16,generator=wg);sync()
                for seed in ([101] if smoke else [101,102,103]):
                    dest=output/f'anchor{i:03d}_K{K}_H{H}_rng{seed}';dest.mkdir(exist_ok=True)
                    if (dest/'record.json').exists():record=json.loads((dest/'record.json').read_text())
                    else:
                        gen=torch.Generator(device=device).manual_seed(seed*100000+i)
                        sync();baseline=torch.cuda.memory_allocated() if device.startswith('cuda') else 0
                        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                        started=time.perf_counter();p=model.propose_joint(context,K=K,H=H,quality=2,generator=gen);sync();pm=time.perf_counter()-started
                        t=time.perf_counter();pred=model.predict({k:np.repeat(v,K,0) for k,v in context.items()},p['actions'].reshape(K,H,3),samples=16,generator=gen);sync();fm=time.perf_counter()-t
                        peak=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0
                        atomic_npz(dest/'committed.npz',**{'joint_'+k:v for k,v in numpy_dict(p).items()},**{'fixed_'+k:v for k,v in numpy_dict(pred).items()})
                        record={'snapshot':meta[i],'K':K,'H':H,'seed':seed,'proposal_seconds':pm,'prediction_seconds':fm,'combined_seconds':pm+fm,
                            'peak_allocated_bytes':peak,'incremental_peak_bytes':max(peak-baseline,0),'active_tokens':2*H,'tensor_tokens':12,
                            'prediction_samples_per_action_sequence':16,'proposal_work':model.inference_work('proposal',H),'prediction_work':model.inference_work('prediction',H)}
                        if seed==101:
                            t=offline_actions([meta[i]],p['actions'].cpu().numpy(),context,dest/'truth.npz');record.update(actual_branch_steps=int(t['actual_steps']),mean_actual_cost=float(t['cost'].mean()),unsafe_fraction=float(t['events'][:,:,:,1:3].any((2,3)).mean()))
                        write_json(dest/'record.json',record)
                    rows.append(record);truth_steps+=record.get('actual_branch_steps',0)
    report={'status':'SMOKE_COMPLETE' if smoke else 'COMPLETE','role':role,'method':model.config.method,'display_name':METHODS[model.config.method],'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,'rows':rows,'query_bundles':len(rows),
        'raw_sequences':sum(x['K'] for x in rows),'truth_branches':sum(x['K'] for x in rows if x['seed']==101),'truth_steps':truth_steps,
        'warmup_bundles':len(picks)*12,'warmup_rule':'One separate unscored bundle per snapshot/grid shape; no benchmark RNG consumed',
        'scope':'Standalone calls, all raw proposals; padded token lengths disclosed; no control, belief, MI, ranking or fallback'}
    write_json(output/'report.json',report);return report
