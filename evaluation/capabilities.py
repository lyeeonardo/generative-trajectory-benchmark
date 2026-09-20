"""Full E1 snapshot capabilities and measured query scaling; no controller simulator."""
from pathlib import Path
import json,time
import numpy as np
import torch
from data.preparation import sha,write_json
from data.banks import load_bank,offline_actions
from evaluation.common import source_identity, BANK,BASE,CONFIG,PUBLIC,METHODS,load_model,configure,identity_file
from evaluation.common import atomic_npz
from training.joint_validation import kinematic
from aif.observation import calibrate,law_from_samples,moments,ObservationLaw


def numpy_dict(d):return {k:v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k,v in d.items()}


def circular_error(a,b):
    d=np.asarray(a)-np.asarray(b);d=d.copy();d[...,6]=np.arctan2(np.sin(d[...,6]),np.cos(d[...,6]));return d


def save_predictions(model,bank,path,H,samples=32,seed=9041):
    if path.exists():
        with np.load(path) as f:return {k:f[k] for k in f.files}
    context={k:np.repeat(bank[k],16,axis=0) for k in PUBLIC};actions=bank['actions'].reshape(-1,6,3)[:,:H];outputs=[]
    rng=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    for start in range(0,len(actions),32):
        outputs.append(numpy_dict(model.predict({k:v[start:start+32] for k,v in context.items()},actions[start:start+32],samples=samples,generator=rng)))
    result={k:np.concatenate([d[k] for d in outputs]) for k in outputs[0]};atomic_npz(path,**result);return result


def fixed_truth(bank,meta,commit):
    if not Path(commit).exists():raise ValueError('Commit predictions before opening offline truth')
    role=json.loads((BANK/'private'/(meta[0]['case_id']+'.json')).read_text())['case']['role']
    path=BANK/'truth'/f'{role}_fixed16.npz'
    if path.exists():
        with np.load(path) as f:return {k:f[k] for k in f.files}
    raise FileNotFoundError('Frozen offline truth is missing: '+str(path))



def physical_metrics(pred,truth,valid):
    err=circular_error(pred,truth);e=err[valid]
    return {'rmse':np.sqrt(np.mean(e**2,axis=0)).tolist(),'mae':np.mean(abs(e),axis=0).tolist(),'bias':np.mean(e,axis=0).tolist(),'absolute_error_p95':np.quantile(abs(e),.95,axis=0).tolist(),
        'ball_position_vector_rmse':float(np.sqrt(np.mean(np.sum(e[:,:2]**2,axis=1)))),'valid_targets':int(valid.sum())}


def mean_prediction(samples):
    return moments(torch.as_tensor(samples))[0].numpy()

def energy_score(samples,truth,valid,scale):
    """TRAIN-scaled multivariate energy score with paired sample differences."""
    x=np.asarray(samples,np.float64);y=np.asarray(truth,np.float64);valid=np.asarray(valid,bool)
    scale=np.asarray(scale,np.float64);delta=(x-y[:,None])/scale
    delta[...,6]=np.arctan2(np.sin(x[...,6]-y[:,None,6]),np.cos(x[...,6]-y[:,None,6]))/scale[6]
    first=np.linalg.norm(delta,axis=-1).mean(1)
    # A deterministic half-split estimates E||X-X'|| without an O(S^2) tensor.
    half=x.shape[1]//2;a=x[:,:half];b=x[:,half:2*half];pair=(a-b)/scale
    pair[...,6]=np.arctan2(np.sin(a[...,6]-b[...,6]),np.cos(a[...,6]-b[...,6]))/scale[6]
    second=np.linalg.norm(pair,axis=-1).mean(1)
    return float(np.mean((first-.5*second)[valid]))



def certify_interface(model,bank):
    context={k:v[:2] for k,v in bank.items() if k in PUBLIC};actions=bank['actions'][:2,0,:3].copy();device=next(model.parameters()).device
    # Same RNG; fixed-action mode removes preferences/goal, and future actions cannot affect earlier outputs.
    rng=lambda:torch.Generator(device=device).manual_seed(901)
    a=model.predict(context,actions,samples=2,generator=rng());changed={k:v.copy() for k,v in context.items()};changed['geometry'][:,:2]+=10
    b=model.predict(changed,actions,samples=2,generator=rng());assert torch.equal(a['observations'],b['observations'])
    later=actions.copy();later[:,1:]=0;c=model.predict(context,later,samples=2,generator=rng())
    assert torch.allclose(a['observations'][:,:,0],c['observations'][:,:,0],atol=1e-5,rtol=0)
    assert torch.equal(a['actions'],torch.tensor(actions,device=device)[:,None].expand_as(a['actions']))
    return {'status':'PASS','fixed_action_fidelity':True,'quality_NULL':True,'goal_invariance':True,'future_action_causality':True,'extra_conditional_queries':3,'samples_per_query':2}


def capabilities(checkpoint,role,output,calibration=None,device='cuda',sampling_temperature=1.,guidance_scale=1.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);configure(101,device);model,saved=load_model(checkpoint,device)
    if not np.isfinite(sampling_temperature) or sampling_temperature<=0:raise ValueError('Sampling temperature must be finite and positive')
    if not np.isfinite(guidance_scale) or guidance_scale<1:raise ValueError('Guidance scale must be finite and at least one')
    if model.config.method!='diffusion' and guidance_scale!=1.:raise ValueError('Guidance override is diffusion-only')
    model.sampling_temperature=float(sampling_temperature);model.guidance_scale=float(guidance_scale)
    identity_file(output/'run.json',{'checkpoint_sha256':sha(checkpoint),'bank_manifest_sha256':sha(BANK/'manifest.json'),'bank_audit_sha256':sha(BANK/'audit.json'),
        'calibration_sha256':sha(calibration) if calibration is not None else None,'role':role,'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,
        'sources':source_identity(),'source_sha256':sha(Path(__file__)),'contract_sha256':sha(CONFIG/'experiment1.json')})
    if (output/'report.json').exists():return json.loads((output/'report.json').read_text())
    bank,meta=load_bank(role);N=len(meta);d=json.loads((CONFIG/'experiment1.json').read_text());q=d['prediction_qualification']
    forecasts={h:save_predictions(model,bank,output/f'fixed_H{h}.npz',h,samples=d['prediction_samples'],seed=9041) for h in [1,3,6]}
    truth=fixed_truth(bank,meta,output/'fixed_H6.npz');y=truth['observations'].reshape(-1,6,7);valid=truth['valid'].reshape(-1,6)
    # During design, fit likelihood width on a declared parent-separated half of DEV.
    # TEST evaluation instead requires an externally frozen CAL-only receipt.
    if calibration is None:
        if role!='development':raise ValueError('Final evaluation requires a frozen calibration receipt')
        case_names=sorted(set(m['case_id'] for m in meta));fit_cases=set(case_names[::2])
        fit_snapshot=np.asarray([m['case_id'] in fit_cases for m in meta]);fit=np.repeat(fit_snapshot,16)
        extra=calibrate(torch.tensor(forecasts[1]['observations'][fit,:,0]),torch.tensor(y[fit,0])).tolist()
        rec={'checkpoint_sha256':sha(checkpoint),'extra_variance':extra,'development_fit_cases':sorted(fit_cases),
             'development_score_cases':sorted(set(case_names)-fit_cases),'fit_rows':int(fit.sum()),'scope':'development-only design calibration'}
        identity_file(output/'calibration.json',rec)
    else:
        rec=json.loads(Path(calibration).read_text());assert rec['checkpoint_sha256']==sha(checkpoint);extra=rec['extra_variance']
        identity_file(output/'calibration.json',rec)
    report={'method':model.config.method,'display_name':METHODS[model.config.method],'role':role,'sampling_temperature':sampling_temperature,'guidance_scale':guidance_scale,'snapshots':N,'fixed_action_branches':N*16,'prediction_samples':d['prediction_samples'],'calibrated_extra_variance':extra}
    prediction={}
    for h,p in forecasts.items():
        mean=mean_prediction(p['observations'][:,:,h-1]);prediction[str(h)]=physical_metrics(mean,y[:,h-1],valid[:,h-1]);prediction[str(h)]['energy_score']=energy_score(p['observations'][:,:,h-1],y[:,h-1],valid[:,h-1],np.asarray(model.normalization['observation']['std']))
    repeat={k:np.repeat(bank[k],16,axis=0) for k in PUBLIC};repeat['actions']=bank['actions'].reshape(-1,6,3)
    kin=kinematic(repeat);persistence=np.repeat(repeat['history_observations'][:,-1,None,:],6,axis=1)
    report['prediction']=prediction;report['references']={name:physical_metrics(value,y,valid) for name,value in [('kinematic',kin),('persistence',persistence)]}
    law=law_from_samples(torch.tensor(forecasts[1]['observations'][:,:,0]),extra);err=circular_error(law.mean.numpy(),y[:,0]);variance=law.variance.numpy()
    coverage=np.mean(abs(err)<=1.96*np.sqrt(variance),axis=0);nll=-law.log_prob(torch.tensor(y[:,0])).numpy()
    report.update(one_step_nll=float(nll.mean()),coverage_95=coverage.tolist(),mean_interval_width_95=(3.92*np.sqrt(variance).mean(0)).tolist())
    # Reference likelihood widths use the same declared development fit rows in M4.
    if role=='development':
        cr={k:np.repeat(bank[k],16,axis=0) for k in PUBLIC};cr['actions']=bank['actions'].reshape(-1,6,3)
        fit_cases=set(rec['development_fit_cases']);fit=np.repeat(np.asarray([m['case_id'] in fit_cases for m in meta]),16)
        for name,cmean,dmean in [('kinematic',kinematic(cr)[:,0],kin[:,0]),('persistence',cr['history_observations'][:,-1],persistence[:,0])]:
            residual=calibrate(torch.tensor(cmean[fit])[:,None],torch.tensor(y[fit,0]));rlaw=ObservationLaw(torch.tensor(dmean),residual.expand(len(dmean),7))
            report['references'][name]['H1_calibrated_nll']=float(-rlaw.log_prob(torch.tensor(y[:,0])).mean())
    gates={}
    for h in [1,6]:
        rmse=np.asarray(prediction[str(h)]['rmse'])
        for name,sl in [('ball_position',slice(0,2)),('ball_velocity',slice(2,4)),('yaw',slice(6,7))]:gates[f'H{h}_{name}']=bool(np.sqrt(np.mean(rmse[sl]**2))<=q[f'H{h}_{name}_rmse_max'])
    gates['coverage']=bool((coverage>=q['minimum_each_channel_95_coverage']).all())
    kr=np.asarray(report['references']['kinematic']['rmse']);scale=np.maximum(kr,q['physical_error_scale_floor'])
    gates['variance_not_unbounded']=bool((np.sqrt(variance).mean(0)<=q['maximum_mean_std_over_kinematic_rmse']*scale).all())
    mean6=np.stack([mean_prediction(forecasts[6]['observations'][:,:,h]) for h in range(6)],1)
    mrmse=np.asarray(physical_metrics(mean6,y,valid)['rmse'])
    gates['position_beats_kinematic']=bool((mrmse[:2]<=q['maximum_position_rmse_over_kinematic']*np.maximum(kr[:2],.001)).all())
    report['prediction_gates']=gates;report['prediction_qualified']=all(gates.values());report['interface']=certify_interface(model,bank)
    # By tilt and physical reference phase: all cells retained, not just qualified cells.
    report['prediction_cells']={}
    for tilt in range(len(d['tilts_degrees'])):
        for stratum in range(3):
            ix=[i for i,m in enumerate(meta) if m['tilt_degrees']==d['tilts_degrees'][tilt] and m['anchor']==stratum]
            ids=np.concatenate([np.arange(i*16,(i+1)*16) for i in ix]);report['prediction_cells'][f'tilt{tilt}/stratum{stratum}']=physical_metrics(mean6[ids],y[ids],valid[ids])
    pred_events=forecasts[6]['event_probabilities'].mean(1);actual_events=truth['events'].reshape(-1,6,5)
    report['event_brier']=np.mean((pred_events[valid]-actual_events[valid])**2,axis=0).tolist()
    report['predicted_safe_but_unsafe_fraction']=float(np.mean((pred_events[:,:,:3][:,:,1:3].max((1,2))<.5)&actual_events[:,:,1:3].any((1,2))))
    proposals={}
    for quality in [2,0,1]:
        path=output/f'quality_{quality}_proposals.npz'
        if not path.exists():
            gen=torch.Generator(device=device).manual_seed(9043);chunks=[]
            for start in range(0,N,8):chunks.append(numpy_dict(model.propose_joint({k:bank[k][start:start+8] for k in PUBLIC},K=32,H=6,quality=quality,generator=gen)))
            atomic_npz(path,**{k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]})
        with np.load(path) as f:p={k:f[k] for k in f.files}
        actual=output/f'quality_{quality}_truth.npz'
        if not actual.exists():offline_actions(meta,p['actions'],{k:bank[k] for k in PUBLIC},actual)
        with np.load(actual) as f:t={k:f[k] for k in f.files}
        count=t['valid'].sum(-1);last=t['observations'][np.arange(N)[:,None],np.arange(32)[None,:],np.maximum(count-1,0)]
        before=np.linalg.norm(bank['history_observations'][:,-1,:2]-bank['geometry'][:,:2],axis=1)[:,None]
        progress=before-np.linalg.norm(last[:,:,:2]-bank['geometry'][:,None,:2],axis=-1)
        unsafe=t['events'][:,:,:,1:3].any((2,3));useful=(count>0)&(~unsafe)&((progress>=.02)|t['events'][:,:,:,0].any(2))
        valid_modes=bank['available_local_modes'].all(1);mode=t['local_mode'];coverage_modes=np.asarray([(mode[i]==-1).any()+(mode[i]==1).any() for i in range(N)],float)/2
        # bool + bool uses numpy OR; cast individually to preserve the two-mode count.
        coverage_modes=np.asarray([int((mode[i]==-1).any())+int((mode[i]==1).any()) for i in range(N)])/2
        proposals[str(quality)]={'raw_sequences':N*32,'mean_actual_cost':float(t['cost'].mean()),'unsafe_fraction':float(unsafe.mean()),'useful_fraction':float(useful.mean()),
            'mean_progress_m':float(progress.mean()),'raw_joint_prediction':physical_metrics(p['joint_observations'],t['observations'],t['valid']),
            'action_conditioned_reprediction':physical_metrics(p['observations'],t['observations'],t['valid']),
            'certified_two_mode_snapshots':int(valid_modes.sum()),'mode_coverage_on_certified':float(coverage_modes[valid_modes].mean()) if valid_modes.any() else None,
            'mean_mode_mass':{str(v):float((mode[valid_modes]==v).mean()) if valid_modes.any() else None for v in [-1,1]},'no_filtering_or_selection':True,
            'actual_branch_steps':int(t['actual_steps'])}
    report['proposals']=proposals;report['guidance']={str(q):{'HIGH_minus_comparison_cost':proposals['2']['mean_actual_cost']-proposals[str(q)]['mean_actual_cost'],
        'HIGH_minus_comparison_useful':proposals['2']['useful_fraction']-proposals[str(q)]['useful_fraction']} for q in [0,1]}
    report['tilt_sensing']=sensing(model,bank,y[:,0],extra,output,d['tilts_degrees'])
    report['status']='COMPLETE';report['checkpoint_sha256']=sha(checkpoint);write_json(output/'report.json',report);return report


def sensing(model,bank,truth,extra,output,tilts):
    path=output/'sensing_log_likelihoods.npz'
    if path.exists():
        with np.load(path) as f:logp=f['log_likelihood']
    else:
        values=[];device=next(model.parameters()).device
        for ti,tilt in enumerate(tilts):
            context={k:np.repeat(bank[k],16,axis=0) for k in PUBLIC};context['tilt'][:]=np.deg2rad(tilt)
            actions=bank['actions'].reshape(-1,6,3)[:,:1];preds=[];rng=torch.Generator(device=device).manual_seed(9100)
            for start in range(0,len(actions),32):preds.append(model.predict({k:v[start:start+32] for k,v in context.items()},actions[start:start+32],samples=16,generator=rng)['observations'][:,:,0].cpu())
            samples=torch.cat(preds);atomic_npz(output/f'sensing_hypothesis_{ti}.npz',samples=samples.numpy())
            values.append(law_from_samples(samples,extra).log_prob(torch.tensor(truth)).numpy())
        logp=np.stack(values,1);atomic_npz(path,log_likelihood=logp)
    posterior=torch.tensor(logp).softmax(-1).numpy();actual=np.repeat([tilts.index(np.rad2deg(z).round().astype(int).tolist()) for z in bank['tilt']],16)
    correct=posterior.argmax(1)==actual;confidence=posterior.max(1);ece=0.
    for low in np.arange(0,1,.1):
        ix=(confidence>=low)&(confidence<low+.1 if low<.9 else confidence<=1)
        if ix.any():ece+=ix.mean()*abs(correct[ix].mean()-confidence[ix].mean())
    entropy=-np.sum(posterior*np.log(np.maximum(posterior,1e-30)),1)
    return {'one_step_accuracy':float(correct.mean()),'chance':1/len(tilts),'categorical_nll':float(-np.log(np.maximum(posterior[np.arange(len(actual)),actual],1e-30)).mean()),
        'uniform_nll':float(np.log(len(tilts))),'ECE_10_bins':float(ece),'mean_entropy_reduction_nats':float(np.log(len(tilts))-entropy.mean()),'rows':len(actual),'hypotheses':len(tilts),
        'scope':'Static one-transition sensing with a uniform prior; no maintained belief or AIF controller'}
