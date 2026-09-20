"""Resumable joint-model training with full exposure and development-only selection."""
from __future__ import annotations
import copy,hashlib,json,math,os,random,signal,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
from data.preparation import sha,write_json
from data.joint_training import JointTrainingData,encode_batch
from generators.registry import create_model
from training.retained_validation import evaluate

ROOT=Path(__file__).resolve().parents[1]
SOURCES=tuple(sorted(str(p.relative_to(ROOT)) for folder in
    ('generators','data','dataset','environment','aif','training','envs','mujoco_task')
    for p in (ROOT/folder).rglob('*.py'))) + ('scripts/train.py','scripts/study_guard.py')

def source_contract(config):
    cache=ROOT/config['cache'];diagnostic=ROOT/config['diagnostic'];initial=config.get('initialize_checkpoint')
    return {'sources':{p:sha(ROOT/p) for p in SOURCES},'cache_manifest':sha(cache.parent/'manifest.json'),
            'cache_audit':sha(ROOT/'datasets/READINESS.json'),'cache_files':sha(cache/'hashes.json'),
            'normalization':sha(cache/'normalization.json'),'diagnostic':sha(diagnostic),'config':config,
            'initialization_checkpoint':sha(ROOT/initial) if initial else None}

def setup(seed,device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(True)

def lr_at(windows,config):
    tr=config['training'];base=tr['learning_rate'];warm=tr['warmup_windows'];initial=tr['initial_windows']
    if windows<warm:return base*max(windows,1)/warm
    phase=min(1.,(windows-warm)/max(1,initial-warm));return base*(.1+.9*.5*(1+math.cos(math.pi*phase)))

def should_extend(history,end,initial,lookback=None):
    span=.20*end if lookback is None else lookback
    recent=[m for m in history if end-span<=m['windows']<=end]
    old=[m for m in recent if m['windows']<=end-.75*span];new=[m for m in recent if m['windows']>=end-.25*span]
    if not old or not new:return False,{'reason':'Insufficient frozen-cadence development measurements; do not infer a plateau'}
    gains={}
    for metric,direction in [('raw_action_usefulness',1),('fixed_prediction_rmse',-1)]:
        a=float(np.mean([m[metric] for m in old]));b=float(np.mean([m[metric] for m in new]))
        gains[metric]=direction*(b-a)/max(abs(a),.01)
    return max(gains.values())>=.01,{'relative_gains':gains,'threshold':.01}

def checkpoint_rank(metrics):
    return (int(metrics['prediction_qualified']),metrics['raw_action_usefulness'],-metrics['one_step_nll'],-metrics['windows'])

def denoiser_retention(model,teacher,batch,first_action_weight=10.):
    """Preserve a qualified teacher on canonical TRAIN proposal contexts."""
    center=(batch['geometry'][:,2].abs()<1e-7)&(batch['tilt'][:,0].abs()<1e-7)&(batch['role']==0)
    if not bool(center.any()):return model.position.sum()*0.
    small={k:v[center] for k,v in batch.items() if torch.is_tensor(v)};encoded=encode_batch(small,model.normalization)
    if getattr(model,'is_ar',False):
        # Preserve the teacher's complete mixture-density parameterization under
        # measured teacher-forced prefixes. The AR head has mixtures*15 outputs,
        # so a seven-coordinate diffusion mask cannot be broadcast onto it.
        t=torch.zeros(len(encoded['x']),device=encoded['x'].device)
        with torch.no_grad():target,_=teacher(encoded['x'],encoded,t)
        prediction,_=model(encoded['x'],encoded,t)
        token_loss=(prediction-target).square().mean(-1)
        weight=encoded['semantic'].any(-1).to(token_loss.dtype)
        weight[:,0]*=first_action_weight
        return (token_loss*weight).sum()/weight.sum().clamp_min(1)
    t=torch.rand(len(encoded['x']),device=encoded['x'].device)*.998+.001
    alpha=torch.cos(t[:,None,None]*math.pi/2);sigma=torch.sin(t[:,None,None]*math.pi/2)
    noisy=(alpha*encoded['x']+sigma*torch.randn_like(encoded['x']))
    noisy=torch.where(encoded['known'],encoded['x'],noisy)*encoded['semantic']
    with torch.no_grad():target,_=teacher(noisy,encoded,t)
    prediction,_=model(noisy,encoded,t);weight=encoded['semantic'].to(prediction.dtype)
    weight[:,0,:3]*=first_action_weight
    return ((prediction-target).square()*weight).sum()/weight.sum().clamp_min(1)

def atomic_save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp');torch.save(state,temp);temp.replace(path)

def weight_architecture_config(model_config):
    """Return only fields that determine checkpoint tensor shapes.

    Loss weights and numerical sampling steps are candidate controls, not weight
    architecture. Strict state-dict loading below remains the final shape guard.
    """
    ignored={'first_action_loss_weight','query_loss_weights','sampling_steps'}
    return {key:value for key,value in model_config.items() if key not in ignored}

def train(config,output,stage='smoke',resume=False,smoke_steps=20,validate_at_end=False,device='cuda',stop_after_updates=None,target_windows=None):
    """stop_after_updates is an internal test interruption, never a scientific budget."""
    if stage in ('development','pilot','full'):
        from scripts.study_guard import require_training_ready
        require_training_ready(config,stage)
    output=Path(output).resolve()
    if any(output.is_relative_to((ROOT/name).resolve()) for name in ('archive','datasets','checkpoints','results')):
        raise ValueError('Training outputs must be outside preserved artifacts')
    setup(config['seed'],device);contract=source_contract(config)
    if output.exists() and not resume:raise FileExistsError('New fit/diagnostic needs an unused output directory')
    if stage not in ('smoke','development','pilot','full'):raise ValueError('Unknown training stage')
    if not output.exists():output.mkdir(parents=True)
    cache=ROOT/config['cache'];audit=json.loads((ROOT/'datasets/READINESS.json').read_text())
    if audit['status']!='PASS':raise ValueError('Training requires passing data audit')
    if audit['dataset_manifest_sha']!=sha(cache.parent/'manifest.json'):raise ValueError('Data audit identity mismatch')
    for name,digest in json.loads((cache/'hashes.json').read_text()).items():
        if sha(cache/name)!=digest:raise ValueError('Training cache changed: '+name)
    if device.startswith('cuda'):
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True)
        if any(abs(float(x)-450)>1 for x in gpu.splitlines()):raise RuntimeError('Required 450 W GPU power limit is not set')
    data=JointTrainingData(cache,device,config['seed'],resident=config['training']['resident_data'],
                           high_success_demo_fraction=config['supervision']['high_success_demo_fraction'],
                           branch_shift_fraction=config['supervision'].get('branch_shift_fraction',0.),
                           membership=config.get('training_membership','all'),canonical_condition=config.get('canonical_condition'))
    model=create_model(config['model'],data.normalization).to(device);teacher=None;initial_record=None
    initial=config.get('initialize_checkpoint')
    if initial:
        path=ROOT/initial;saved_initial=torch.load(path,map_location='cpu',weights_only=False)
        if saved_initial.get('normalization')!=data.normalization:raise ValueError('Initialization normalization mismatch')
        if weight_architecture_config(saved_initial.get('model_config',{}))!=weight_architecture_config(config['model']):raise ValueError('Initialization architecture mismatch')
        model.load_state_dict(saved_initial['model'],strict=True)
        initial_record={'path':str(initial),'sha256':sha(path)}
        retention=float(config['supervision'].get('retention_weight',0.))
        if retention:
            teacher=copy.deepcopy(model).eval();teacher.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=.01,fused=device.startswith('cuda'))
    ema_decay=float(config['training'].get('ema_decay',0.))
    if not math.isfinite(ema_decay) or not 0.<=ema_decay<1.:raise ValueError('EMA decay must be finite in [0,1)')
    ema_model=copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None
    default_target=config['training']['pilot_windows'] if stage in ('development','pilot') else config['training']['initial_windows']
    requested_target=default_target if target_windows is None else int(target_windows)
    if stage!='smoke' and not 1<=requested_target<=config['training']['maximum_windows']:raise ValueError('Target windows outside configured development budget')
    windows=updates=0;history=[];extensions=[];best_rank=None;train_seconds=0.;pending_validation=False;target=requested_target;run_kind=stage
    if resume:
        saved=torch.load(output/'latest.pt',map_location=device,weights_only=False)
        if saved['contract']!=contract:raise ValueError('Resume source/data/recipe mismatch; use a new output directory for changed training')
        if (saved['run_kind']=='smoke')!=(stage=='smoke'):raise ValueError('Smoke work cannot be relabeled as a primary fit')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);data.load_state_dict(saved['sampler'])
        if bool(ema_model)!=(saved.get('ema_model') is not None) or float(saved.get('ema_decay',0.))!=ema_decay:raise ValueError('Resume EMA contract mismatch')
        if ema_model is not None:ema_model.load_state_dict(saved['ema_model'])
        windows=saved['windows'];updates=saved['updates'];history=saved['history'];extensions=saved['extensions'];best_rank=saved['best_rank'];train_seconds=saved['train_seconds'];target=saved['target_windows'];run_kind=saved['run_kind'];pending_validation=saved.get('pending_validation',False)
        if stage in ('development','pilot') and target_windows is not None:
            if requested_target<windows:raise ValueError('Requested target is below completed exposure')
            target=max(target,requested_target)
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    tr=config['training'];B=tr['batch_size'];initial=tr['initial_windows'];maximum=tr['maximum_windows']
    limit=smoke_steps*B if stage=='smoke' else target
    last_save=windows;last_progress=time.monotonic();started=time.monotonic();start_windows=windows;start_updates=updates;stop=False
    def request_stop(sig,frame):
        nonlocal stop;stop=True
    previous={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    if not resume:
        for name in SOURCES:
            dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write_json(output/'run.json',{'contract':contract,'run_kind':run_kind,'current_stage':stage,'fit_id':config['fit_id'],
        'parameter_count':sum(p.numel() for p in model.parameters()),'device':device,'torch':torch.__version__,
        'precision':'FP32 parameters/loss with TF32 matrix products; deterministic kernels','power_limit_w':450,
        'initial_windows':initial,'maximum_windows':maximum,'pilot_windows':tr['pilot_windows'],'requested_target_windows':target,
        'training_membership':config.get('training_membership','all'),'eligible_training_windows':len(data.allowed_rows),
        'baseline_reference_batch':256,'registered_fit':stage!='smoke' and config.get('fit_role')!='diagnostic','status':'RUNNING','initialization':initial_record,'status_note':'Initialized weights count as prior design work; windows/updates in this run count new exposure only','loss_recipe':{'first_action_weight':config['model']['first_action_loss_weight'],'query_weights':config['model']['query_loss_weights'],'retention_weight':config['supervision'].get('retention_weight',0.),'ema_decay':ema_decay},'supervision':config['supervision'],'inference_operating_point':config.get('inference_operating_point')})
    log=(output/'progress.jsonl').open('a')
    def save(status):
        elapsed=train_seconds+time.monotonic()-started
        state={'contract':contract,'model':model.state_dict(),'ema_model':ema_model.state_dict() if ema_model is not None else None,'ema_decay':ema_decay,'model_config':config['model'],'normalization':data.normalization,'optimizer':opt.state_dict(),'sampler':data.state_dict(),
               'windows':windows,'updates':updates,'pending_validation':pending_validation,'history':history,'extensions':extensions,'best_rank':best_rank,
               'train_seconds':elapsed,'target_windows':target,'run_kind':run_kind,
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else []}
        atomic_save(output/'latest.pt',state)
        summary={'status':status,'stage':stage,'fit_id':config['fit_id'],'registered_fit':stage!='smoke' and config.get('fit_role')!='diagnostic',
                 'windows':windows,'updates':updates,'initial_windows':initial,'maximum_windows':maximum,
                 'elapsed_seconds':elapsed,'current_session_windows_per_second':(windows-start_windows)/max(time.monotonic()-started,1e-9),
                 'sampler_exposures':data.exposure,'unique_training_windows':int(data.seen.sum()),
                 'unique_training_anchors':len(np.unique(data.raw['anchor_id'][data.seen])),'eligible_training_windows':len(data.allowed_rows),
                 'training_membership':data.membership,
                 'extensions':extensions,'latest_metrics':history[-1] if history else None,
                 'full_fit_complete':status=='FULL_BUDGET_COMPLETE','budget_limited':bool(windows>=maximum and extensions and extensions[-1]['extend_requested']),'scientific_qualification':False}
        write_json(output/'status.json',summary);return summary
    def validate():
        nonlocal best_rank,pending_validation
        pending_validation=True;save('VALIDATING')
        dest=output/'development'/f'w{windows:012d}'
        if dest.exists():
            # Preserve interrupted diagnostic work; the exact model/RNG resumes from pre-validation checkpoint.
            archive=output/'interrupted_diagnostics';archive.mkdir(exist_ok=True)
            dest.rename(archive/(dest.name+'_'+str(time.time_ns())))
        evaluation_model=ema_model if ema_model is not None else model
        metrics=evaluate(evaluation_model,cache,ROOT/config['diagnostic'],dest);metrics['windows']=windows;history.append(metrics)
        checkpoint={'model':evaluation_model.state_dict(),'model_config':config['model'],'normalization':data.normalization,'metrics':metrics,'contract':contract,'ema_decay':ema_decay}
        atomic_save(output/'checkpoints'/f'w{windows:012d}.pt',checkpoint)
        rank=checkpoint_rank(metrics)
        if windows>=initial and (best_rank is None or rank>tuple(best_rank)):
            best_rank=rank;atomic_save(output/'best.pt',checkpoint)
        pending_validation=False;save('RUNNING')
        log.write(json.dumps({'event':'development',**metrics})+'\n');log.flush();print(json.dumps({'event':'development','windows':windows,'qualified':metrics['prediction_qualified'],'useful':metrics['raw_action_usefulness']}),flush=True)
    try:
        if pending_validation:validate()
        while True:
            while windows<limit and not stop:
                before=time.monotonic();model.train();batch=data.sample(min(B,limit-windows));opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=lr_at(windows+B,config)
                if hasattr(model,'set_training_progress'):model.set_training_progress(windows)
                loss,metrics=model.loss(batch)
                if teacher is not None:
                    retention_loss=denoiser_retention(model,teacher,batch,config['model']['first_action_loss_weight'])
                    loss=loss+float(config['supervision']['retention_weight'])*retention_loss;metrics={**metrics,'retention_loss':retention_loss.detach()}
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if ema_model is not None:
                    with torch.no_grad():
                        for average,current in zip(ema_model.parameters(),model.parameters()):average.lerp_(current,1.-ema_decay)
                        for average,current in zip(ema_model.buffers(),model.buffers()):average.copy_(current)
                windows+=len(batch['actions']);updates+=1
                if stop_after_updates is not None and updates-start_updates>=stop_after_updates:stop=True
                do_validation=stage!='smoke' and windows//tr['validation_windows']>(windows-len(batch['actions']))//tr['validation_windows']
                if do_validation:validate()
                if windows-last_save>=tr['checkpoint_windows']:
                    save('RUNNING');last_save=windows
                if time.monotonic()-last_progress>=tr['progress_seconds']:
                    record={'event':'progress','windows':windows,'updates':updates,'loss':float(loss.detach()),**{k:float(v) for k,v in metrics.items()},'session_windows_per_second':(windows-start_windows)/(time.monotonic()-started)}
                    log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True);last_progress=time.monotonic()
            if stop:break
            if (stage!='smoke' or validate_at_end) and (not history or history[-1]['windows']!=windows):validate()
            if stage!='full':break
            extend,detail=should_extend(history,windows,initial,tr.get('extension_lookback_windows'))
            extensions.append({'windows':windows,'extend_requested':extend,**detail})
            if not extend or target>=maximum:break
            target=min(target+tr.get('extension_windows',initial//4),maximum);limit=target;save('EXTENDING')
        status='INTERRUPTED_RESUMABLE' if stop else 'SMOKE_COMPLETE' if stage=='smoke' else 'DEVELOPMENT_MILESTONE_COMPLETE' if stage in ('development','pilot') else 'FULL_BUDGET_COMPLETE'
        summary=save(status);print(json.dumps({k:summary[k] for k in ('status','windows','updates','elapsed_seconds')}),flush=True)
        return summary
    except Exception:
        # Previous atomic checkpoint stays intact; do not claim a failed update completed.
        write_json(output/'failure.json',{'windows':windows,'updates':updates,'status':'FAILED','latest_checkpoint_exists':(output/'latest.pt').exists()})
        raise
    finally:
        log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
