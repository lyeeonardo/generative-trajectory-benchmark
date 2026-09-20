#!/usr/bin/env python
"""Resumable four-family full E1 training, selection, calibration, and locked test runner."""
from __future__ import annotations
import argparse,copy,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CONTRACT=ROOT/'configs/experiment1_full.json';E1_CONFIG=ROOT/'configs/experiment1.json'
OUTPUT=ROOT/'outputs/e1_full';STATE=OUTPUT/'state.json';ATTEMPTS=OUTPUT/'attempts.jsonl';FREEZE=OUTPUT/'freeze.json'
PYTHON=Path(sys.executable)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def append(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')
def settings():return read(CONTRACT)
def fits():
    c=settings();return [(family,seed) for seed in c['seeds'] for family in c['families']]
def fit_root(family,seed):return OUTPUT/'fits'/family/f'seed_{seed}'
def recipe_path(family,seed):return OUTPUT/'recipes'/f'{family}_seed{seed}.json'
def selected_path(family,seed):return fit_root(family,seed)/'selected/checkpoint.pt'
def calibration_path(family,seed):return fit_root(family,seed)/'calibration/calibration.json'

def recipe(family,seed):
    c=settings();item=c['recipes'][family];budget=int(item['training']['budget_windows']);common=c['common']
    return {
      'cache':c['cache'],'diagnostic':'configs/training_diagnostic.json',
      'display_name':f"{c['display_names'][family]} full E1 seed {seed}",
      'fit_id':f'e1_{family}_seed{seed}','fit_role':'primary','seed':seed,
      'model':copy.deepcopy(item['model']),'primary_behavior_K':1,
      'oracle_test_guidance':['true_tilt','HIGH_request'],'training_membership':'all',
      'supervision':{**common['supervision'],'high_success_demo_fraction':1.0,
        'meaning':'Frozen all-condition E1 training. HIGH proposal/completion draws use successful TRAIN references; stored labels remain immutable; route labels and private state are excluded.',
        'branch_shift_fraction':0.0 if family=='diffusion' else 0.5},
      'training':{'batch_size':item['training']['batch_size'],'learning_rate':item['training']['learning_rate'],
        'warmup_windows':item['training']['warmup_windows'],'initial_windows':budget,'maximum_windows':budget,
        'pilot_windows':min(5370500,budget),'checkpoint_windows':item['training']['checkpoint_windows'],
        'validation_windows':item['training']['validation_windows'],'progress_seconds':600,'resident_data':True,
        'extension_windows':budget,'extension_lookback_windows':max(item['training']['validation_windows']*4,1),
        'ema_decay':item['training']['ema_decay']},
      'inference_operating_point':{'sampling_temperature':item['sampling_temperature'],'guidance_scale':item['guidance_scale'],
        'source':'Frozen before full E1 training; no weak-model tuning'},
      'checkpoint_selection':'Common development-only rank frozen in configs/experiment1_full.json',
      'status':'frozen_full_e1_recipe','development_training_ready':True,'full_training_ready':True,
      'qualification_override':'Included by the frozen benchmark policy regardless of preliminary qualification'
    }

def initial_state():
    return {'status':'RUNNING','stage':'prepare','created_at':now(),'updated_at':now(),'test_unlocked':False,'test_accessed':False,
            'live_job':None,'fits':{f'{f}/seed_{s}':{'training':'PENDING','selection':'PENDING','calibration':'PENDING','test':'PENDING'} for f,s in fits()}}
def load_state():return read(STATE) if STATE.exists() else initial_state()
def save_state(state):state['updated_at']=now();write(STATE,state)

def gpu_sample():
    query='power.draw,utilization.gpu,memory.used,temperature.gpu,power.limit'
    text=subprocess.check_output(['nvidia-smi',f'--query-gpu={query}','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
    values=[float(x.strip()) for x in text.split(',')]
    return dict(zip(('power_w','utilization_percent','memory_mib','temperature_c','power_limit_w'),values))
def require_power():
    sample=gpu_sample()
    if abs(sample['power_limit_w']-settings()['power_limit_w'])>1:raise RuntimeError('Required 450 W power limit is not set')
    return sample

def run_command(name,command,complete,telemetry_path,retries=2):
    if complete():return {'status':'REUSED_COMPLETE'}
    for attempt in range(1,retries+1):
        state=load_state();state['live_job']={'name':name,'command':command,'attempt':attempt,'started_at':now()};save_state(state)
        append(ATTEMPTS,{'time':now(),'event':'START','name':name,'attempt':attempt,'command':command})
        started=time.monotonic();proc=subprocess.Popen(command,cwd=ROOT)
        while proc.poll() is None:
            try:sample=gpu_sample();append(telemetry_path,{'time':now(),'job':name,'attempt':attempt,**sample})
            except Exception as exc:append(telemetry_path,{'time':now(),'job':name,'telemetry_error':repr(exc)})
            state=load_state();state['live_job']['pid']=proc.pid;state['live_job']['elapsed_seconds']=time.monotonic()-started;save_state(state)
            time.sleep(30)
        elapsed=time.monotonic()-started;ok=proc.returncode==0 and complete()
        append(ATTEMPTS,{'time':now(),'event':'EXIT','name':name,'attempt':attempt,'exit_code':proc.returncode,'elapsed_seconds':elapsed,'verified':ok})
        state=load_state();state['live_job']=None;save_state(state)
        if ok:return {'status':'COMPLETE','elapsed_seconds':elapsed,'attempt':attempt}
    raise RuntimeError(f'{name} failed or did not produce its verified receipt after {retries} attempts')

def prepare():
    c=settings();OUTPUT.mkdir(parents=True,exist_ok=True);require_power()
    cap=int(c['training_budget_policy']['maximum_sampled_windows_per_fit'])
    reference=int(c['recipes'][c['training_budget_policy']['reference_family']]['training']['budget_windows'])
    if reference!=cap:raise RuntimeError('Training-budget reference does not match the declared cap')
    mismatch={family:int(c['recipes'][family]['training']['budget_windows']) for family in c['families'] if int(c['recipes'][family]['training']['budget_windows'])!=cap}
    if mismatch:raise RuntimeError(f'Full E1 fits must use the Diffusion/DiT training budget: {mismatch}')
    audit=read(ROOT/c['dataset_readiness'])
    if audit['status']!='PASS':raise RuntimeError('Dataset readiness failed')
    counts={split:len(np.load(ROOT/c['cache']/split/'actions.npy',mmap_mode='r')) for split in ('train','development','calibration','test')}
    if counts!={'train':53705,'development':6522,'calibration':6666,'test':13212}:raise RuntimeError(f'Dataset counts changed: {counts}')
    hashes={}
    for family,seed in fits():
        path=recipe_path(family,seed);value=recipe(family,seed)
        if path.exists() and read(path)!=value:raise ValueError('Frozen recipe changed: '+str(path))
        write(path,value);hashes[str(path.relative_to(ROOT))]=sha(path)
    receipt={'status':'READY','contract_sha256':sha(CONTRACT),'dataset_manifest_sha256':sha(ROOT/'datasets/uphill_push_v1/manifest.json'),
      'dataset_readiness_sha256':sha(ROOT/c['dataset_readiness']),'normalization_sha256':sha(ROOT/c['normalization']),
      'cache_hashes_sha256':sha(ROOT/c['cache']/'hashes.json'),'counts':counts,'recipe_sha256':hashes,
      'families':c['families'],'seeds':c['seeds'],'fits':len(fits()),'test_accessed':False,'prepared_at':now()}
    write(OUTPUT/'prepare.json',receipt);state=load_state();state['stage']='prepared';save_state(state);return receipt

def smoke():
    prepare()
    for family in settings()['families']:
        seed=13;out=fit_root(family,seed)/'smoke';status=out/'status.json'
        command=[str(PYTHON),'-u','scripts/train.py','--config',str(recipe_path(family,seed)),'--output',str(out),'--stage','smoke','--smoke-steps','3','--device','cuda']
        run_command(f'smoke_{family}',command,lambda p=status:p.exists() and read(p).get('status')=='SMOKE_COMPLETE',OUTPUT/'telemetry/smoke.jsonl')
    state=load_state();state['stage']='smoke_complete';save_state(state)

def train_all():
    prepare()
    for family,seed in fits():
        require_power();root=fit_root(family,seed);out=root/'training';status=out/'status.json';key=f'{family}/seed_{seed}'
        def complete(p=status):return p.exists() and read(p).get('status')=='FULL_BUDGET_COMPLETE'
        if complete():
            state=load_state();state['fits'][key]['training']='COMPLETE';save_state(state);continue
        command=[str(PYTHON),'-u','scripts/train.py','--config',str(recipe_path(family,seed)),'--output',str(out),'--stage','full','--device','cuda']
        if out.exists():command.append('--resume')
        try:
            run_command(f'train_{family}_seed{seed}',command,complete,root/'training_gpu.jsonl')
            state=load_state();state['fits'][key]['training']='COMPLETE';save_state(state)
        except Exception:
            state=load_state();state['fits'][key]['training']='FAILED_TECHNICAL';state['fits'][key]['training_error']=traceback.format_exc();save_state(state);raise
    state=load_state();state['stage']='training_complete';save_state(state)

def metric_rank(row):
    return (int(bool(row['prediction_qualified'])),float(row['raw_action_usefulness']),-float(row['one_step_nll']),
            -float(row['fixed_prediction_rmse']),-int(row['windows']))
def eligible_candidates(history,training_root,budget):
    cap=min(int(budget),int(settings()['checkpoint_selection']['maximum_windows_inclusive']))
    floor=settings()['checkpoint_selection']['minimum_budget_fraction']*int(budget)
    candidates=[m for m in history if floor<=m['windows']<=cap and (Path(training_root)/'checkpoints'/f"w{m['windows']:012d}.pt").exists()]
    return candidates,cap

def select_all():
    for family,seed in fits():
        key=f'{family}/seed_{seed}';root=fit_root(family,seed);status=read(root/'training/status.json')
        if status['status']!='FULL_BUDGET_COMPLETE':raise RuntimeError('Incomplete training: '+key)
        import torch
        latest=torch.load(root/'training/latest.pt',map_location='cpu',weights_only=False);history=latest['history'];budget=recipe(family,seed)['training']['initial_windows']
        candidates,cap=eligible_candidates(history,root/'training',budget)
        if not candidates:raise RuntimeError('No eligible development checkpoint: '+key)
        chosen=max(candidates,key=metric_rank);source=root/'training/checkpoints'/f"w{chosen['windows']:012d}.pt";dest=selected_path(family,seed)
        record={'status':'SELECTED','family':family,'seed':seed,'split':'development','budget_windows':budget,
          'minimum_budget_fraction':settings()['checkpoint_selection']['minimum_budget_fraction'],'maximum_windows_inclusive':cap,'rank':settings()['checkpoint_selection']['rank'],
          'selected_windows':chosen['windows'],'selected_metrics':chosen,'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),
          'eligible_candidates':[{'windows':m['windows'],'rank':metric_rank(m),'prediction_qualified':m['prediction_qualified'],'raw_action_usefulness':m['raw_action_usefulness'],'one_step_nll':m['one_step_nll'],'fixed_prediction_rmse':m['fixed_prediction_rmse']} for m in candidates],
          'test_accessed':False}
        selection=dest.parent/'selection.json'
        if selection.exists():
            old=read(selection)
            if old!=record or not dest.exists() or sha(dest)!=record['source_sha256']:raise ValueError('Selection changed: '+key)
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);write(selection,record)
        state=load_state();state['fits'][key]['selection']='COMPLETE';save_state(state)
    state=load_state();state['stage']='selection_complete';save_state(state)

def calibrate_all():
    for family,seed in fits():
        root=fit_root(family,seed);key=f'{family}/seed_{seed}';point=settings()['recipes'][family];receipt=calibration_path(family,seed)
        command=[str(PYTHON),'-u','evaluation/calibrate_experiment1.py','--checkpoint',str(selected_path(family,seed)),'--output',str(receipt.parent),'--device','cuda','--samples','32','--sampling-temperature',str(point['sampling_temperature']),'--guidance-scale',str(point['guidance_scale'])]
        run_command(f'calibrate_{family}_seed{seed}',command,lambda p=receipt:p.exists() and read(p).get('status')=='COMPLETE',root/'calibration_gpu.jsonl')
        state=load_state();state['fits'][key]['calibration']='COMPLETE';save_state(state)
    state=load_state();state['stage']='calibration_complete';save_state(state)

def release_tests():
    """Run the retained release suite and record the exact pre-freeze result."""
    command=[str(PYTHON),"-m","pytest","-q"]
    completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    receipt={"status":"PASS" if completed.returncode==0 else "FAIL","command":command,
      "returncode":completed.returncode,"ran_at":now(),"stdout":completed.stdout[-12000:],"stderr":completed.stderr[-12000:]}
    write(OUTPUT/"checks/pytest.json",receipt)
    if completed.returncode:raise RuntimeError("Release tests failed; see outputs/e1_full/checks/pytest.json")
    return receipt

def freeze():
    release_tests()
    selected={};calibrations={}
    for family,seed in fits():
        cp=selected_path(family,seed);cal=calibration_path(family,seed)
        if not cp.exists() or not cal.exists():raise RuntimeError('Selection/calibration incomplete')
        if read(cal)['checkpoint_sha256']!=sha(cp):raise ValueError('Calibration/checkpoint mismatch')
        selected[f'{family}/seed_{seed}']={'path':str(cp.relative_to(ROOT)),'sha256':sha(cp),'selection_sha256':sha(cp.parent/'selection.json')}
        calibrations[f'{family}/seed_{seed}']={'path':str(cal.relative_to(ROOT)),'sha256':sha(cal)}
    config=read(E1_CONFIG);config.update(status='ready_for_locked_test_execution',execution_ready=True,
      full_contract='configs/experiment1_full.json',full_freeze=str(FREEZE.relative_to(ROOT)),final_families=settings()['families'],final_seeds=settings()['seeds'])
    write(E1_CONFIG,config)
    from evaluation.common import source_identity
    sources=source_identity();sources.update({'scripts/experiment1.py':sha(ROOT/'scripts/experiment1.py'),'scripts/experiment1_full.py':sha(Path(__file__)),
      'evaluation/calibrate_experiment1.py':sha(ROOT/'evaluation/calibrate_experiment1.py'),'configs/experiment1.json':sha(E1_CONFIG),'configs/experiment1_full.json':sha(CONTRACT)})
    record={'status':'READY','frozen_at':now(),'test_accessed':False,'contract_sha256':sha(CONTRACT),'experiment1_sha256':sha(E1_CONFIG),
      'sources':sources,'selected':selected,'calibrations':calibrations,'recipes':{str(recipe_path(f,s).relative_to(ROOT)):sha(recipe_path(f,s)) for f,s in fits()},
      'dataset_manifest_sha256':sha(ROOT/'datasets/uphill_push_v1/manifest.json'),'evaluation_manifest_sha256':sha(ROOT/'datasets/evaluation/manifest.json'),
      'evaluation_audit_sha256':sha(ROOT/'datasets/evaluation/audit.json'),'test_protocol_sha256':sha(ROOT/'datasets/evaluation/test_cases.json'),
      'success_criterion_sha256':sha(ROOT/'configs/success_criterion.json'),'pytest_sha256':sha(OUTPUT/'checks/pytest.json')}
    write(FREEZE,record);state=load_state();state['stage']='frozen';state['test_unlocked']=True;save_state(state);return record

def verify_freeze():
    record=read(FREEZE)
    if record['status']!='READY' or record['test_accessed']:raise RuntimeError('Invalid E1 freeze')
    for name,digest in record['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for row in record['selected'].values():
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Selected checkpoint changed')
    for row in record['calibrations'].values():
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Calibration changed')
    return record

def test_all():
    verify_freeze();state=load_state();state['test_accessed']=True;state['stage']='locked_test_running';save_state(state)
    for family,seed in fits():
        root=fit_root(family,seed);key=f'{family}/seed_{seed}';cp=selected_path(family,seed);cal=calibration_path(family,seed);point=settings()['recipes'][family]
        common=[str(PYTHON),'-u','scripts/experiment1.py','--checkpoint',str(cp),'--split','test','--device','cuda','--sampling-temperature',str(point['sampling_temperature']),'--guidance-scale',str(point['guidance_scale'])]
        cap=root/'test/capabilities/report.json';every=root/'test/behavior/every_step/report.json';reuse=root/'test/behavior/agreement/report.json'
        wrong=root/'test/wrong_tilt/every_step/report.json';scale=root/'test/scaling/report.json'
        jobs=[('capabilities',common+['--panel','capabilities','--calibration',str(cal),'--output',str(root/'test/capabilities')],lambda p=cap:p.exists() and read(p).get('status')=='COMPLETE'),
              ('behavior',common+['--panel','behavior','--policy','both','--output',str(root/'test/behavior')],lambda a=every,b=reuse:a.exists() and b.exists() and read(a).get('status')=='COMPLETE' and read(b).get('status')=='COMPLETE'),
              ('wrong_tilt',common+['--panel','behavior','--policy','every_step','--tilt-label','wrong','--output',str(root/'test/wrong_tilt')],lambda p=wrong:p.exists() and read(p).get('status')=='COMPLETE'),
              ('scaling',common+['--panel','scaling','--output',str(root/'test/scaling')],lambda p=scale:p.exists() and read(p).get('status')=='COMPLETE')]
        for panel,command,complete in jobs:
            run_command(f'test_{panel}_{family}_seed{seed}',command,complete,root/f'test_{panel}_gpu.jsonl')
        state=load_state();state['fits'][key]['test']='COMPLETE';save_state(state)
    state=load_state();state['stage']='test_complete';save_state(state)

def audit():
    errors=[];record=read(FREEZE) if FREEZE.exists() else None
    try:verify_freeze()
    except Exception as exc:errors.append(repr(exc))
    complete=0
    for family,seed in fits():
        root=fit_root(family,seed)
        expected=[root/'test/capabilities/report.json',root/'test/behavior/every_step/report.json',root/'test/behavior/agreement/report.json',root/'test/wrong_tilt/every_step/report.json',root/'test/scaling/report.json']
        missing=[str(p.relative_to(ROOT)) for p in expected if not p.exists()]
        if missing:errors.append(f'{family}/seed_{seed} missing {missing}')
        else:
            complete+=1
            if any(read(p).get('status')!='COMPLETE' for p in expected):errors.append(f'{family}/seed_{seed} incomplete report status')
    result={'status':'PASS' if not errors else 'FAIL','errors':errors,'complete_fits':complete,'expected_fits':len(fits()),'audited_at':now(),
      'freeze_sha256':sha(FREEZE) if FREEZE.exists() else None,'test_accessed':load_state()['test_accessed']}
    write(OUTPUT/'audit.json',result);return result

def status():
    state=load_state();state['completed_training_fits']=sum(v['training']=='COMPLETE' for v in state['fits'].values());state['completed_test_fits']=sum(v['test']=='COMPLETE' for v in state['fits'].values());return state

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['status','prepare','smoke','train','select','calibrate','freeze','test','audit','all'],default='status');args=parser.parse_args()
    if args.stage=='status':result=status()
    elif args.stage=='prepare':result=prepare()
    elif args.stage=='smoke':smoke();result=status()
    elif args.stage=='train':train_all();result=status()
    elif args.stage=='select':select_all();result=status()
    elif args.stage=='calibrate':calibrate_all();result=status()
    elif args.stage=='freeze':result=freeze()
    elif args.stage=='test':test_all();result=status()
    elif args.stage=='audit':result=audit()
    else:
        smoke();train_all();select_all();calibrate_all();freeze();test_all();result=audit()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
