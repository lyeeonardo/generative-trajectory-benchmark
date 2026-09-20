"""Inspectable cache preparation and quality audit for the 200-pair dataset."""
from pathlib import Path
import json
import numpy as np
from data.preparation import ROOT,FIELDS,sha,write_json,fit_normalization,batch_costs
from aif.operational_cost import horizon_cost,quality_labels
from data.constants import TILTS


SPLITS=('train','development_selection','validation_calibration')

def build(root):
 DATA=Path(root)
 manifest=json.loads((DATA/'manifest.json').read_text());config=json.loads((DATA/'collection_config.json').read_text());cache=DATA/'cache'
 if len(manifest['cases'])!=1200:raise ValueError('A complete 200-pair/six-tilt collection is required')
 if cache.exists():raise FileExistsError('Preserve existing cache; audit it or use a fresh output directory')
 cache.mkdir();counts=np.zeros(3,np.int64);refs={}
 for case in manifest['cases']:
  r=json.loads((DATA/'records'/(case['case_id']+'.json')).read_text());counts+=r['rows_by_partition']
  if sha(DATA/'raw'/(case['case_id']+'.npz'))!=r['raw_sha256']:raise ValueError('Raw hash mismatch')
  refs[str(case['scenario_id'])]={'kind':'fresh_200_pair_case','case_id':case['case_id'],'base_layout_id':case['base_layout_id']}
 fields={**FIELDS,'scenario_id':((),'int32'),'diagnostic':((),'bool')};arrays={}
 # Prediction checks cover12 frozen pairs, all six tilts and early/middle/late reference phases.
 diagnostic_pairs=set(manifest['pair_ids'][::17][:12])
 for part,split in enumerate(SPLITS):
  folder=cache/split;folder.mkdir()
  arrays[part]={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=dtype,shape=(int(counts[part]),)+shape) for k,(shape,dtype) in fields.items()}
 offsets=np.zeros(3,int)
 for case in manifest['cases']:
  with np.load(DATA/'raw'/(case['case_id']+'.npz')) as f:
   for part in range(3):
    index=np.flatnonzero(f['partition']==part);n=len(index);sl=slice(offsets[part],offsets[part]+n)
    for k in FIELDS:arrays[part][k][sl]=f[k][index]
    arrays[part]['scenario_id'][sl]=case['scenario_id']
    arrays[part]['diagnostic'][sl]=case['base_layout_id'] in diagnostic_pairs
    offsets[part]+=n
 for a in arrays.values():
  for v in a.values():v.flush()
 del arrays
 norm=fit_normalization(cache/'train');write_json(cache/'normalization.json',norm)
 cfg={'sizes':dict(zip(SPLITS,counts.tolist())),'pair_count':200,'scenario_count':1200,
      'allowed_tilts_degrees':TILTS,'source_table':refs,'source_codes':{'matched_intentional':0,'successful_reference_window':1,'imposed_action':2},
      'source_hashes':{str(p.resolve()):sha(p) for p in [Path(__file__),DATA/'manifest.json',DATA/'collection_config.json']},
      'normalization_sha256':sha(cache/'normalization.json'),'partition_semantics':config['partition_rule'],
      'diagnostic_pair_ids':sorted(diagnostic_pairs),'fixed_prediction_quality':'NULL',
      'schema_note':'Model fields unchanged; scenario_id/diagnostic/provenance never passed as conditioning.'}
 write_json(cache/'manifest.json',cfg)
 hashes={str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*.npy'))};hashes['normalization.json']=sha(cache/'normalization.json')
 write_json(cache/'cache_hashes.json',hashes);return cache

def audit(root):
 DATA=Path(root)
 cache=DATA/'cache';manifest=json.loads((DATA/'manifest.json').read_text());config=json.loads((DATA/'collection_config.json').read_text());errors=[]
 profiles={};coverage=[];return_max=0.;cross_split_duplicates=0;intentional_duplicate_rows=0;all_rows=0;zero_spread=0
 raw_hashes={};refs=0;branch_steps=0;branch_count=0;restore=0.
 for case in manifest['cases']:
  path=DATA/'raw'/(case['case_id']+'.npz');receipt=json.loads((DATA/'records'/(case['case_id']+'.json')).read_text());raw_hashes[str(path.relative_to(DATA))]=sha(path)
  if raw_hashes[str(path.relative_to(DATA))]!=receipt['raw_sha256']:errors.append('raw_hash')
  refs+=receipt['reference_steps'];branch_steps+=receipt['branch_steps'];branch_count+=receipt['train_branches']+receipt['diagnostic_branches'];restore=max(restore,receipt['maximum_restore_error'])
  if not receipt['reference_success']:errors.append('reference_failure')
  with np.load(path) as f:d={k:f[k] for k in f.files}
  n=d['valid'].sum(1)
  if not np.array_equal(d['valid'],np.arange(6)[None,:]<n[:,None]) or (n<1).any():errors.append('invalid_mask')
  for k in ['history_observations','history_actions','geometry','tilt','future_observations','actions','returns']:
   if not np.isfinite(d[k]).all():errors.append('nonfinite_'+k)
  if (np.linalg.norm(d['actions'][...,:2],axis=-1)>.800001).any() or (np.abs(d['actions'][...,2])>4.000001).any():errors.append('action_bounds')
  if d['proposal_eligible'][d['source']==2].any():errors.append('D_action_supervision')
  if (d['quality'][d['partition']!=0]!=0).any():errors.append('heldout_quality')
  if not np.allclose(d['tilt'],case['tilt_radians'],atol=1e-7,rtol=0):errors.append('tilt_mismatch')
  for row in range(len(n)):
   terminal=np.flatnonzero(d['events'][row,:,:4].any(-1)&d['valid'][row])
   if len(terminal)>1 or (len(terminal) and terminal[0]!=n[row]-1):errors.append('terminal_alignment')
  # Distinct applied trajectories within each public anchor, not artificial independent data counts.
  keys=[(int(d['anchor_id'][i]),int(n[i]),d['actions'][i,:n[i]].tobytes()) for i in range(len(n))]
  parts=[set(k for k,p in zip(keys,d['partition']) if p==j) for j in range(3)]
  cross_split_duplicates+=sum(len(parts[i]&parts[j]) for i in range(3) for j in range(i+1,3))
  intentional_duplicate_rows+=int((d['partition']==0).sum())-len(parts[0]);all_rows+=len(n)
  train=d['partition']==0;support=np.zeros((3,3),int);phase=np.zeros((3,3,3),int)
  for hi,h in enumerate([1,3,6]):
   for q in range(3):
    mask=train&d['proposal_eligible']&(d['quality'][:,hi]==q);support[hi,q]=int(mask.sum())
    for ph in range(3):phase[hi,q,ph]=int((mask&(d['stratum']==ph)).sum())
   for anchor in np.unique(d['anchor_id'][train&(d['source']==0)]):
    ix=np.flatnonzero(train&(d['anchor_id']==anchor)&(d['source']!=1))
    expected,detail=quality_labels(d['returns'][ix,hi],d['proposal_eligible'][ix],min_spread=config['label_minimum_spread'])
    if not np.array_equal(expected,d['quality'][ix,hi]):errors.append('quality_label_parity')
    zero_spread+=int(not detail['supported'])
  if (support==0).any():errors.append('scenario_horizon_quality_gap:'+case['case_id'])
  # Check an actual branch at each reference phase against scalar canonical cost.
  for ph in range(3):
   ix=np.flatnonzero(train&(d['source']==0)&(d['stratum']==ph))
   if not len(ix):errors.append('phase_missing');continue
   i=int(ix[len(ix)//2])
   for hi,h in enumerate([1,3,6]):
    y=np.r_[d['history_observations'][i,-1:,:],d['future_observations'][i,:h]]
    cost,_=horizon_cost(y,d['actions'][i,:h],d['events'][i,:h],d['valid'][i,:h],d['geometry'][i],d['history_actions'][i,-1])
    return_max=max(return_max,abs(cost+float(d['returns'][i,hi])))
  coverage.append({'scenario_id':case['scenario_id'],'case_id':case['case_id'],'safe_quality_counts_by_H':support.tolist(),'phase_quality_counts':phase.tolist()})
 for split in SPLITS:
  raw={k:np.load(cache/split/(k+'.npy'),mmap_mode='r') for k in ['scenario_id','base_id','geometry','source','quality','valid']}
  if len(np.unique(raw['base_id']))!=200 or len(np.unique(raw['scenario_id']))!=1200:errors.append('split_scenario_coverage:'+split)
  profiles[split]={'windows':len(raw['source']),'pairs':len(np.unique(raw['base_id'])),'scenarios':len(np.unique(raw['scenario_id'])),
   'physical_target_entries':int(raw['valid'].sum()),'source_windows':np.bincount(raw['source'],minlength=3).tolist(),
   'obstacle_geometries':np.unique(raw['geometry'][:,2:],axis=0).tolist()}
 norm=json.loads((cache/'normalization.json').read_text());reproduced=fit_normalization(cache/'train')==norm
 if not reproduced:errors.append('normalization_not_reproduced')
 if cross_split_duplicates:errors.append('cross_partition_identical_actions')
 if return_max>1e-4:errors.append('cost_parity')
 if restore>1e-7:errors.append('snapshot_restore')
 normalized={}
 for split in SPLITS:
  obs=np.load(cache/split/'history_observations.npy',mmap_mode='r')[:,-1]
  z=(obs-np.asarray(norm['observation']['mean']))/np.asarray(norm['observation']['std'])
  normalized[split]={'absolute_z_p99':np.quantile(np.abs(z),.99,axis=0).tolist(),'absolute_z_max':np.max(np.abs(z),axis=0).tolist()}
 report={'status':'PASS' if not errors else 'FAIL','errors':sorted(set(errors)),'profiles':profiles,'scenario_coverage':coverage,
  'train_only_normalization_reproduced':reproduced,'normalized_observation_profile':normalized,
  'cross_partition_identical_context_action_windows':cross_split_duplicates,'within_train_duplicate_context_action_windows':intentional_duplicate_rows,
  'duplicate_note':'Reference and matched reference branches can repeat the same trajectory with different label/source roles; overlapping windows are correlated, not independent episodes.',
  'unsupported_anchor_horizon_groups_retained_NULL':zero_spread,'maximum_scalar_return_error':return_max,'maximum_snapshot_restore_error':restore,
  'work_accounting':{'successful_reference_episodes':1200,'reference_execution_steps':refs,'new_action_branches':branch_count,'new_branch_execution_steps':branch_steps},
  'scope':'In-domain200-pair dataset. Every geometry is trained. DEV/CAL differ in imposed actions, not geometry; no generalization claim.',
  'dataset_ready':not errors,'model_quality_established':False}
 write_json(DATA/'artifact_hashes.json',raw_hashes);write_json(DATA/'quality_report.json',report)
 write_json(cache/'audit.json',{k:v for k,v in report.items() if k!='scenario_coverage'})
 write_json(DATA/'preparation_status.json',{'status':'DATA_READY' if not errors else 'AUDIT_FAILED','generation_complete':True,'training_ready':not errors})
 print(json.dumps({k:v for k,v in report.items() if k not in ['scenario_coverage','normalized_observation_profile']},indent=2),flush=True)
 if errors:raise RuntimeError('Dataset audit failed; preserve and repair: '+str(sorted(set(errors))[:8]))
 return report
