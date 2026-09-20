#!/usr/bin/env python
"""Verify the retained uphill dataset, evaluation banks, and split boundaries."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from data.preparation import sha,fit_normalization,write_json
from scripts.audit_study_data import audit as support_audit


def check_hashes(base,entries):
    checked=0
    for relative,digest in entries.items():
        path=base/relative
        if not path.is_file() or sha(path)!=digest:raise ValueError('Artifact mismatch: '+str(path))
        checked+=1
    return checked


def check():
    data=ROOT/'datasets/uphill_push_v1';cache=data/'cache'
    audit=json.loads((data/'audit.json').read_text());ready=json.loads((ROOT/'datasets/READINESS.json').read_text())
    if audit['status']!='PASS' or ready['status']!='PASS':raise ValueError('Retained data audit not passing')
    if ready['dataset_manifest_sha']!=sha(data/'manifest.json'):raise ValueError('Readiness/manifest identity mismatch')
    manifest=json.loads((data/'manifest.json').read_text())
    checked=check_hashes(data,manifest['artifact_hashes'])
    checked+=check_hashes(cache,json.loads((cache/'hashes.json').read_text()))
    if fit_normalization(cache/'train')!=json.loads((cache/'normalization.json').read_text()):raise ValueError('TRAIN normalization changed')
    parents={split:set(np.load(cache/split/'parent_episode.npy',mmap_mode='r').tolist()) for split in ('train','development','calibration','test')}
    if any(parents[a]&parents[b] for i,a in enumerate(parents) for b in list(parents)[i+1:]):raise ValueError('Parent split leakage')
    evaluation=json.loads((ROOT/'datasets/evaluation/manifest.json').read_text())
    if evaluation['status']!='COMPLETE' or evaluation['dataset_manifest_sha256']!=sha(data/'manifest.json'):raise ValueError('Evaluation bank identity mismatch')
    checked+=check_hashes(ROOT/'datasets/evaluation',evaluation['artifacts'])
    support=support_audit();counts={k:v['rows'] for k,v in support['splits'].items()}
    expected={'train':53705,'development':6522,'calibration':6666,'test':13212}
    if counts!=expected:raise ValueError('Retained split counts changed')
    scenario=np.load(cache/'train/scenario_id.npy',mmap_mode='r');geometry=np.load(cache/'train/geometry.npy',mmap_mode='r');tilt=np.load(cache/'train/tilt.npy',mmap_mode='r')
    canonical=np.isclose(geometry[:,2],0,atol=1e-7,rtol=0)&np.isclose(tilt[:,0],0,atol=1e-7,rtol=0)&np.isclose(tilt[:,1],np.deg2rad(20),atol=1e-7,rtol=0)
    canonical_parents=np.unique(np.load(cache/'train/parent_episode.npy',mmap_mode='r')[canonical])
    if int(canonical.sum())!=5879 or np.unique(scenario[canonical]).tolist()!=[4] or len(canonical_parents)!=16:raise ValueError('Canonical physical membership changed')
    return {'status':'PASS','hashed_artifacts':checked,'dataset_manifest_sha256':sha(data/'manifest.json'),'normalization':'TRAIN_ONLY',
            'split_rows':counts,'parent_split_overlap':0,'canonical_train_rows':int(canonical.sum()),'canonical_parent_episodes':len(canonical_parents),
            'evaluation_roles':evaluation['roles'],'evaluation_fixed_branches':support['banks'],'event_support':{k:v['event_positive_windows'] for k,v in support['splits'].items()}}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args();result=check()
    if args.output:write_json(args.output,result)
    print(json.dumps(result,indent=2))
