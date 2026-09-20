"""Reproduce the retained-data support audit without changing dataset files."""
from pathlib import Path
import hashlib, json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def audit():
    result = {'scope': 'Split/event/bank support audit; not a replacement for exhaustive replay audit', 'splits': {}, 'banks': {}}
    for split in ('train', 'development', 'calibration', 'test'):
        folder = ROOT/'datasets/uphill_push_v1/cache'/split
        get = lambda key: np.load(folder/(key+'.npy'), mmap_mode='r')
        valid = get('valid'); events = get('events') & valid[..., None] & get('event_known')
        result['splits'][split] = dict(rows=len(valid), parents=len(np.unique(get('parent_episode'))), imposed_rows=int((get('source')==2).sum()), valid_transition_entries=int(valid.sum()), failure_windows=int(events[...,1:3].any((1,2)).sum()), event_positive_windows={name:int(events[...,i].any(1).sum()) for i,name in enumerate(('success','collision','fall','timeout','contact'))})
    for split in ('development', 'calibration', 'test'):
        path = ROOT/f'datasets/evaluation/truth/{split}_fixed16.npz'
        with np.load(path) as d:
            valid=d['valid']; events=d['events'] & valid[...,None]
            result['banks'][split] = dict(snapshots=len(valid), branches=int(np.prod(valid.shape[:2])), transitions=int(valid.sum()), event_positive_branches={name:int(events[...,i].any(-1).sum()) for i,name in enumerate(('success','collision','fall','timeout','contact'))})
        flags=[]
        for path in sorted((ROOT/'datasets/evaluation/public').glob(split+'_*.npz')):
            with np.load(path) as d: flags.extend(d['available_local_modes'].all(-1).tolist())
        result['banks'][split]['two_mode_flags_not_independent_certificates']=sum(flags)
    paths=['datasets/READINESS.json','datasets/uphill_push_v1/manifest.json','datasets/evaluation/manifest.json','datasets/uphill_push_v1/cache/normalization.json','environment/task.py']
    result['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    return result

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
