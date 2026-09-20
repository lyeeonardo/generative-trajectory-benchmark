"""Dataset schema, atomic metadata, training-only normalization, and batch costs."""
from pathlib import Path
import hashlib
import json
import numpy as np
from aif.operational_cost import batch_costs
ROOT = Path(__file__).resolve().parents[1]
FIELDS={
 'history_observations':((5,7),'float32'),'history_actions':((4,3),'float32'),
 'history_mask':((5,),'bool'),'geometry':((5,),'float32'),'tilt':((2,),'float32'),
 'time_fraction':((),'float32'),'future_observations':((6,7),'float32'),
 'actions':((6,3),'float32'),'valid':((6,),'bool'),'events':((6,5),'bool'),
 'event_known':((6,5),'bool'),'quality':((3,),'int8'),'proposal_eligible':((),'bool'),
 'source':((),'int8'),'tilt_id':((),'int16'),'stratum':((),'int8'),
 'base_id':((),'int32'),'anchor_id':((),'int64'),'source_ref':((),'int32'),
 'source_row':((),'int32'),'returns':((3,),'float32')}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()


def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def tilt_ids(tilts):
 degrees=np.rint(np.rad2deg(tilts)).astype(int)
 ids=((degrees[:,0]+10)//10)*4+degrees[:,1]//10
 if not np.isin(degrees[:,0],[-10,0,10]).all() or not np.isin(degrees[:,1],[0,20]).all():raise ValueError('Unknown represented training tilt')
 return ids.astype(np.int16)




def empty_batch(n):return {k:np.zeros((n,)+shape,dtype) for k,(shape,dtype) in FIELDS.items()}


def fit_normalization(train):
 a={k:np.load(train/(k+'.npy'),mmap_mode='r') for k in ('history_observations','future_observations','valid','geometry','tilt')}
 sums={k:np.zeros(d,np.float64) for k,d in [('observation',7),('delta',7),('geometry',5),('tilt',2)]}
 sq={k:np.zeros_like(v) for k,v in sums.items()};counts={k:0 for k in sums}
 for start in range(0,len(a['valid']),32768):
  sl=slice(start,start+32768);now=np.asarray(a['history_observations'][sl,-1],dtype=np.float64)
  delta=np.asarray(a['future_observations'][sl],dtype=np.float64)-now[:,None]
  delta[:,:,6]=np.arctan2(np.sin(delta[:,:,6]),np.cos(delta[:,:,6]))
  values={'observation':now,'delta':delta[a['valid'][sl]],'geometry':a['geometry'][sl],'tilt':a['tilt'][sl]}
  for k,v in values.items():
   v=np.asarray(v,np.float64);sums[k]+=v.sum(0);sq[k]+=(v*v).sum(0);counts[k]+=len(v)
 floors={'observation':[.02,.02,.05,.05,.02,.02,.1],'delta':[.01,.01,.05,.05,.01,.01,.05],'geometry':[.02]*5,'tilt':[.01]*2}
 result={'fit_split':'train','action_scale':[.8,.8,4.], 'target_encoding':'delta from current physical observation, wrapped yaw delta'}
 for k in sums:
  mean=sums[k]/counts[k];std=np.sqrt(np.maximum(sq[k]/counts[k]-mean*mean,0));std=np.maximum(std,floors[k])
  result[k]={'mean':mean.tolist(),'std':std.tolist(),'count':counts[k],'floor':floors[k]}
 return result
