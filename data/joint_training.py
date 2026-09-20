"""Public tensor encoding and coverage-balanced sampling for the joint world model."""
from pathlib import Path
import copy
import json
import numpy as np
import torch

def encode_batch(batch, normalization):
    """Never consumes returns, actual future validity or private state as context."""
    b=batch;device=b['actions'].device;dtype=torch.float32
    def stats(k):return (torch.tensor(normalization[k]['mean'],device=device,dtype=dtype),torch.tensor(normalization[k]['std'],device=device,dtype=dtype))
    om,os=stats('observation');dm,ds=stats('delta');gm,gs=stats('geometry');zm,zs=stats('tilt')
    scale=torch.tensor(normalization['action_scale'],device=device)
    hm=b['history_mask'].bool();ho=(b['history_observations'].float()-om)/os
    ho=ho*hm[...,None];ha=b['history_actions'].float()/scale
    ha=ha*hm[:,:-1,None]
    fixed=(b['role']==1)|(b['role']==2)
    g=(b['geometry'].float()-gm)/gs;g=g.clone();g[fixed,:2]=0
    quality=torch.where(fixed,0,b['quality'])
    context=torch.cat([ho.flatten(1),ha.flatten(1),hm.float(),g,(b['tilt']-zm)/zs,b['time_fraction'][:,None]],-1)
    now=b['history_observations'][:,-1];delta=b['future_observations']-now[:,None]
    delta=delta.clone();delta[...,6]=torch.atan2(torch.sin(delta[...,6]),torch.cos(delta[...,6]))
    n=len(now);x=torch.zeros(n,12,7,device=device)
    x[:,::2,:3]=b['actions']/scale;x[:,1::2]=(delta-dm)/ds
    semantic=torch.zeros_like(x,dtype=torch.bool);semantic[:,::2,:3]=True;semantic[:,1::2]=True
    active=torch.arange(6,device=device)[None,:]<b['horizon'][:,None]
    semantic &= active.repeat_interleave(2,dim=1)[...,None]
    known=torch.zeros_like(semantic);known[:,::2,:3]=(torch.arange(6,device=device)[None,:]<b['known_steps'][:,None])[...,None]
    for field, tokens, width in [('known_actions',slice(0,None,2),3),('known_observations',slice(1,None,2),7)]:
        if field in b:
            mask=b[field].bool()
            if mask.shape!=(n,6,width):raise ValueError('Invalid known-entry mask: '+field)
            if (mask & ~active[...,None]).any():raise ValueError('Known entries outside requested horizon')
            if field=='known_observations' and (mask.any((1,2)) & (b['role']!=3)).any():
                raise ValueError('Future observations are only allowed for completion')
            known[:,tokens,:width] |= mask
    known &= semantic
    loss_mask=semantic & b['valid'].repeat_interleave(2,1)[...,None] & ~known
    # Future labels and terminal padding affect losses only, never conditioning.
    return {'x':x,'context':context,'quality':quality,'role':b['role'],'horizon':b['horizon'],
            'known':known,'semantic':semantic,'loss_mask':loss_mask,
            'events':b['events'].float(),'event_mask':b['event_known'] & active[...,None] & b['valid'][...,None]}

MODEL_FIELDS=('history_observations','history_actions','history_mask','geometry','tilt',
              'time_fraction','future_observations','actions','valid','events','event_known')

def shift_measured_windows(batch,offsets):
    """Move measured branch prefixes into history and retain measured suffixes.

    This is a TRAIN-only augmentation helper. It never inserts predicted states.
    The caller must choose an offset from 1..3 that is smaller than the row's
    valid target length. Offset zero leaves a row unchanged.
    """
    offsets=torch.as_tensor(offsets,device=batch['actions'].device,dtype=torch.long)
    if offsets.shape!=(len(batch['actions']),) or (offsets<0).any() or (offsets>3).any():
        raise ValueError('Branch-shift offsets must have shape [batch] and values 0..3')
    out={k:v.clone() for k,v in batch.items()}
    for offset in (1,2,3):
        ids=torch.nonzero(offsets==offset,as_tuple=True)[0]
        if not len(ids):continue
        if not batch['valid'][ids,offset].all():
            raise ValueError('Branch shift must leave at least one measured target')
        observations=torch.cat((batch['history_observations'][ids],batch['future_observations'][ids,:offset]),1)
        actions=torch.cat((batch['history_actions'][ids],batch['actions'][ids,:offset]),1)
        masks=torch.cat((batch['history_mask'][ids],torch.ones((len(ids),offset),dtype=torch.bool,device=offsets.device)),1)
        out['history_observations'][ids]=observations[:,-5:]
        out['history_actions'][ids]=actions[:,-4:]
        out['history_mask'][ids]=masks[:,-5:]
        out['time_fraction'][ids]=batch['time_fraction'][ids]+offset/100.
        for key in ('future_observations','actions','valid','events','event_known'):
            value=batch[key][ids]
            out[key][ids]=torch.cat((value[:,offset:],torch.zeros_like(value[:,:offset])),1)
    out['branch_shift_offset']=offsets
    return out

class JointTrainingData:
    """TRAIN-only condition/phase balancing with an explicit demonstration mixture.

    Stored quality labels remain immutable. Half of HIGH proposal/completion draws
    request successful reference demonstrations as a declared additional HIGH source.
    Terminal windows remain short; no fictitious post-terminal targets are added.
    """
    def __init__(self,cache,device='cpu',seed=13,resident=True,high_success_demo_fraction=.5,
                 branch_shift_fraction=0.,membership='all',canonical_condition=None):
        self.root=Path(cache);self.device=torch.device(device);self.rng=np.random.default_rng(seed)
        self.normalization=json.loads((self.root/'normalization.json').read_text())
        if self.normalization.get('fit_split')!='train':raise ValueError('TRAIN-only normalization required')
        if not 0<=high_success_demo_fraction<=1:raise ValueError('Invalid demonstration fraction')
        if not 0<=branch_shift_fraction<=1:raise ValueError('Invalid branch-shift fraction')
        self.demo_fraction=high_success_demo_fraction
        self.branch_shift_fraction=float(branch_shift_fraction)
        if membership not in ('all','canonical'):raise ValueError('Training membership must be all or canonical')
        self.membership=membership
        self.canonical_condition=copy.deepcopy(canonical_condition or {'obstacle_x':0.,'lateral_degrees':0.,'longitudinal_degrees':20.})
        names=set(MODEL_FIELDS)|{'source','quality','proposal_eligible','tilt_id','stratum','anchor_id','scenario_id','base_id','parent_episode'}
        self.raw={k:np.load(self.root/'train'/(k+'.npy'),mmap_mode='r') for k in names}
        if membership=='canonical':
            c=self.canonical_condition;geometry=self.raw['geometry'];tilt=self.raw['tilt']
            selected=(np.isclose(geometry[:,2],float(c['obstacle_x']),atol=1e-7,rtol=0)&
                      np.isclose(tilt[:,0],np.deg2rad(float(c['lateral_degrees'])),atol=1e-7,rtol=0)&
                      np.isclose(tilt[:,1],np.deg2rad(float(c['longitudinal_degrees'])),atol=1e-7,rtol=0))
        else:selected=np.ones(len(self.raw['scenario_id']),bool)
        self.allowed_rows=np.flatnonzero(selected)
        if not len(self.allowed_rows):raise ValueError('Training membership selected no rows')
        self.scenario_ids=np.unique(self.raw['scenario_id'][self.allowed_rows]);self.count=len(self.scenario_ids)
        expected=1 if membership=='canonical' else 9
        if self.count!=expected:raise ValueError(f'Expected {expected} physical training scenario(s), found {self.count}')
        if membership=='all' and len(np.unique(self.raw['tilt_id'][self.allowed_rows]))!=3:raise ValueError('Expected retained three-tilt dataset')
        self.tensor={k:torch.tensor(np.asarray(self.raw[k]),device=self.device if resident else 'cpu') for k in MODEL_FIELDS}
        now=self.raw['history_observations'][:,-1];distance=np.linalg.norm(now[:,:2]-self.raw['geometry'][:,:2],axis=1);speed=np.linalg.norm(now[:,2:4],axis=1)
        self.phase=np.where(distance>.25,0,np.where(distance>.12,1,np.where(speed>=.1,2,3)))
        self.rows_by_condition={int(s):self.allowed_rows[self.raw['scenario_id'][self.allowed_rows]==s] for s in self.scenario_ids}
        self.pools={};self.windows=0;self.visits=np.zeros((self.count,4),np.int64);self.high=np.zeros(self.count,np.int64)
        self.permutations=[self.rng.permutation(self.count) for _ in range(4)];self.positions=np.zeros(4,int)
        self.seen=np.zeros(len(now),bool)
        self.exposure=dict(windows=0,roles=[0]*4,sources=[0]*3,requested_quality=[0]*3,applied_quality=[0]*3,
            phases=[0]*4,role_phase_counts=np.zeros((4,4),int).tolist(),condition_phase_counts=np.zeros((self.count,4),int).tolist(),
            horizons={str(h):0 for h in (1,3,6)},phase_fallback_draws=0,quality_fallback_draws=0,
            high_reference_draws=0,shifted_branch_HIGH_draws=0,branch_shift_offsets=[0]*4,
            branch_shift_fraction=self.branch_shift_fraction,training_membership=self.membership,eligible_training_windows=len(self.allowed_rows),
            physical_scenario_ids=self.scenario_ids.tolist(),short_terminal_draws=0,completion_masks=[0]*4,draw_cells={})
    def scenarios(self,role,n):
        out=[]
        while n:
            if self.positions[role]==self.count:self.permutations[role]=self.rng.permutation(self.count);self.positions[role]=0
            take=min(n,self.count-self.positions[role]);i=self.positions[role]
            out.extend(self.permutations[role][i:i+take]);self.positions[role]+=take;n-=take
        return np.asarray(out,int)
    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.bit_generator.state,exposure=self.exposure,seen=self.seen,windows=self.windows,
            permutations=self.permutations,positions=self.positions,scenario_role_counts=self.visits,scenario_HIGH_counts=self.high,
            scenario_ids=self.scenario_ids,demo_fraction=self.demo_fraction,branch_shift_fraction=self.branch_shift_fraction,
            membership=self.membership,allowed_rows=self.allowed_rows,canonical_condition=self.canonical_condition))
    def load_state_dict(self,s):
        if (not np.array_equal(s['scenario_ids'],self.scenario_ids) or s['demo_fraction']!=self.demo_fraction
                or s.get('branch_shift_fraction',0.)!=self.branch_shift_fraction or s.get('membership','all')!=self.membership
                or not np.array_equal(s.get('allowed_rows'),self.allowed_rows) or s.get('canonical_condition')!=self.canonical_condition):
            raise ValueError('Sampler contract changed')
        s=copy.deepcopy(s);self.rng.bit_generator.state=s['rng'];self.exposure=s['exposure'];self.seen=s['seen'];self.windows=s['windows']
        self.permutations=s['permutations'];self.positions=s['positions'];self.visits=s['scenario_role_counts'];self.high=s['scenario_HIGH_counts']
    def pool(self,sid,role,phase,q,h,source):
        key=(sid,role,phase,q,h,source)
        if key in self.pools:return self.pools[key]
        available=self.rows_by_condition[int(self.scenario_ids[sid])]
        r=self.raw;mask=np.ones(len(available),bool);applied=q
        if role in (0,3):
            mask &= r['proposal_eligible'][available]
            if source==1:mask &= r['source'][available]==1
            elif source==3:
                mask &= (r['source'][available]==0)&(r['quality'][available,[1,3,6].index(h)]==q)&(r['valid'][available].sum(1)>=2)
            else:mask &= r['quality'][available,[1,3,6].index(h)]==q
        else:mask &= (r['source'][available]==2) if source==2 else r['proposal_eligible'][available]
        rows=available[mask];quality_fallback=False
        if not len(rows):
            if source==3:raise ValueError('No safe measured HIGH branch supports the requested condition/horizon')
            # Never label a missing LOW/HIGH cell as though it existed.
            rows=available[r['proposal_eligible'][available]];applied=0;quality_fallback=True
        subset=rows[self.phase[rows]==phase];phase_fallback=not len(subset)
        if len(subset):rows=subset
        if not len(rows):raise ValueError('Empty training support')
        self.pools[key]=(rows,applied,phase_fallback,quality_fallback)
        return self.pools[key]
    def sample(self,n):
        role=np.repeat(np.arange(4),[9,5,4,2])[(np.arange(n)+self.windows)%20]
        sid=np.empty(n,int);visit=np.empty(n,int)
        for r in range(4):
            ix=np.flatnonzero(role==r);sid[ix]=self.scenarios(r,len(ix))
            for j in ix:visit[j]=self.visits[sid[j],r];self.visits[sid[j],r]+=1
        # Independent radix cycles cross every quality, phase and horizon per condition/role.
        q=visit%3;fixed=(role==1)|(role==2);q[fixed]=0
        phase=(visit//3)%4;h=np.array([1,3,6,6])[(visit//12)%4];h[role==1]=1;h[role==3]=6
        h[role==2]=np.where((visit[role==2]//12)%2==0,3,6)
        source=np.where(fixed,np.where(visit%2==0,2,0),0)
        shifted=(~fixed)&(q==2)&(self.rng.random(n)<self.branch_shift_fraction);source[shifted]=3
        demo=(~fixed)&(~shifted)&(q==2)&(self.rng.random(n)<self.demo_fraction);source[demo]=1
        requested=q.copy();index=np.empty(n,int)
        keys=np.stack([sid,role,phase,q,h,source],1)
        unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        for i,key in enumerate(unique):
            ix=np.flatnonzero(inverse==i);rows,applied,pf,qf=self.pool(*map(int,key))
            index[ix]=self.rng.choice(rows,len(ix));q[ix]=applied
            self.exposure['phase_fallback_draws']+=int(pf)*len(ix);self.exposure['quality_fallback_draws']+=int(qf)*len(ix)
        q[(q>0)&(self.rng.random(n)<.2)]=0 # Declared classifier-free quality dropout.
        ix=torch.as_tensor(index,device=next(iter(self.tensor.values())).device)
        b={k:v[ix].to(self.device) for k,v in self.tensor.items()}
        offsets=np.zeros(n,np.int64)
        if shifted.any():
            valid_length=self.raw['valid'][index[shifted]].sum(1)
            limit=np.minimum(3,valid_length-1)
            offsets[shifted]=1+(self.rng.random(int(shifted.sum()))*limit).astype(np.int64)
            b=shift_measured_windows(b,torch.as_tensor(offsets,device=self.device))
        else:b['branch_shift_offset']=torch.zeros(n,dtype=torch.long,device=self.device)
        valid_length=self.raw['valid'][index].sum(1)-offsets
        known=np.where(fixed,h,0)
        for k,v in [('horizon',h),('quality',q),('role',role),('known_steps',known)]:b[k]=torch.as_tensor(v,device=self.device)
        ka=np.zeros((n,6,3),bool);ko=np.zeros((n,6,7),bool)
        for j in np.flatnonzero(role==3):
            length=min(int(valid_length[j]),6);kind=(visit[j]//48)%4
            if kind==0:ko[j,length-1,:2]=True
            elif kind==1:ko[j,min(2,length-1),:2]=True
            elif kind==2:
                ko[j,min(2,length-1),:2]=True;ka[j,:min(2,length-1)]=True
            else:ka[j,0,0]=True # Partial action coordinate; other actions remain unknown.
            self.exposure['completion_masks'][kind]+=1
        b['known_actions']=torch.tensor(ka,device=self.device);b['known_observations']=torch.tensor(ko,device=self.device)
        self.seen[index]=True;self.windows+=n;np.add.at(self.high,sid[(role==0)&(q==2)],1)
        e=self.exposure;e['windows']=self.windows
        actual_phase=self.phase[index].copy()
        if shifted.any():
            current=self.raw['future_observations'][index[shifted],offsets[shifted]-1]
            distance=np.linalg.norm(current[:,:2]-self.raw['geometry'][index[shifted],:2],axis=1);speed=np.linalg.norm(current[:,2:4],axis=1)
            actual_phase[shifted]=np.where(distance>.25,0,np.where(distance>.12,1,np.where(speed>=.1,2,3)))
        for k,v,size in [('roles',role,4),('sources',self.raw['source'][index],3),('requested_quality',requested,3),('applied_quality',q,3),('phases',actual_phase,4)]:e[k]=(np.asarray(e[k])+np.bincount(v,minlength=size)).tolist()
        rp=np.asarray(e['role_phase_counts']);np.add.at(rp,(role,actual_phase),1);e['role_phase_counts']=rp.tolist()
        cp=np.asarray(e['condition_phase_counts']);np.add.at(cp,(sid,actual_phase),1);e['condition_phase_counts']=cp.tolist()
        cells,counts=np.unique(np.stack([sid,role,actual_phase,self.raw['source'][index],requested,q],1),axis=0,return_counts=True)
        for cell,count in zip(cells,counts):
            key='/'.join(map(str,cell));e['draw_cells'][key]=e['draw_cells'].get(key,0)+int(count)
        e['high_reference_draws']+=int(((q==2)&(self.raw['source'][index]==1)&(~fixed)).sum())
        e['shifted_branch_HIGH_draws']+=int(shifted.sum());e['branch_shift_offsets']=(np.asarray(e['branch_shift_offsets'])+np.bincount(offsets,minlength=4)).tolist()
        e['short_terminal_draws']+=int((valid_length<h).sum())
        for hh in (1,3,6):e['horizons'][str(hh)]+=int((h==hh).sum())
        e.update(covered_scenarios_by_role=(self.visits>0).sum(0).tolist(),minimum_draws_per_scenario_by_role=self.visits.min(0).tolist(),HIGH_applied_scenarios=int((self.high>0).sum()),minimum_applied_HIGH_draws=int(self.high.min()))
        return b
