"""One causal trajectory backbone/checkpoint for joint and imposed-action queries.

No simulator imports. The diffusion/flow conditional queries are trained masked
approximations, not claims of exact conditioning of a global joint density.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from data.joint_training import encode_batch

METHODS=('diffusion','flow_matching','autoregressive')
@dataclass
class JointConfig:
    method: str='diffusion'
    width: int=256
    layers: int=6
    heads: int=8
    mixtures: int=8
    sampling_steps: int=8
    first_action_loss_weight: float=10.0
    query_loss_weights: tuple=(.45,.25,.20,.10)
    conditioning_mode: str='add'

class CausalBlock(nn.Module):
    def __init__(self,width,heads,adaptive_conditioning=False):
        super().__init__();self.heads=heads
        self.n1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width)
        self.n2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.modulation=(nn.Sequential(nn.SiLU(),nn.Linear(width,4*width))
                         if adaptive_conditioning else None)
    def forward(self,x,condition=None):
        if self.modulation is None:
            first=self.n1(x);second=lambda value:self.n2(value)
        else:
            if condition is None:raise ValueError('Adaptive conditioning requires a context vector')
            shift1,scale1,shift2,scale2=self.modulation(condition).chunk(4,-1)
            first=self.n1(x)*(1+scale1[:,None])+shift1[:,None]
            second=lambda value:self.n2(value)*(1+scale2[:,None])+shift2[:,None]
        n,t,d=x.shape;q,k,v=self.qkv(first).reshape(n,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        att=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.proj(att.transpose(1,2).reshape(n,t,d));return x+self.ff(second(x))

def reduce_query_losses(token_loss,e,event_logits,*,first_action_weight=10.,query_weights=(.45,.25,.20,.10)):
    """Normalize coordinates within each sample/query, then apply fixed role weights.

    Action and observation groups each have unit scale when present. A0 has 10x
    the weight of later action coordinates for planning; clamped coordinates and
    padding have zero weight. H1 never gains arbitrary scale from its short length.
    """
    counts=e['loss_mask'].sum(-1).to(token_loss.dtype)
    aw=counts[:,::2].clone()
    first=torch.where(e['role']==0,first_action_weight,1.).to(token_loss.dtype)
    aw[:,0]*=first
    ow=counts[:,1::2]
    action=(token_loss[:,::2]*aw).sum(-1)/aw.sum(-1).clamp_min(1)
    observation=(token_loss[:,1::2]*ow).sum(-1)/ow.sum(-1).clamp_min(1)
    event_mask=e['event_mask'].to(token_loss.dtype)
    event=(F.binary_cross_entropy_with_logits(event_logits,e['events'],reduction='none')*event_mask).sum((1,2))/event_mask.sum((1,2)).clamp_min(1)
    means=[];metrics={}
    for role,weight in enumerate(query_weights):
        mask=(e['role']==role).to(token_loss.dtype);den=mask.sum().clamp_min(1)
        vals=[(value*mask).sum()/den for value in (action,observation,event)]
        means.append(vals)
        metrics[f'role_{role}_loss']=(vals[0]+vals[1]+.1*vals[2]).detach()
    a=sum(w*v[0] for w,v in zip(query_weights,means))
    o=sum(w*v[1] for w,v in zip(query_weights,means))
    ev=sum(w*v[2] for w,v in zip(query_weights,means))
    metrics.update(action_loss=a.detach(),observation_loss=o.detach(),event_loss=ev.detach())
    return a+o+.1*ev,metrics


class JointWorldModel(nn.Module):
    def __init__(self,config:JointConfig,normalization):
        super().__init__();self.config=config;self.normalization=normalization
        if config.method not in METHODS:raise ValueError(config.method)
        if not math.isfinite(config.first_action_loss_weight) or config.first_action_loss_weight<1:raise ValueError('Invalid first-action weight')
        if len(config.query_loss_weights)!=4 or any(not math.isfinite(w) or w<0 for w in config.query_loss_weights) or not math.isclose(sum(config.query_loss_weights),1.):raise ValueError('Four query-loss weights must sum to one')
        if config.width<1 or config.layers<1 or config.heads<1 or config.width%config.heads or config.sampling_steps<1 or config.mixtures<1:raise ValueError('Invalid model dimensions or sampling settings')
        if config.conditioning_mode not in ('add','adaptive'):raise ValueError('conditioning_mode must be add or adaptive')
        self.sampling_temperature=1.0;self.guidance_scale=1.0
        d=config.width;self.input=nn.Linear(21,d);self.context=nn.Linear(60,d)
        self.constraints=nn.Linear(168,d,bias=False)
        self.quality=nn.Embedding(3,d);self.role=nn.Embedding(4,d);self.horizon=nn.Embedding(7,d)
        self.position=nn.Parameter(torch.randn(1,12,d)*.01)
        self.time=nn.Sequential(nn.Linear(4,d),nn.SiLU(),nn.Linear(d,d))
        adaptive=config.conditioning_mode=='adaptive'
        self.blocks=nn.ModuleList(CausalBlock(d,config.heads,adaptive) for _ in range(config.layers));self.norm=nn.LayerNorm(d)
        self.output=nn.Linear(d,7 if not self.is_ar else config.mixtures*15)
        self.event=nn.Linear(d,5)
    @property
    def is_ar(self):return self.config.method=='autoregressive'
    def features(self,x,e,t):
        known_x=torch.where(e['known'],e['x'],0.)
        if self.is_ar:
            # Teacher/sample prefix shifted: token j never sees its own target.
            x=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],1)
        inp=torch.cat([x,known_x,e['known'].float()],-1)
        temb=torch.stack([t,torch.sin(math.pi*t),torch.cos(math.pi*t),t*t],-1)
        c=self.context(e['context'])+self.quality(e['quality'])+self.role(e['role'])+self.horizon(e['horizon'])+self.time(temb)
        c=c+self.completion_context(e)
        h=self.input(inp)+self.position+c[:,None]
        for block in self.blocks:h=block(h,c)
        return self.norm(h)
    def inference_work(self,query,H=6):
        if query not in ('proposal','prediction','completion'):raise ValueError('Unknown query role')
        method=self.config.method;extra=query!='prediction'
        if method=='autoregressive':primary=2*H+1
        elif method=='flow_matching':primary=2*self.config.sampling_steps+1
        else:primary=self.config.sampling_steps+1
        if method=='diffusion' and extra and self.guidance_scale!=1.:primary+=self.config.sampling_steps
        # Proposal/completion re-predict returned bounded actions with NULL quality,
        # so that second query never pays classifier-free guidance work.
        physical=(self.config.sampling_steps+1 if method=='diffusion' else 2*self.config.sampling_steps+1 if method=='flow_matching' else 2*H+1) if extra else 0
        return dict(backbone_evaluations=primary+physical,imposed_action_reprediction_queries=int(extra),sampling_temperature=self.sampling_temperature,guidance_scale=self.guidance_scale if method=='diffusion' else 1.)

    def completion_context(self,e):
        # Only declared known constraints; never unknown future targets. Fixed-action
        # physical queries keep their original causal attention and see no suffix summary.
        values=torch.where(e['known'],e['x'],0.)
        packed=torch.cat([values.flatten(1),e['known'].float().flatten(1)],-1)
        return self.constraints(packed)*(e['role']==3)[:,None]

    def constrain_actions(self,x,e):
        value=x[:,::2,:3].clone();known=e['known'][:,::2,:3];target=e['x'][:,::2,:3]
        value=torch.where(known,target,value)
        fixed_xy=torch.where(known[...,:2],value[...,:2],0.)
        free_xy=torch.where(known[...,:2],0.,value[...,:2])
        allowance=(1-fixed_xy.square().sum(-1,keepdim=True)).clamp_min(0).sqrt()
        free_xy=free_xy*torch.minimum(torch.ones_like(allowance),allowance/torch.linalg.vector_norm(free_xy,dim=-1,keepdim=True).clamp_min(1e-12))
        value[...,:2]=fixed_xy+free_xy
        value[...,2]=torch.where(known[...,2],target[...,2],value[...,2].clamp(-1,1))
        x=x.clone();x[:,::2,:3]=value
        return torch.where(e['known'],e['x'],x)*e['semantic']

    def consistent_output(self,batch,encoded,actions,observations,events,generator):
        """Re-predict the actual bounded actions, using this same checkpoint.

        joint_observations retains the raw conditional completion (including hard
        waypoints). observations is an independent physical forecast of the actions
        actually returned. The extra model query must be charged to proposal timing.
        """
        action_known=encoded['known'][:,::2,:3]
        actions=torch.where(action_known,batch['actions'],actions)
        result={'actions':actions,'observations':observations.clone(),
                'joint_observations':observations,'event_probabilities':events.sigmoid()}
        need=((~action_known)&encoded['semantic'][:,::2,:3]).any((1,2)) | (batch['role']==3)
        if need.any():
            fixed={k:v[need].clone() for k,v in batch.items() if torch.is_tensor(v)}
            fixed.pop('known_actions',None);fixed.pop('known_observations',None)
            fixed['actions']=actions[need];fixed['known_steps']=fixed['horizon'].clone()
            fixed['role']=torch.where(fixed['horizon']==1,1,2);fixed['quality']=torch.zeros_like(fixed['role'])
            # Unknown targets never become conditioning in imposed-action mode.
            prediction=self.generate(fixed,generator)
            result['observations'][need]=prediction['observations']
            result['event_probabilities'][need]=prediction['event_probabilities']
        return result

    def forward(self,x,e,t):
        h=self.features(x,e,t);return self.output(h),self.event(h[:,1::2])
    def denoise(self,x,e,t):
        conditional,events=self(x,e,t);scale=float(self.guidance_scale)
        if not math.isfinite(scale) or scale<1:raise ValueError('Guidance scale must be finite and at least one')
        guided=e['quality']>0
        if scale!=1. and bool(guided.any()):
            null={**e,'quality':torch.zeros_like(e['quality'])};unconditional,_=self(x,null,t)
            value=unconditional+scale*(conditional-unconditional)
            conditional=torch.where(guided[:,None,None],value,conditional)
        return conditional,events
    def distribution(self,out):
        n,t,_=out.shape;k=self.config.mixtures;v=out.reshape(n,t,k,15)
        return v[...,0],v[...,1:8],v[...,8:15].clamp(-5,2)
    def loss(self,batch):
        e=encode_batch(batch,self.normalization);x=e['x'];n=len(x)
        if self.is_ar:
            out,events=self(x,e,torch.zeros(n,device=x.device))
            logits,mean,logs=self.distribution(out);mask=e['loss_mask'][:,:,None].float()
            component=(-.5*((x[:,:,None]-mean)*torch.exp(-logs)).square()-logs-.5*math.log(2*math.pi))*mask
            lp=torch.logsumexp(F.log_softmax(logits,-1)+component.sum(-1),-1)
            count=mask.sum((-1,-2));token_loss=(-lp/count.clamp_min(1))*(count>0)
        else:
            t=torch.rand(n,device=x.device)*.998+.001;noise=torch.randn_like(x)
            if self.config.method=='diffusion':
                alpha=torch.cos(t*math.pi/2)[:,None,None];sigma=torch.sin(t*math.pi/2)[:,None,None]
                xt=alpha*x+sigma*noise;target=x
            else:
                tt=t[:,None,None];xt=(1-tt)*noise+tt*x;target=x-noise
            xt=torch.where(e['known'],x,xt)*e['semantic'];out,events=self(xt,e,t)
            squared=(out-target).square()*e['loss_mask'];count=e['loss_mask'].sum(-1)
            token_loss=squared.sum(-1)/count.clamp_min(1)
        return reduce_query_losses(token_loss,e,events,first_action_weight=self.config.first_action_loss_weight,query_weights=self.config.query_loss_weights)
    @torch.no_grad()
    def generate(self,batch,generator=None):
        e=encode_batch(batch,self.normalization);shape=e['x'].shape;device=e['x'].device
        temperature=float(self.sampling_temperature)
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('Sampling temperature must be finite and positive')
        x=torch.randn(shape,device=device,generator=generator)*temperature*e['semantic']
        x=torch.where(e['known'],e['x'],x);n=len(x)
        if self.is_ar:
            x=torch.zeros_like(x)
            for j in range(int(batch['horizon'].max())*2):
                out,events=self(x,e,torch.zeros(n,device=device))
                logits,mean,logs=self.distribution(out[:,j:j+1]);p=(logits[:,0]/temperature).softmax(-1)
                comp=torch.multinomial(p,1,generator=generator)[:,0];idx=torch.arange(n,device=device)
                value=mean[idx,0,comp]+temperature*logs[idx,0,comp].exp()*torch.randn(n,7,device=device,generator=generator)
                if j%2==0:value[:,:3]=self.project_action(value[:,:3])
                x[:,j]=torch.where(e['known'][:,j],e['x'][:,j],value)*e['semantic'][:,j]
                if j%2==0:x=self.constrain_actions(x,e)
        else:
            steps=self.config.sampling_steps
            grid=torch.linspace(.999,0,steps+1,device=device) if self.config.method=='diffusion' else torch.linspace(0,1,steps+1,device=device)
            for i in range(steps):
                t=grid[i].expand(n);next_t=grid[i+1]
                pred,_=self.denoise(x,e,t) if self.config.method=='diffusion' else self(x,e,t)
                if self.config.method=='diffusion':
                    a=torch.cos(t[0]*math.pi/2);s=torch.sin(t[0]*math.pi/2)
                    eps=(x-a*pred)/s.clamp_min(1e-5)
                    nxt=torch.cos(next_t*math.pi/2)*pred+torch.sin(next_t*math.pi/2)*eps
                else:
                    dt=next_t-t[0];trial=torch.where(e['known'],e['x'],x+dt*pred)*e['semantic']
                    pred2,_=self(trial,e,next_t.expand(n));nxt=x+dt*.5*(pred+pred2)
                x=torch.where(e['known'],e['x'],nxt)*e['semantic']
            x=self.constrain_actions(x,e)
        _,events=self(x,e,torch.zeros(n,device=device) if self.config.method!='flow_matching' else torch.ones(n,device=device))
        dm=torch.tensor(self.normalization['delta']['mean'],device=device);ds=torch.tensor(self.normalization['delta']['std'],device=device)
        scale=torch.tensor(self.normalization['action_scale'],device=device)
        observations=x[:,1::2]*ds+dm+batch['history_observations'][:,-1,None]
        observations[...,6]=torch.atan2(torch.sin(observations[...,6]),torch.cos(observations[...,6]))
        actions=x[:,::2,:3]*scale
        return self.consistent_output(batch,e,actions,observations,events,generator)
    @staticmethod
    def project_action(x):
        x=x.clone();x[...,:2]/=torch.linalg.vector_norm(x[...,:2],dim=-1,keepdim=True).clamp_min(1);x[...,2].clamp_(-1,1);return x
    def _public_batch(self,context,horizon,quality,actions=None):
        if set(context)=={'observations','actions','history_mask','geometry','tilt','time_fraction'}:
            context={('history_observations' if k=='observations' else 'history_actions' if k=='actions' else k):v for k,v in context.items()}
        required={'history_observations','history_actions','history_mask','geometry','tilt','time_fraction'}
        if set(context)!=required:raise ValueError('Only the public context whitelist is accepted')
        if horizon not in (1,3,6):raise ValueError('Only trained horizons H1/H3/H6 are supported')
        device=next(self.parameters()).device
        b={k:torch.as_tensor(v,device=device) for k,v in context.items()}
        single=b['tilt'].ndim==1
        if single:b={k:v.unsqueeze(0) for k,v in b.items()}
        n=len(b['tilt'])
        expected={'history_observations':(n,5,7),'history_actions':(n,4,3),'history_mask':(n,5),'geometry':(n,5),'tilt':(n,2),'time_fraction':(n,)}
        for k,shape in expected.items():
            if tuple(b[k].shape)!=shape or not torch.isfinite(b[k]).all():raise ValueError('Invalid public field: '+k)
        if not b['history_mask'][:,-1].all() or (b['time_fraction']<0).any() or (b['time_fraction']>=1).any():raise ValueError('Invalid current history/time')
        if quality not in (0,1,2):raise ValueError('Quality must be NULL=0, LOW=1, HIGH=2')
        for k in ('history_observations','history_actions','geometry','tilt','time_fraction'):b[k]=b[k].float()
        fixed=actions is not None;b['horizon']=torch.full((n,),horizon,device=device,dtype=torch.long)
        b['quality']=torch.full((n,),0 if fixed else quality,device=device,dtype=torch.long)
        b['role']=torch.full((n,),1 if fixed and horizon==1 else 2 if fixed else 0,device=device,dtype=torch.long)
        b['known_steps']=torch.full((n,),horizon if fixed else 0,device=device,dtype=torch.long)
        b['actions']=torch.zeros(n,6,3,device=device)
        if fixed:
            actions=torch.as_tensor(actions,device=device,dtype=torch.float32)
            if actions.ndim==2:actions=actions.unsqueeze(0)
            if actions.shape!=(n,horizon,3) or not torch.isfinite(actions).all():raise ValueError('Invalid fixed action shape/values')
            if (torch.linalg.vector_norm(actions[...,:2],dim=-1)>.800001).any() or (actions[...,2].abs()>4.000001).any():raise ValueError('Supply actual bounded executed actions')
            b['actions'][:,:horizon]=actions
        b['future_observations']=torch.zeros(n,6,7,device=device);b['valid']=torch.ones(n,6,device=device,dtype=torch.bool)
        b['events']=torch.zeros(n,6,5,device=device);b['event_known']=torch.zeros(n,6,5,device=device,dtype=torch.bool)
        return b
    @torch.no_grad()
    def propose_joint(self,context,K=1,quality=2,H=6,generator=None):
        if not isinstance(K,int) or not 1<=K<=32:raise ValueError('K must be an integer in [1,32]')
        self.eval();b=self._public_batch(context,H,quality)
        b={k:v.repeat_interleave(K,0) for k,v in b.items()};out=self.generate(b,generator)
        return {k:v[:,:H].reshape(-1,K,H,v.shape[-1]) for k,v in out.items()}
    @torch.no_grad()
    def predict(self,context,actions,samples=16,generator=None):
        if samples<1:raise ValueError('Prediction sample count must be positive')
        self.eval();actions=torch.as_tensor(actions);H=actions.shape[-2];b=self._public_batch(context,H,0,actions)
        b={k:v.repeat_interleave(samples,0) for k,v in b.items()};out=self.generate(b,generator)
        return {k:v[:,:H].reshape(-1,samples,H,v.shape[-1]) for k,v in out.items()}

    @torch.no_grad()
    def complete_joint(self,context,actions,observations,known_actions,known_observations,K=1,H=6,quality=2,generator=None):
        """Complete explicit physical-unit constraints; return raw and physical predictions."""
        if not isinstance(K,int) or not 1<=K<=32:raise ValueError('K must be 1..32')
        self.eval();b=self._public_batch(context,H,quality);device=b['actions'].device;n=len(b['actions'])
        for name,value,width,mask in [('actions',actions,3,known_actions),('future_observations',observations,7,known_observations)]:
            value=torch.as_tensor(value,device=device,dtype=torch.float32)
            mask=torch.as_tensor(mask,device=device,dtype=torch.bool)
            if value.shape!=(n,H,width) or mask.shape!=value.shape:raise ValueError('Invalid completion shape')
            if not torch.isfinite(value[mask]).all():raise ValueError('Nonfinite known constraint')
            b[name][:,:H]=torch.where(mask,value,0.)
            key='known_actions' if name=='actions' else 'known_observations'
            b[key]=torch.zeros(n,6,width,device=device,dtype=torch.bool);b[key][:,:H]=mask
        a=b['actions']
        if (torch.linalg.vector_norm(a[...,:2],dim=-1)>.800001).any() or (a[...,2].abs()>4.000001).any():raise ValueError('Infeasible known actions')
        b['role'].fill_(3);b={k:v.repeat_interleave(K,0) for k,v in b.items()}
        out=self.generate(b,generator)
        return {k:v[:,:H].reshape(n,K,H,v.shape[-1]) for k,v in out.items()}
