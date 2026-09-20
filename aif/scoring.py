"""Operational-cost estimates from learned futures; no hypothetical simulator."""
from __future__ import annotations
import numpy as np
from aif.operational_cost import batch_costs
from environment.success import CRITERION, history_count, settling


def estimate_cost(prediction,context,actions,seed=9051,*,event_uniforms=None):
    """Equal-weight sampled futures and Bernoulli event-head draws.

    Success requires the shared position, speed, dwell and safety contract.
    Only actual history initializes dwell; sampled futures are never padded.
    Collision/fall heads are supplemented by predicted ball geometry. Capability
    evaluation and control share this estimator; validation sets risk thresholds.
    """
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    y=array(prediction['observations']);prob=array(prediction['event_probabilities']);actions=array(actions)
    n,s,h,_=y.shape
    if h not in (1,3,6) or not np.isfinite(y).all() or not np.isfinite(prob).all():raise ValueError('Invalid learned predictions')
    geometry=np.repeat(array(context['geometry']),s,0);future=y.reshape(n*s,h,7)
    uniforms=np.random.default_rng(seed).random(prob.shape) if event_uniforms is None else np.broadcast_to(np.asarray(event_uniforms),prob.shape)
    if not np.isfinite(uniforms).all() or (uniforms<0).any() or (uniforms>=1).any():raise ValueError('Invalid event uniforms')
    ev=uniforms<np.clip(prob,0,1);ev=ev.reshape(n*s,h,5)
    clear=np.linalg.norm(future[:,:,:2]-geometry[:,None,2:4],axis=-1)-geometry[:,None,4]-.04
    ev[:,:,1] |= clear<0;ev[:,:,2] |= (abs(future[:,:,0])>.8)|(abs(future[:,:,1])>1.)
    elapsed=np.repeat(array(context['time_fraction'])*100,s);ev[:,:,3]=elapsed[:,None]+np.arange(1,h+1)[None,:]>=100-1e-4
    history=array(context['history_observations']);mask=array(context.get('history_mask',np.ones(history.shape[:2],bool)))
    counts=np.repeat([history_count(row,goal[:2],mask=present,elapsed_steps=int(round(t*100)))
        for row,goal,present,t in zip(history,array(context['geometry']),mask,array(context['time_fraction']))],s)
    for step in range(h):
        safe=~ev[:,step,1:3].any(-1)
        counts=np.where(safe & settling(future[:,step],geometry[:,:2]),counts+1,0)
        ev[:,step,0]=counts>=CRITERION.steps
        ev[:,step,3] &= safe & ~ev[:,step,0]
    terminal=ev[:,:,:4].any(-1);first=np.where(terminal.any(1),terminal.argmax(1),h-1)
    valid=np.arange(h)[None,:]<=first[:,None];ev &= valid[...,None]
    b={'history_observations':np.repeat(array(context['history_observations']),s,0),
       'history_actions':np.repeat(array(context['history_actions']),s,0),'geometry':geometry,
       'actions':np.repeat(actions,s,0),'future_observations':future,'valid':valid,'events':ev}
    cost=batch_costs(b,h).reshape(n,s)
    adverse=(ev[:,:,1:3].any(-1)&valid).any(-1).reshape(n,s)
    return {'expected_cost':cost.mean(1),'risk':adverse.mean(1),'sample_costs':cost,
        'risk_samples':adverse,'risk_standard_error':adverse.std(1,ddof=1)/np.sqrt(s),
        'cost_standard_error':cost.std(1,ddof=1)/np.sqrt(s),
        'events':ev.reshape(n,s,h,5),'valid':valid.reshape(n,s,h)}
