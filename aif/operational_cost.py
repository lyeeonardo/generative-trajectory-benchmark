"""Simulator-free physical horizon cost for joint-model data/control."""
from __future__ import annotations

import numpy as np

COMPONENTS = ('goal_terminal', 'goal_path', 'backward_progress', 'collision',
              'fall_out', 'clearance', 'boundary', 'smoothness', 'effort', 'timeout')
DEFAULT_WEIGHTS = dict(zip(COMPONENTS, (12., 1., 5., 150., 150., 15., 8., .05, .08, 25.)))


def applied_action(action, max_speed=.8, max_omega=4.):
    value = np.asarray(action, dtype=np.float32).copy()
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError('Actions must be finite (...,3) arrays')
    speed = np.linalg.norm(value[..., :2], axis=-1, keepdims=True)
    value[..., :2] *= np.minimum(1., max_speed / np.maximum(speed, 1e-12))
    value[..., 2] = np.clip(value[..., 2], -max_omega, max_omega)
    return value


def horizon_cost(observations, actions, events, valid_mask, geometry, previous_action,
                 *, weights=None, ball_radius=.04, clearance_margin=.04,
                 workspace_x=(-.8, .8), workspace_y=(-1., 1.)):
    """Current+future 7D physical observations; events=[success,collision,fall,timeout,contact].

    Path/effort/smoothness use means over actual valid steps. Terminal distance is
    charged once at the last valid state, including a nonterminal horizon boundary.
    Event penalties occur once per horizon. Post-terminal padding is ignored.
    """
    obs = np.asarray(observations, dtype=np.float64)
    act = np.asarray(actions, dtype=np.float64)
    evt = np.asarray(events, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if obs.shape != (len(act)+1, 7) or evt.shape != (len(act), 5) or valid.shape != (len(act),):
        raise ValueError('Expected aligned current+H observations, H actions/events/mask')
    n = int(valid.sum())
    if n < 1 or not np.array_equal(valid, np.arange(len(act)) < n):
        raise ValueError('Validity must be a nonempty contiguous prefix')
    if evt[:max(n-1, 0), :4].any():
        raise ValueError('Transitions after a terminal event are not physical targets')
    geom = np.asarray(geometry, dtype=np.float64)
    prev = np.asarray(previous_action, dtype=np.float64)
    if geom.shape != (5,) or prev.shape != (3,):
        raise ValueError('Geometry=(goal_xy, obstacle_xy, obstacle_radius), previous action=(3,)')
    if not all(np.isfinite(v).all() for v in (obs[:n+1], act[:n], geom, prev)):
        raise ValueError('Valid physical inputs must be finite')
    distances = np.linalg.norm(obs[:n+1, :2] - geom[:2], axis=-1)
    clearance = np.linalg.norm(obs[1:n+1, :2] - geom[2:4], axis=-1) - geom[4] - ball_radius
    boundary = sum(np.mean(np.maximum(lo-obs[1:n+1, j], 0.)**2 + np.maximum(obs[1:n+1, j]-hi, 0.)**2)
                   for j, (lo, hi) in enumerate((workspace_x, workspace_y)))
    changes = np.diff(np.concatenate((prev[None], act[:n]), axis=0), axis=0)
    raw = np.array([distances[-1], distances[1:].mean(), max(0., distances[-1]-distances[0]),
                    evt[:n, 1].any(), evt[:n, 2].any(), max(0., clearance_margin-clearance.min())**2,
                    boundary, np.mean(np.sum(changes**2, axis=-1)),
                    np.mean(np.sum(act[:n]**2, axis=-1)), evt[:n, 3].any()], dtype=np.float64)
    w = DEFAULT_WEIGHTS if weights is None else weights
    parts = raw * np.array([w[key] for key in COMPONENTS])
    return float(parts.sum()), parts


def quality_labels(returns, safe_proposal, *, min_spread, minimum_count=6):
    """Train-anchor tertiles only. Caller controls split; D rows always NULL=0."""
    values = np.asarray(returns, dtype=np.float64)
    eligible = np.asarray(safe_proposal, dtype=bool) & np.isfinite(values)
    labels = np.zeros(values.shape, dtype=np.int8)  # NULL=0, LOW=1, HIGH=2
    selected = values[eligible]
    result = {'count': int(selected.size), 'spread': 0., 'supported': False, 'cutpoints': None}
    if selected.size:
        result['spread'] = float(np.ptp(selected))
    if selected.size < minimum_count or result['spread'] < min_spread:
        return labels, result
    low, high = np.quantile(selected, [1/3, 2/3])
    result['cutpoints'] = [float(low), float(high)]
    # Tied observations at a cutpoint remain NULL; no arbitrary split of ties.
    labels[eligible & (values < low)] = 1
    labels[eligible & (values > high)] = 2
    result['supported'] = bool(np.any(labels == 1) and np.any(labels == 2))
    return labels, result

def batch_costs(batch,horizon):
 """Vectorized canonical horizon cost; independently parity-checked on real rows."""
 y=batch['future_observations'][:,:horizon].astype(np.float64)
 now=batch['history_observations'][:,-1].astype(np.float64)
 a=batch['actions'][:,:horizon].astype(np.float64);valid=batch['valid'][:,:horizon]
 ev=batch['events'][:,:horizon];g=batch['geometry'].astype(np.float64)
 n=valid.sum(1);last=y[np.arange(len(y)),n-1]
 dist=np.linalg.norm(y[:,:,:2]-g[:,None,:2],axis=-1)
 last_dist=np.linalg.norm(last[:,:2]-g[:,:2],axis=-1)
 first_dist=np.linalg.norm(now[:,:2]-g[:,:2],axis=-1)
 clear=np.linalg.norm(y[:,:,:2]-g[:,None,2:4],axis=-1)-g[:,None,4]-.04
 clear=np.where(valid,clear,np.inf).min(1)
 boundary=np.zeros(len(y))
 for j,(lo,hi) in enumerate(((-.8,.8),(-1.,1.))):
  boundary+=((np.maximum(lo-y[:,:,j],0)**2+np.maximum(y[:,:,j]-hi,0)**2)*valid).sum(1)/n
 prev=batch['history_actions'][:,-1].astype(np.float64)
 changes=np.diff(np.concatenate((prev[:,None],a),axis=1),axis=1)
 raw=np.stack([last_dist,(dist*valid).sum(1)/n,np.maximum(last_dist-first_dist,0),
   (ev[:,:,1]&valid).any(1),(ev[:,:,2]&valid).any(1),np.maximum(.04-clear,0)**2,
   boundary,(np.sum(changes**2,-1)*valid).sum(1)/n,(np.sum(a**2,-1)*valid).sum(1)/n,
   (ev[:,:,3]&valid).any(1)],axis=-1)
 return (raw*np.array(list(DEFAULT_WEIGHTS.values()))).sum(-1)
