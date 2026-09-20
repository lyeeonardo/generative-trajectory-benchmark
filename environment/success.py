"""Single success contract for physical execution and learned prediction."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs/success_criterion.json"
@dataclass(frozen=True)
class SuccessCriterion:
    version: int
    goal_radius: float
    steps: int
    speed: float

_raw = json.loads(CONTRACT_PATH.read_text())
CRITERION = SuccessCriterion(_raw['version'], _raw['goal_region']['radius_m'],
    _raw['success']['consecutive_safe_steps'], _raw['success']['ball_speed_strictly_below_m_s'])

def settling(observation, goal):
    observation=np.asarray(observation)
    return (np.linalg.norm(observation[..., :2]-np.asarray(goal),axis=-1)<=CRITERION.goal_radius) & (np.linalg.norm(observation[..., 2:4],axis=-1)<CRITERION.speed)

def history_count(history, goal, *, mask=None, elapsed_steps=None):
    """Count trailing observed transitions; padded slots and initial state never count.

    Online histories exist only while execution is safe. A safety failure ends the
    episode, so no hidden collision state is needed to reconstruct this count.
    """
    history=np.asarray(history)
    if history.ndim!=2:raise ValueError('Expected history x channels')
    present=np.ones(len(history),bool) if mask is None else np.asarray(mask,bool).copy()
    indices=np.flatnonzero(present)
    if elapsed_steps is not None:indices=indices[-min(len(indices),int(elapsed_steps)):] if elapsed_steps else indices[:0]
    else:indices=indices[1:]
    count=0
    for index in indices[::-1]:
        if not settling(history[index],goal):break
        count+=1
        if count>=CRITERION.steps:break
    return count

def score_trace(observations, goal, safe_steps=None):
    """Score transitions up to first success or safety failure, never padded future."""
    observations=np.asarray(observations)
    if observations.ndim!=2 or observations.shape[1]<4 or not len(observations):raise ValueError('Expected initial plus next observations')
    safe=np.ones(len(observations)-1,bool) if safe_steps is None else np.asarray(safe_steps,bool)
    if safe.shape!=(len(observations)-1,):raise ValueError('One safety value per transition required')
    count=0;attained=None;entry=None
    for step,obs in enumerate(observations[1:],1):
        if np.linalg.norm(obs[:2]-goal)<=CRITERION.goal_radius and entry is None:entry=step
        count=count+1 if safe[step-1] and settling(obs,goal) else 0
        if not safe[step-1]:break
        if count>=CRITERION.steps:attained=step;break
    return dict(success=attained is not None,success_step=attained,goal_entry=entry is not None,first_goal_entry_step=entry)
