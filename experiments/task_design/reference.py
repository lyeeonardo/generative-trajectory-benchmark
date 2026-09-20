"""Offline reference policy: timed routes with gravity/contact feedback."""
import numpy as np
from aif.operational_cost import applied_action

def control(obs, scene, t, p, side=-1):
    # Smooth timed path, with an uphill/contact feedback controller.
    duration=p['duration'];u=np.clip((t*.05)/duration,0,1)
    x=scene.obstacle_center[0]+side*p['clear']
    # Blend from center to the selected side and back continuously.
    pos=np.array([x*np.sin(np.pi*u)**2,-.65+1.49*u])
    vel=np.array([x*np.pi*np.sin(2*np.pi*u)/duration,1.49/duration]) if u<1 else np.zeros(2)
    grav=np.array([np.sin(scene.lateral_tilt),-np.cos(scene.lateral_tilt)*np.sin(scene.longitudinal_tilt)])*9.81*5/7
    force=p['kp']*(pos-obs[:2])+p['kd']*(vel-obs[2:4])-p['gravity']*grav
    direction=force/max(np.linalg.norm(force),1e-6)
    rod_target=obs[:2]-p['gap']*direction
    v=obs[2:4]+p['rodgain']*(rod_target-obs[4:6])+p['feed']*force
    angle=np.arctan2(direction[1],direction[0])+np.pi/2
    delta=np.arctan2(np.sin(angle-obs[6]),np.cos(angle-obs[6]))
    return applied_action(np.r_[v,p['yawgain']*delta])
