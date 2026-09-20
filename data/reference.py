"""Offline reference routes and branch generation; never used for online selection."""
from dataclasses import asdict
import math
import numpy as np
from mujoco_task.sim.scene import wrap_angle
def _norm(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(2)
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        if fallback is None:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return _norm(np.asarray(fallback, dtype=np.float32).reshape(2))
    return (vector / length).astype(np.float32)


def _segment_clearance(point_a: np.ndarray, point_b: np.ndarray, center: np.ndarray) -> float:
    a = np.asarray(point_a, dtype=np.float32).reshape(2)
    b = np.asarray(point_b, dtype=np.float32).reshape(2)
    c = np.asarray(center, dtype=np.float32).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-8:
        return float(np.linalg.norm(a - c))
    t = float(np.clip(np.dot(c - a, ab) / denom, 0.0, 1.0))
    nearest = a + t * ab
    return float(np.linalg.norm(nearest - c))


def route_waypoints(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    ball_radius: float,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
) -> list[list[np.ndarray]]:
    start = np.asarray(start, dtype=np.float32).reshape(2)
    goal = np.asarray(goal, dtype=np.float32).reshape(2)
    center = np.asarray(obstacle_center, dtype=np.float32).reshape(2)
    routes: list[list[np.ndarray]] = []
    blocked_distance = float(obstacle_radius) + float(ball_radius) + 0.18
    if _segment_clearance(start, goal, center) > blocked_distance:
        routes.append([goal.copy()])

    direction = _norm(goal - start)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    min_xy = np.asarray([float(workspace_x[0]) + 0.10, float(workspace_y[0]) + 0.10], dtype=np.float32)
    max_xy = np.asarray([float(workspace_x[1]) - 0.10, float(workspace_y[1]) - 0.10], dtype=np.float32)
    for clearance in (0.34, 0.44, 0.56, 0.66):
        for side in (-1.0, 1.0):
            before = np.clip(center + side * clearance * normal - 0.20 * direction, min_xy, max_xy)
            after = np.clip(center + side * clearance * normal + 0.24 * direction, min_xy, max_xy)
            routes.append([before.astype(np.float32), after.astype(np.float32), goal.copy()])

    for x_value in (-0.62, -0.46, 0.46, 0.62):
        x = float(np.clip(x_value, min_xy[0], max_xy[0]))
        routes.append(
            [
                np.asarray([x, start[1]], dtype=np.float32),
                np.asarray([x, 0.0], dtype=np.float32),
                np.asarray([x, goal[1]], dtype=np.float32),
                goal.copy(),
            ]
        )
    for y_value in (-0.78, -0.46, 0.46, 0.78):
        y = float(np.clip(y_value, min_xy[1], max_xy[1]))
        routes.append(
            [
                np.asarray([start[0], y], dtype=np.float32),
                np.asarray([0.0, y], dtype=np.float32),
                np.asarray([goal[0], y], dtype=np.float32),
                goal.copy(),
            ]
        )
    return routes


def waypoint_action(
    obs: np.ndarray,
    waypoint: np.ndarray,
    *,
    lateral_tilt: float,
    longitudinal_tilt: float,
    ball_radius: float,
    rod_half_width: float,
    action_max_speed: float,
    action_max_omega: float,
    position_gain: float,
    push_speed: float,
    drift_compensation: float,
) -> np.ndarray:
    ball = np.asarray(obs[:2], dtype=np.float32)
    rod = np.asarray(obs[4:6], dtype=np.float32)
    yaw = float(obs[6])
    desired = _norm(np.asarray(waypoint, dtype=np.float32) - ball, fallback=np.asarray(obs[7:9]) - ball)
    downhill = np.asarray([math.sin(lateral_tilt), -math.sin(longitudinal_tilt)], dtype=np.float32)
    push = _norm(desired - float(drift_compensation) * downhill, fallback=desired)
    gap = float(ball_radius) + float(rod_half_width) + 0.012
    target_rod = ball - gap * push
    rod_error = target_rod - rod
    velocity = float(position_gain) * rod_error + float(push_speed) * push
    speed = float(np.linalg.norm(velocity))
    if speed > float(action_max_speed):
        velocity = velocity / max(speed, 1e-8) * float(action_max_speed)
    target_yaw = math.atan2(float(push[1]), float(push[0])) + 0.5 * math.pi
    omega = float(np.clip(7.0 * wrap_angle(target_yaw - yaw), -float(action_max_omega), float(action_max_omega)))
    return np.asarray([velocity[0], velocity[1], omega], dtype=np.float32)

def pack_state(state):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(state).items()}


def plan_branches(obs, reference, config, rng):
    geom=obs[7:12];goal=geom[:2];center=geom[2:4]
    direction=goal-obs[:2];direction/=max(float(np.linalg.norm(direction)),1e-8)
    normal=np.array([-direction[1],direction[0]],dtype=np.float32)
    plans=[reference.copy(),reference*.8,reference*.6,reference*1.15]
    for side in (-1,1):
        target=center+side*(geom[4]+.04+.22)*normal
        if float(np.dot(center-obs[:2],direction))<0:
            target=goal+side*.06*normal
        for speed in (.35,.65):
            a=waypoint_action(obs,target,lateral_tilt=float(obs[12]),longitudinal_tilt=float(obs[13]),
                 ball_radius=.04,rod_half_width=.03,action_max_speed=.8,action_max_omega=4.,
                 position_gain=6.5,push_speed=speed,drift_compensation=.12)
            plans.append(np.repeat(a[None],6,axis=0))
    brake=np.r_[-obs[2:4]*2.,0.].astype(np.float32)
    wrong=waypoint_action(obs,goal,lateral_tilt=-float(obs[12]),longitudinal_tilt=0.,ball_radius=.04,
             rod_half_width=.03,action_max_speed=.8,action_max_omega=4.,position_gain=6.5,push_speed=.58,drift_compensation=.12)
    plans.extend([np.zeros((6,3)),np.repeat(reference[:1],6,axis=0),np.repeat(brake[None],6,axis=0),
        np.repeat(rng.uniform([-1.,-1.,-5.],[1.,1.,5.],size=(1,3)),6,axis=0),
        rng.uniform([-1.,-1.,-5.],[1.,1.,5.],size=(6,3)),reference+rng.normal(0,[.35,.35,1.5],size=(6,3)),
        np.repeat(wrong[None],6,axis=0),-reference])
    plans.extend([reference*.9,reference*1.05])
    for angle in (-.08,.08):
        seq=reference.copy();rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        seq[:,:2]=seq[:,:2]@rot.T;plans.append(seq)
    return np.asarray(plans,dtype=np.float32)
