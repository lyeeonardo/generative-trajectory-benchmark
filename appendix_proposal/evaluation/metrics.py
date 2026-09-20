"""Stage 1 control, proposal, and posterior metrics."""

from __future__ import annotations

import numpy as np


def action_smoothness(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[0] <= 1:
        return 0.0
    return float(np.mean(np.sum(np.diff(actions, axis=0) ** 2, axis=-1)))


def action_diversity(candidates: np.ndarray) -> float:
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.shape[0] <= 1:
        return 0.0
    flat = candidates.reshape(candidates.shape[0], -1)
    center = flat.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(flat - center, axis=1)))


def endpoint_diversity(rollouts: list[object]) -> float:
    if len(rollouts) <= 1:
        return 0.0
    endpoints = np.asarray([rollout.observations[-1, :2] for rollout in rollouts], dtype=np.float32)
    center = endpoints.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(endpoints - center, axis=1)))


def summarize_episode_logs(episodes: list[dict[str, object]]) -> dict[str, float]:
    if not episodes:
        return {}
    keys = ("success", "collision", "fall_out", "timeout")
    summary = {f"{key}_rate": float(np.mean([bool(ep.get(key, False)) for ep in episodes])) for key in keys}
    preferred = (
        "final_distance_to_goal",
        "path_length",
        "episode_length",
        "mean_action_magnitude",
        "action_smoothness",
        "minimum_obstacle_clearance",
        "planning_time_per_decision",
        "proposal_time_per_decision",
        "rollout_scoring_time_per_decision",
        "total_planning_time_per_episode",
        "wall_clock_time_per_decision",
        "total_wall_clock_time_per_episode",
    )
    for key in preferred:
        values = [float(ep[key]) for ep in episodes if key in ep]
        if values:
            summary[key] = float(np.mean(values))
    for key in sorted(set().union(*(ep.keys() for ep in episodes))):
        if key in summary or key in {"scene_id", "seed", "success", "collision", "fall_out", "timeout"}:
            continue
        values = []
        for ep in episodes:
            value = ep.get(key)
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                values.append(float(value))
        if values:
            summary[key] = float(np.mean(values))
    return summary


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def proposal_metrics(plan_result) -> dict[str, float]:
    rollouts = plan_result.rollouts
    candidates = plan_result.candidates
    G = np.asarray([score.G_total for score in plan_result.score_breakdowns], dtype=np.float32)
    routes = [rollout.route_label for rollout in rollouts]
    route_array = np.asarray(routes, dtype=str)
    fall_out = np.asarray([bool(getattr(rollout, "fall_out", False)) for rollout in rollouts])
    valid = np.asarray([not rollout.collision and not bool(getattr(rollout, "fall_out", False)) and np.all(np.isfinite(rollout.observations)) for rollout in rollouts])
    collision = np.asarray([rollout.collision for rollout in rollouts])
    success_proxy = np.asarray([rollout.success for rollout in rollouts], dtype=np.float32)
    route_counts = {label: routes.count(label) for label in ("left", "right", "center", "invalid")}
    route_probs = np.asarray(list(route_counts.values()), dtype=np.float32) / max(len(routes), 1)
    route_entropy = float(-np.sum(route_probs * np.log(np.clip(route_probs, 1e-12, 1.0))))
    selected_index = int(plan_result.selected_index)
    selected_rank = float(int(np.argsort(G).tolist().index(selected_index)))
    log_prob_values = plan_result.diagnostics.get("proposal_log_prob") if getattr(plan_result, "diagnostics", None) else None
    log_prob = None if log_prob_values is None else np.asarray(log_prob_values, dtype=np.float32).reshape(-1)
    selected_log_prob = float(log_prob[selected_index]) if log_prob is not None and log_prob.shape[0] == G.shape[0] else float("nan")
    log_prob_neg_g_corr = _corrcoef(log_prob, -G) if log_prob is not None and log_prob.shape[0] == G.shape[0] else float("nan")
    log_prob_success_corr = _corrcoef(log_prob, success_proxy) if log_prob is not None and log_prob.shape[0] == G.shape[0] else float("nan")
    proposal_time = max(float(plan_result.proposal_time), 1e-12)
    return {
        "feasible_proposal_rate": float(np.mean(valid)),
        "candidate_collision_rate": float(np.mean(collision)),
        "candidate_fall_out_rate": float(np.mean(fall_out)),
        "best_of_K_final_distance": float(min(rollout.final_distance_to_goal for rollout in rollouts)),
        "best_of_K_G_total": float(np.min(G)),
        "median_G_total": float(np.median(G)),
        "best_of_K_success_proxy": float(any(rollout.success for rollout in rollouts)),
        "route_mode_count": float(sum(1 for count in route_counts.values() if count > 0)),
        "left_route_covered": float(route_counts["left"] > 0),
        "right_route_covered": float(route_counts["right"] > 0),
        "both_side_routes_covered": float(route_counts["left"] > 0 and route_counts["right"] > 0),
        "left_route_fraction": float(route_counts["left"] / max(len(routes), 1)),
        "right_route_fraction": float(route_counts["right"] / max(len(routes), 1)),
        "center_route_fraction": float(route_counts["center"] / max(len(routes), 1)),
        "invalid_route_fraction": float(route_counts["invalid"] / max(len(routes), 1)),
        "feasible_left_route_fraction": float(np.mean((route_array == "left") & valid)) if route_array.size else 0.0,
        "feasible_right_route_fraction": float(np.mean((route_array == "right") & valid)) if route_array.size else 0.0,
        "feasible_center_route_fraction": float(np.mean((route_array == "center") & valid)) if route_array.size else 0.0,
        "feasible_side_route_coverage": float(np.any((route_array == "left") & valid) and np.any((route_array == "right") & valid)) if route_array.size else 0.0,
        "route_entropy": route_entropy,
        "action_diversity": action_diversity(candidates),
        "endpoint_diversity": endpoint_diversity(rollouts),
        "invalid_action_fraction": float(np.mean(~np.isfinite(candidates))),
        "selected_candidate_rank_by_G": selected_rank,
        "selected_candidate_G_total": float(G[selected_index]),
        "selected_candidate_score": float(G[selected_index]),
        "posterior_entropy": float(plan_result.posterior_entropy),
        "max_policy_probability": float(np.max(plan_result.policy_posterior)),
        "posterior_mass_left": float(plan_result.route_distribution.get("left", 0.0)),
        "posterior_mass_right": float(plan_result.route_distribution.get("right", 0.0)),
        "posterior_left_mass": float(plan_result.route_distribution.get("left", 0.0)),
        "posterior_right_mass": float(plan_result.route_distribution.get("right", 0.0)),
        "selected_vs_best_of_K_gap": float(G[selected_index] - np.min(G)),
        "selected_vs_best_gap": float(G[selected_index] - np.min(G)),
        "selected_vs_median_gap": float(G[selected_index] - np.median(G)),
        "selected_candidate_log_prob": selected_log_prob,
        "log_prob_neg_G_correlation": log_prob_neg_g_corr,
        "log_prob_success_proxy_correlation": log_prob_success_corr,
        "candidates_per_second": float(candidates.shape[0] / proposal_time),
    }
