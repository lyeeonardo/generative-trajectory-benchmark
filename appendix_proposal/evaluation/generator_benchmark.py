"""Common Generator benchmark evaluator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from aif.belief import BeliefState
from aif.context import generator_context_dim
from aif.hidden_tilt_belief import HiddenTiltBelief
from aif.planner import AIFPlanner
from aif.proposal_mixture import ProposalMixtureGenerator, rho_from_omega
from aif.reliability import make_reliability_gate
from envs.tilted_board_adapter import TiltedBoardEnvAdapter
from evaluation.compute_budget import compute_benchmark_hashes, save_benchmark_hashes
from evaluation.metrics import action_smoothness, proposal_metrics, summarize_episode_logs
from evaluation.visualize import plot_diversity_quality, plot_posterior_selected_comparison, plot_proposal_cloud_comparison, plot_runtime_pareto, plot_topk_rollout_overlay
from generators.composite import OracleSourceSelectorGenerator
from generators.probe_router import LinearRoutingModel, ProbeAndRouteGenerator
from generators.registry import get_generator

# Campaign and primary manifests belonged to removed non-paper OOD tooling.
# The final Appendix A configs never set either manifest path.
class _RemovedManifest:

    def load(cls, path):
        raise ValueError(f"Manifest-based OOD campaigns are not included in the paper release: {path}")

class _RemovedScene:
    pass

CampaignManifest = PrimaryManifest = _RemovedManifest
CampaignScene = PrimaryScene = _RemovedScene


def _as_list(value, *, cast=str) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [cast(part) for part in parts]
    return [cast(item) for item in value]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _campaign_row_fields(
    scene: CampaignScene | PrimaryScene | None,
    manifest: CampaignManifest | PrimaryManifest | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    training_seed = -1 if config.get("training_seed") is None else int(config["training_seed"])
    if scene is None or manifest is None:
        return {
            "campaign_split": "",
            "campaign_manifest_hash": "",
            "shift_stratum": "",
            "shift_severity": float("nan"),
            "campaign_feasibility_status": "",
            "benchmark_population": "",
            "transfer_severity": float("nan"),
            "training_seed": training_seed,
            "checkpoint_id": str(config.get("checkpoint_id", "")),
        }
    fields: dict[str, Any] = {
        "campaign_split": manifest.split,
        "campaign_manifest_hash": str(manifest.manifest_hash),
        "shift_stratum": scene.shift_stratum,
        "shift_severity": float(scene.shift_severity),
        "campaign_feasibility_status": scene.feasibility_status,
        "campaign_scene_seed": int(scene.seed),
        "campaign_physics_params": json.dumps(scene.physics_params, sort_keys=True),
        "campaign_sim_params": json.dumps(scene.sim_params, sort_keys=True),
        "training_seed": training_seed,
        "checkpoint_id": str(config.get("checkpoint_id", "")),
    }
    if isinstance(scene, PrimaryScene):
        fields.update(
            {
                "shift_stratum": scene.population,
                "shift_severity": float(scene.transfer_severity),
                "benchmark_population": scene.population,
                "transfer_severity": float(scene.transfer_severity),
            }
        )
    else:
        fields.update(
            {
                "benchmark_population": "",
                "transfer_severity": float("nan"),
                **{f"shift_{key}": float(value) for key, value in scene.shift_factors.items()},
            }
        )
    return fields


def _trajectory_artifacts(records: list[dict[str, Any]], *, action_dim: int = 3) -> dict[str, np.ndarray]:
    if not records:
        return {
            "trajectory_xy": np.zeros((0, 0, 2), dtype=np.float32),
            "trajectory_mask": np.zeros((0, 0), dtype=bool),
            "actions": np.zeros((0, 0, action_dim), dtype=np.float32),
            "action_mask": np.zeros((0, 0), dtype=bool),
            "goal_xy": np.zeros((0, 2), dtype=np.float32),
            "obstacle_xy": np.zeros((0, 2), dtype=np.float32),
            "obstacle_radius": np.zeros((0,), dtype=np.float32),
            "run_id": np.asarray([], dtype=str),
            "model": np.asarray([], dtype=str),
            "experiment": np.asarray([], dtype=str),
            "variant": np.asarray([], dtype=str),
            "benchmark_mode": np.asarray([], dtype=str),
            "split": np.asarray([], dtype=str),
            "preset": np.asarray([], dtype=str),
            "scene_id": np.asarray([], dtype=np.int64),
            "seed": np.asarray([], dtype=np.int64),
            "candidate_budget": np.asarray([], dtype=np.int64),
            "success": np.asarray([], dtype=bool),
            "collision": np.asarray([], dtype=bool),
            "fall_out": np.asarray([], dtype=bool),
            "timeout": np.asarray([], dtype=bool),
            "terminal_reason": np.asarray([], dtype=str),
            "final_distance_to_goal": np.asarray([], dtype=np.float32),
            "episode_length": np.asarray([], dtype=np.int64),
            "physics_backend": np.asarray([], dtype=str),
            "ood": np.asarray([], dtype=bool),
            "true_tilt_lateral": np.asarray([], dtype=np.float32),
            "true_tilt_longitudinal": np.asarray([], dtype=np.float32),
            "training_seed": np.asarray([], dtype=np.int64),
            "checkpoint_id": np.asarray([], dtype=str),
            "campaign_split": np.asarray([], dtype=str),
            "campaign_manifest_hash": np.asarray([], dtype=str),
            "shift_stratum": np.asarray([], dtype=str),
            "shift_severity": np.asarray([], dtype=np.float32),
        }

    max_trajectory_len = max(int(np.asarray(record["trajectory_xy"]).shape[0]) for record in records)
    max_action_len = max(int(np.asarray(record["actions"]).shape[0]) for record in records)
    trajectory_xy = np.zeros((len(records), max_trajectory_len, 2), dtype=np.float32)
    trajectory_mask = np.zeros((len(records), max_trajectory_len), dtype=bool)
    actions = np.zeros((len(records), max_action_len, action_dim), dtype=np.float32)
    action_mask = np.zeros((len(records), max_action_len), dtype=bool)

    for index, record in enumerate(records):
        path = np.asarray(record["trajectory_xy"], dtype=np.float32).reshape(-1, 2)
        action_array = np.asarray(record["actions"], dtype=np.float32).reshape(-1, action_dim)
        trajectory_xy[index, : path.shape[0]] = path
        trajectory_mask[index, : path.shape[0]] = True
        if action_array.size:
            actions[index, : action_array.shape[0]] = action_array
            action_mask[index, : action_array.shape[0]] = True

    return {
        "trajectory_xy": trajectory_xy,
        "trajectory_mask": trajectory_mask,
        "actions": actions,
        "action_mask": action_mask,
        "goal_xy": np.asarray([record["goal_xy"] for record in records], dtype=np.float32),
        "obstacle_xy": np.asarray([record["obstacle_xy"] for record in records], dtype=np.float32),
        "obstacle_radius": np.asarray([record["obstacle_radius"] for record in records], dtype=np.float32),
        "run_id": np.asarray([record["run_id"] for record in records], dtype=str),
        "model": np.asarray([record["model"] for record in records], dtype=str),
        "experiment": np.asarray([record.get("experiment", "") for record in records], dtype=str),
        "variant": np.asarray([record.get("variant", "") for record in records], dtype=str),
        "benchmark_mode": np.asarray([record["benchmark_mode"] for record in records], dtype=str),
        "split": np.asarray([record["split"] for record in records], dtype=str),
        "preset": np.asarray([record["preset"] for record in records], dtype=str),
        "scene_id": np.asarray([record["scene_id"] for record in records], dtype=np.int64),
        "seed": np.asarray([record["seed"] for record in records], dtype=np.int64),
        "candidate_budget": np.asarray([record.get("candidate_budget", 0) for record in records], dtype=np.int64),
        "success": np.asarray([record["success"] for record in records], dtype=bool),
        "collision": np.asarray([record["collision"] for record in records], dtype=bool),
        "fall_out": np.asarray([record.get("fall_out", False) for record in records], dtype=bool),
        "timeout": np.asarray([record["timeout"] for record in records], dtype=bool),
        "terminal_reason": np.asarray([record.get("terminal_reason", "") for record in records], dtype=str),
        "final_distance_to_goal": np.asarray([record["final_distance_to_goal"] for record in records], dtype=np.float32),
        "episode_length": np.asarray([record["episode_length"] for record in records], dtype=np.int64),
        "physics_backend": np.asarray([record.get("physics_backend", "mujoco_rigid") for record in records], dtype=str),
        "ood": np.asarray([record.get("ood", False) for record in records], dtype=bool),
        "true_tilt_lateral": np.asarray([record.get("true_tilt_lateral", np.nan) for record in records], dtype=np.float32),
        "true_tilt_longitudinal": np.asarray([record.get("true_tilt_longitudinal", np.nan) for record in records], dtype=np.float32),
        "training_seed": np.asarray([record.get("training_seed", -1) for record in records], dtype=np.int64),
        "checkpoint_id": np.asarray([record.get("checkpoint_id", "") for record in records], dtype=str),
        "campaign_split": np.asarray([record.get("campaign_split", "") for record in records], dtype=str),
        "campaign_manifest_hash": np.asarray([record.get("campaign_manifest_hash", "") for record in records], dtype=str),
        "shift_stratum": np.asarray([record.get("shift_stratum", "") for record in records], dtype=str),
        "shift_severity": np.asarray([record.get("shift_severity", np.nan) for record in records], dtype=np.float32),
    }


def _model_config(config: dict[str, Any], model: str) -> dict[str, Any]:
    model_cfg = dict(config)
    model_cfg["model"] = model
    model_cfg.setdefault("context_dim", generator_context_dim(4))
    per_model = config.get("model_configs", {})
    if isinstance(per_model, dict) and isinstance(per_model.get(model), dict):
        model_cfg.update(per_model[model])
    checkpoint_paths = config.get("checkpoint_paths", {})
    if isinstance(checkpoint_paths, dict) and checkpoint_paths.get(model):
        model_cfg["checkpoint_path"] = checkpoint_paths[model]
    return model_cfg


def _tilt_pairs_from_degrees(spec: Any) -> list[tuple[float, float]]:
    if not isinstance(spec, dict):
        return []
    longitudinal = spec.get("longitudinal", [])
    lateral = spec.get("lateral", [])
    if isinstance(longitudinal, (int, float, str)):
        longitudinal = [longitudinal]
    if isinstance(lateral, (int, float, str)):
        lateral = [lateral]
    pairs: list[tuple[float, float]] = []
    for lon in longitudinal:
        for lat in lateral:
            pairs.append((float(np.deg2rad(float(lat))), float(np.deg2rad(float(lon)))))
    return pairs


def _tilt_override_for_run(config: dict[str, Any], *, scene_id: int, seed: int) -> tuple[float, float] | None:
    explicit = config.get("tilt_override_degrees")
    if isinstance(explicit, dict):
        lateral = float(explicit.get("lateral", explicit.get("lateral_tilt_deg", 0.0)))
        longitudinal = float(explicit.get("longitudinal", explicit.get("longitudinal_tilt_deg", 0.0)))
        return float(np.deg2rad(lateral)), float(np.deg2rad(longitudinal))
    pairs = _tilt_pairs_from_degrees(config.get("ood_tilt_degrees"))
    if not pairs:
        return None
    index = (int(scene_id) * 1009 + int(seed) * 9173) % len(pairs)
    return pairs[int(index)]


def _posterior_tilt_replacement(tilt_belief: HiddenTiltBelief | None, mode: str) -> np.ndarray:
    if tilt_belief is None:
        return np.zeros((2,), dtype=np.float32)
    key = str(mode)
    if key == "map":
        return tilt_belief.map_tilt.astype(np.float32)
    if key == "posterior_mean":
        probs = np.asarray(tilt_belief.probabilities, dtype=np.float32).reshape(-1)
        return np.sum(tilt_belief.tilt_grid * probs[:, None], axis=0).astype(np.float32)
    return np.zeros((2,), dtype=np.float32)


def _proposal_rho_for_source(config: dict[str, Any], proposal_source: str, omega_t: float) -> float:
    source = str(proposal_source)
    if source == "generator":
        return 1.0
    if source == "aif_cem":
        return 0.0
    if source == "probe_router":
        return float("nan")
    return rho_from_omega(
        float(omega_t),
        omega_min=float(config.get("omega_min", 0.0)),
        omega_max=float(config.get("omega_max", 1.0)),
        rho_min=float(config.get("rho_min", 0.0)),
        rho_max=float(config.get("rho_max", 1.0)),
    )


def _source_diagnostics(plan) -> dict[str, Any]:
    proposal = plan.diagnostics.get("proposal_diagnostics", {}) if getattr(plan, "diagnostics", None) else {}
    raw_sources = proposal.get("candidate_sources")
    sources = [str(item) for item in raw_sources] if raw_sources is not None else ["generator"] * int(plan.candidates.shape[0])
    G = np.asarray([score.G_total for score in plan.score_breakdowns], dtype=np.float32)
    invalid = np.asarray(
        [
            bool(rollout.collision) or bool(getattr(rollout, "fall_out", False)) or not np.all(np.isfinite(rollout.observations))
            for rollout in plan.rollouts
        ],
        dtype=bool,
    )
    selected_source = sources[int(plan.selected_index)] if int(plan.selected_index) < len(sources) else "unknown"
    diagnostics: dict[str, Any] = {
        "selected_source": selected_source,
        "candidate_invalidity": float(np.mean(invalid)) if invalid.size else 0.0,
        "rho_t": float(proposal.get("rho_t", np.nan)),
        "omega_t": float(proposal.get("omega_t", np.nan)),
        "proposal_mixture_entropy": float(proposal.get("proposal_mixture_entropy", 0.0)),
    }
    # Preserve the auditable pre-action router contract and development-only
    # oracle label in the step table.  These are proposal diagnostics, not
    # post-action gate inputs.
    passthrough_keys = {
        "router_type",
        "router_model_hash",
        "probe_per_source",
        "requested_rho_t",
        "K_generator",
        "K_fallback",
        "primary_probe_best_G",
        "fallback_probe_best_G",
        "primary_probe_mean_G",
        "fallback_probe_mean_G",
        "oracle_primary_wins",
        "oracle_source_winner",
        "oracle_primary_key",
        "oracle_fallback_key",
        "oracle_candidate_count_per_source",
        "oracle_true_state_used_for_label_only",
        "oracle_selected_source",
        "oracle_selected_source_best_index",
        "oracle_selected_source_key",
        "oracle_source_keys",
        "oracle_source_count",
        "oracle_true_rollouts_used",
    }
    for key, value in proposal.items():
        if key in passthrough_keys or str(key).startswith("routing_feature_"):
            diagnostics[str(key)] = value
    best_by_source: dict[str, float] = {}
    for label in sorted(set(sources)):
        mask = np.asarray([source == label for source in sources], dtype=bool)
        if not np.any(mask):
            continue
        diagnostics[f"{label}_candidate_count"] = int(np.sum(mask))
        diagnostics[f"{label}_candidate_invalidity"] = float(np.mean(invalid[mask]))
        diagnostics[f"{label}_candidate_collision_rate"] = float(
            np.mean([bool(plan.rollouts[index].collision) for index, keep in enumerate(mask.tolist()) if keep])
        )
        diagnostics[f"{label}_candidate_fall_out_rate"] = float(
            np.mean([bool(getattr(plan.rollouts[index], "fall_out", False)) for index, keep in enumerate(mask.tolist()) if keep])
        )
        best = float(np.min(G[mask]))
        diagnostics[f"{label}_best_G_total"] = best
        best_by_source[label] = best
    if G.size:
        global_best = float(np.min(G))
        selected_G = float(G[int(plan.selected_index)])
        diagnostics["best_source"] = min(best_by_source, key=best_by_source.get) if best_by_source else selected_source
        diagnostics["selected_source_regret"] = float(max(0.0, selected_G - global_best))
        selected_best = float(best_by_source.get(selected_source, selected_G))
        diagnostics["selected_source_best_gap"] = float(selected_G - selected_best)
    gen_best = diagnostics.get("generator_best_G_total")
    fallback_best = diagnostics.get("fallback_best_G_total")
    if gen_best is not None and fallback_best is not None:
        diagnostics["generator_vs_fallback_G_gap"] = float(float(gen_best) - float(fallback_best))
        diagnostics["generator_regret_vs_fallback"] = float(max(0.0, float(gen_best) - float(fallback_best)))
    return diagnostics


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _one_step_generated_prediction_error(plan, observed_next_obs: np.ndarray) -> float:
    generated_diag = plan.diagnostics.get("generated_state_consistency", {}) if getattr(plan, "diagnostics", None) else {}
    generated = generated_diag.get("selected_generated_one_step")
    if generated is None:
        return float("nan")
    generated_obs = np.asarray(generated, dtype=np.float32).reshape(-1)
    observed = np.asarray(observed_next_obs, dtype=np.float32).reshape(-1)
    width = min(generated_obs.shape[0], observed.shape[0])
    if width <= 0 or not np.all(np.isfinite(generated_obs[:width])):
        return float("nan")
    raw_indices = generated_diag.get("generated_state_feature_indices")
    if raw_indices:
        indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        indices = indices[(indices >= 0) & (indices < width)]
    else:
        indices = np.arange(width, dtype=np.int64)
    if indices.size <= 0:
        return float("nan")
    return float(np.linalg.norm(generated_obs[indices] - observed[indices]))


def _positive_scaled(value: Any, scale: float) -> float:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric):
        return 0.0
    return float(max(0.0, numeric) / max(float(scale), 1e-12))


def _clip_unit(value: Any) -> float:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric):
        return 0.0
    return float(np.clip(numeric, 0.0, 1.0))


def _build_gate_signals(
    config: dict[str, Any],
    *,
    metrics: dict[str, float],
    source_metrics: dict[str, Any],
    generated_diag: dict[str, Any],
    one_step_generated_prediction_error: float,
    tilt_update_diag: dict[str, Any],
    is_ood_run: bool,
) -> dict[str, Any]:
    """Build reliability-gate signals from generator consistency and source quality."""

    energy_scale = max(float(config.get("gate_energy_scale", 50.0)), 1e-12)
    consistency_scale = max(float(config.get("gate_consistency_scale", energy_scale)), 1e-12)
    real_scale = max(float(config.get("gate_real_prediction_scale", 1.0)), 1e-12)

    selected_xi = _float_or_nan(generated_diag.get("selected_Xi"))
    fallback_xi = _float_or_nan(generated_diag.get("mean_candidate_Xi"))
    xi_for_gate = selected_xi if np.isfinite(selected_xi) else fallback_xi
    consistency_missing_penalty = float(config.get("gate_missing_generated_state_penalty", 0.0))
    consistency_signal = _positive_scaled(xi_for_gate, consistency_scale)
    if not np.isfinite(xi_for_gate) and bool(generated_diag.get("generated_state_available", False)):
        consistency_signal = consistency_missing_penalty

    real_prediction_signal = _positive_scaled(one_step_generated_prediction_error, real_scale)
    generator_invalidity = _clip_unit(source_metrics.get("generator_candidate_invalidity", 0.0))
    generator_safety_rate = _clip_unit(
        _float_or_nan(source_metrics.get("generator_candidate_collision_rate", 0.0))
        + _float_or_nan(source_metrics.get("generator_candidate_fall_out_rate", 0.0))
    )
    all_candidate_invalidity = _clip_unit(source_metrics.get("candidate_invalidity", 0.0))
    predicted_safety_rate = generator_safety_rate
    generator_best = _float_or_nan(source_metrics.get("generator_best_G_total"))
    fallback_best = _float_or_nan(source_metrics.get("fallback_best_G_total"))
    best_of_k = _float_or_nan(metrics.get("best_of_K_G_total"))
    regret_raw = generator_best - fallback_best if np.isfinite(generator_best) and np.isfinite(fallback_best) else float("nan")
    score_regret_signal = _positive_scaled(regret_raw, energy_scale)
    quality_source = str(config.get("gate_candidate_quality_source", "generator_best"))
    quality_raw = best_of_k if quality_source == "best_of_k" or not np.isfinite(generator_best) else generator_best
    candidate_quality_signal = _positive_scaled(quality_raw, energy_scale)

    z_delta_ab = float(
        float(config.get("gate_w_cons", 1.0)) * consistency_signal
        + float(config.get("gate_w_real", 1.0)) * real_prediction_signal
        + float(config.get("gate_w_invalid", 1.0)) * generator_invalidity
        + float(config.get("gate_w_regret", 1.0)) * score_regret_signal
        + float(config.get("gate_w_safety", 1.0)) * predicted_safety_rate
    )
    delta_d = _positive_scaled(tilt_update_diag.get("belief_update_kl", 0.0), float(config.get("gate_belief_kl_scale", 1.0)))
    return {
        "z_delta_AB": z_delta_ab,
        "delta_AB": z_delta_ab,
        "delta_R": real_prediction_signal,
        "candidate_invalidity": generator_invalidity,
        "all_candidate_invalidity": all_candidate_invalidity,
        "candidate_quality": candidate_quality_signal,
        "Delta_d": delta_d,
        "ood": bool(is_ood_run),
        "consistency_signal": consistency_signal,
        "real_prediction_signal": real_prediction_signal,
        "generator_invalidity_signal": generator_invalidity,
        "score_regret_signal": score_regret_signal,
        "safety_signal": predicted_safety_rate,
        "generator_safety_signal": generator_safety_rate,
        "candidate_quality_signal": candidate_quality_signal,
        "selected_Xi_raw": selected_xi,
        "one_step_generated_prediction_error_raw": _float_or_nan(one_step_generated_prediction_error),
        "generator_best_G_total": generator_best,
        "fallback_best_G_total": fallback_best,
        "generator_vs_fallback_G_gap": regret_raw,
    }


def _episode_summary(
    actions: list[np.ndarray],
    trajectory: list[np.ndarray],
    obs: np.ndarray,
    info: dict[str, Any],
    env,
    plan_metrics: list[dict[str, float]],
    times: dict[str, list[float]],
    *,
    step_limit: int,
) -> dict[str, Any]:
    actions_array = np.asarray(actions, dtype=np.float32) if actions else np.zeros((0, 3), dtype=np.float32)
    trajectory_array = np.asarray(trajectory, dtype=np.float32)
    success = bool(info.get("success", env.is_success(obs)))
    collision = bool(info.get("collision", env.is_collision(obs)))
    fall_out = bool(info.get("fall_out", False))
    timeout = (bool(info.get("timeout", False)) or len(actions) >= step_limit) and not success and not collision and not fall_out
    episode = {
        "success": success,
        "collision": collision,
        "fall_out": fall_out,
        "timeout": timeout,
        "terminal_reason": str(info.get("terminal_reason", "success" if success else "collision" if collision else "fall_out" if fall_out else "timeout" if timeout else "running")),
        "final_distance_to_goal": float(np.linalg.norm(obs[:2] - obs[7:9])),
        "path_length": float(np.sum(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))) if len(trajectory_array) > 1 else 0.0,
        "episode_length": int(len(actions)),
        "mean_action_magnitude": float(np.mean(np.linalg.norm(actions_array, axis=1))) if len(actions_array) else 0.0,
        "action_smoothness": action_smoothness(actions_array),
        "planning_time_per_decision": float(np.mean(times["planning"])) if times["planning"] else 0.0,
        "proposal_time_per_decision": float(np.mean(times["proposal"])) if times["proposal"] else 0.0,
        "rollout_scoring_time_per_decision": float(np.mean(times["scoring"])) if times["scoring"] else 0.0,
        "total_planning_time_per_episode": float(np.sum(times["planning"])),
        "wall_clock_time_per_decision": float(np.mean(times.get("step_wall", []))) if times.get("step_wall") else 0.0,
        "total_wall_clock_time_per_episode": float(np.sum(times.get("step_wall", []))) if times.get("step_wall") else 0.0,
    }
    if plan_metrics:
        for key in plan_metrics[0].keys():
            values = [float(row[key]) for row in plan_metrics if key in row]
            if values:
                episode[f"mean_{key}"] = float(np.mean(values))
    return episode


def _aggregate(episodes: list[dict[str, Any]], failures: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"models": {}, "failure_count": len(failures)}
    for model in models:
        rows = [row for row in episodes if row.get("model") == model]
        summary = summarize_episode_logs(rows)
        numeric_ci: dict[str, dict[str, float]] = {}
        keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
        for key in keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool, np.integer, np.floating))]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                numeric_ci[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "ci95": float(1.96 * np.std(arr) / np.sqrt(max(arr.size, 1))),
                    "n": int(arr.size),
                }
        aggregate["models"][model] = {
            "episodes": len(rows),
            "failures": len([failure for failure in failures if failure.get("model") == model]),
            "summary": summary,
            "numeric_ci": numeric_ci,
        }
    return aggregate


def run_generator_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    models = _as_list(cfg.get("models", cfg.get("include_models", cfg.get("model", "random_shooting_aif"))), cast=str)
    if not models:
        raise ValueError("At least one model is required.")
    seeds = _as_list(cfg.get("seeds", [0]), cast=int) or [0]
    campaign_manifest: CampaignManifest | None = None
    primary_manifest: PrimaryManifest | None = None
    campaign_manifest_path = cfg.get("campaign_manifest_path")
    primary_manifest_path = cfg.get("primary_manifest_path")
    if campaign_manifest_path and primary_manifest_path:
        raise ValueError("campaign_manifest_path and primary_manifest_path are mutually exclusive")
    if campaign_manifest_path:
        campaign_manifest = CampaignManifest.load(campaign_manifest_path)
        requested_split = cfg.get("campaign_split")
        if requested_split is not None and str(requested_split) != campaign_manifest.split:
            raise ValueError(
                f"campaign_split={requested_split!r} does not match manifest split {campaign_manifest.split!r}"
            )
        if bool(cfg.get("require_oracle_feasible", False)):
            incomplete = [
                scene.scene_id
                for scene in campaign_manifest.scenes
                if scene.feasibility_status != "oracle_pass"
            ]
            if incomplete:
                raise ValueError(
                    "Campaign requires oracle-screened scenes; "
                    f"{len(incomplete)} scene(s) are not oracle_pass, starting with {incomplete[:5]}"
                )
        cfg["campaign_manifest_hash"] = campaign_manifest.manifest_hash
        cfg["campaign_split"] = campaign_manifest.split
    if primary_manifest_path:
        primary_manifest = PrimaryManifest.load(primary_manifest_path)
        requested_split = cfg.get("primary_split")
        if requested_split is not None and str(requested_split) != primary_manifest.split:
            raise ValueError(
                f"primary_split={requested_split!r} does not match manifest split {primary_manifest.split!r}"
            )
        if bool(cfg.get("require_oracle_feasible", False)):
            incomplete = [
                scene.scene_id
                for scene in primary_manifest.scenes
                if scene.feasibility_status != "oracle_pass"
            ]
            if incomplete:
                raise ValueError(
                    "Primary benchmark requires oracle-screened scenes; "
                    f"{len(incomplete)} scene(s) are not oracle_pass, starting with {incomplete[:5]}"
                )
        cfg["campaign_manifest_hash"] = primary_manifest.manifest_hash
        cfg["primary_manifest_hash"] = primary_manifest.manifest_hash
        cfg["primary_split"] = primary_manifest.split
    evaluation_manifest = campaign_manifest if campaign_manifest is not None else primary_manifest
    requested_scene_ids = cfg.get("scene_ids")
    if requested_scene_ids is None and evaluation_manifest is not None:
        scene_ids = list(evaluation_manifest.scene_ids)
    else:
        scene_ids = _as_list([0] if requested_scene_ids is None else requested_scene_ids, cast=int) or [0]
    if evaluation_manifest is not None:
        missing_scene_ids = sorted(set(scene_ids).difference(evaluation_manifest.scene_ids))
        if missing_scene_ids:
            raise ValueError(f"scene_ids are absent from the benchmark manifest: {missing_scene_ids[:8]}")
    split = str(cfg.get("split", "test"))
    preset = str(cfg.get("preset", "smoke"))
    output_dir = Path(cfg.get("output_dir", "outputs/generator/metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)
    K = int(cfg.get("K", 8))
    H = int(cfg.get("H", cfg.get("horizon", 4)))
    gamma_t = float(cfg.get("gamma_t", 2.0))
    max_steps = cfg.get("max_steps")
    benchmark_mode = str(cfg.get("benchmark_mode", cfg.get("mode", "equal_k")))
    if benchmark_mode not in {"equal_k", "equal_time"}:
        raise ValueError("benchmark_mode must be equal_k or equal_time.")
    proposal_source = str(cfg.get("proposal_source", "generator"))
    if proposal_source not in {"generator", "aif_cem", "mixture", "probe_router", "oracle_source_selector"}:
        raise ValueError("proposal_source must be generator, aif_cem, mixture, probe_router, or oracle_source_selector.")
    fallback_model = str(cfg.get("fallback_model", "cem_aif"))
    selection_mode = str(cfg.get("selection_mode", cfg.get("controller_mode", "aif_score")))
    experiment_name = str(cfg.get("experiment", cfg.get("experiment_name", "generator_benchmark")))
    variant_name = str(cfg.get("variant", cfg.get("variant_name", proposal_source)))
    is_ood_run = bool(cfg.get("ood", False) or cfg.get("ood_tilt_degrees") or campaign_manifest is not None)
    record_per_step = bool(cfg.get("record_per_step", True))
    record_per_step_fields = tuple(str(value) for value in cfg.get("record_per_step_fields", ()))
    record_trajectories = bool(cfg.get("record_trajectories", True))

    per_step: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    hash_records: dict[str, dict[str, str | None]] = {}
    model_cards: dict[str, Any] = {}
    first_plans: dict[str, Any] = {}
    for model_name in models:
        model_cfg = _model_config({**cfg, "split": split, "preset": preset, "H": H, "K": K}, model_name)
        try:
            generator = get_generator(model_name, model_cfg)
            model_cards[model_name] = generator.diagnostics()
            hash_records[model_name] = compute_benchmark_hashes(model_cfg)
            fallback_generator = None
            if proposal_source in {"aif_cem", "mixture", "probe_router", "oracle_source_selector"}:
                fallback_cfg = _model_config({**cfg, "split": split, "preset": preset, "H": H, "K": K}, fallback_model)
                fallback_generator = get_generator(fallback_model, fallback_cfg)
        except Exception as exc:  # noqa: BLE001 - benchmark should isolate model failures.
            failures.append({"model": model_name, "stage": "load", "error": repr(exc)})
            continue
        cached_env: TiltedBoardEnvAdapter | None = None
        cached_scene_id: int | None = None
        for scene_id in scene_ids:
            for seed in seeds:
                campaign_scene = None if evaluation_manifest is None else evaluation_manifest.scene(int(scene_id))
                episode_is_ood = bool(
                    is_ood_run
                    or (isinstance(campaign_scene, PrimaryScene) and campaign_scene.population != "nominal")
                )
                scene_physics_params = dict(cfg.get("physics_params") or {})
                scene_sim_params = dict(cfg.get("sim_params") or {})
                if campaign_scene is not None:
                    scene_physics_params.update(campaign_scene.physics_params)
                    scene_sim_params.update(campaign_scene.sim_params)
                training_seed = -1 if cfg.get("training_seed") is None else int(cfg["training_seed"])
                run_id = (
                    f"{model_name}:{benchmark_mode}:train{training_seed}:"
                    f"scene{int(scene_id)}:seed{int(seed)}"
                )
                if cached_env is None or cached_scene_id != int(scene_id):
                    cached_env = TiltedBoardEnvAdapter(
                        preset=preset,
                        split=split,
                        physics_backend=str(cfg.get("physics_backend", "mujoco_rigid")),
                        physics_params=scene_physics_params,
                        sim_params=scene_sim_params,
                        dataset_root=cfg.get("dataset_root"),
                    )
                    cached_scene_id = int(scene_id)
                env = cached_env
                try:
                    if campaign_scene is None:
                        reset_tilt = _tilt_override_for_run(cfg, scene_id=int(scene_id), seed=int(seed))
                        obs = env.reset(seed=int(seed), scene_id=int(scene_id), tilt=reset_tilt)
                    else:
                        obs = env.reset_scene(
                            campaign_scene.to_scene_spec(),
                            scene_id=campaign_scene.scene_id,
                            family_id=campaign_scene.family_id,
                        )
                    scene_context = env.get_scene_context()
                    true_tilt = np.asarray(scene_context.get("tilt", obs[12:14]), dtype=np.float32).reshape(2)
                    tilt_belief = HiddenTiltBelief.hidden_uniform() if bool(cfg.get("hidden_tilt_enabled", False)) else None
                    if tilt_belief is not None:
                        model_cfg["tilt_replacement"] = _posterior_tilt_replacement(tilt_belief, str(cfg.get("tilt_context_mode", "masked")))
                        model_cfg["tilt_belief"] = tilt_belief
                    belief = BeliefState.from_observation(obs, env.get_scene_context(), model_cfg)
                    gate = make_reliability_gate({**cfg, "seed": int(seed)}) if bool(cfg.get("gate_enabled", False)) else None
                    active_generator = generator
                    if proposal_source == "aif_cem":
                        active_generator = fallback_generator
                    elif proposal_source == "mixture":
                        if fallback_generator is None:
                            raise RuntimeError("proposal_source='mixture' requires a fallback generator.")
                        omega = float(gate.omega_t if gate is not None else cfg.get("omega_init", cfg.get("omega_t", 1.0)))
                        active_generator = ProposalMixtureGenerator(
                            generator,
                            fallback_generator,
                            omega_t=omega,
                            omega_min=float(cfg.get("omega_min", 0.0)),
                            omega_max=float(cfg.get("omega_max", 1.0)),
                            rho_min=float(cfg.get("rho_min", 0.0)),
                            rho_max=float(cfg.get("rho_max", 1.0)),
                            benchmark_mode=benchmark_mode,
                        )
                    elif proposal_source == "probe_router":
                        if fallback_generator is None:
                            raise RuntimeError("proposal_source='probe_router' requires a fallback generator.")
                        routing_path = cfg.get("routing_model_path")
                        routing_model = (
                            LinearRoutingModel.load(routing_path)
                            if routing_path
                            else LinearRoutingModel.default(
                                rho_min=float(cfg.get("rho_min", 0.05)),
                                rho_max=float(cfg.get("rho_max", 0.95)),
                            )
                        )
                        active_generator = ProbeAndRouteGenerator(
                            generator,
                            fallback_generator,
                            routing_model=routing_model,
                            probe_per_source=int(cfg.get("router_probe_per_source", 8)),
                            risk_mode=str(cfg.get("router_risk_mode", "cvar")),
                            cvar_alpha=float(cfg.get("router_cvar_alpha", 0.25)),
                            risk_beta=float(cfg.get("router_risk_beta", 1.0)),
                            score_scale=float(cfg.get("router_score_scale", 50.0)),
                            prediction_error_scale=float(cfg.get("router_prediction_error_scale", 1.0)),
                            belief_kl_scale=float(cfg.get("router_belief_kl_scale", 1.0)),
                            regret_scale=float(cfg.get("router_regret_scale", 50.0)),
                            collect_oracle_labels=bool(cfg.get("router_collect_oracle_labels", False)),
                            oracle_candidate_count=cfg.get("router_oracle_candidate_count"),
                        )
                    elif proposal_source == "oracle_source_selector":
                        if fallback_generator is None:
                            raise RuntimeError("proposal_source='oracle_source_selector' requires a fallback generator.")
                        active_generator = OracleSourceSelectorGenerator(
                            [generator, fallback_generator],
                            source_labels=["generator", "fallback"],
                        )
                    planner = AIFPlanner(
                        K=K,
                        H=H,
                        gamma_t=gamma_t,
                        seed=int(seed),
                        config={**model_cfg, "use_generator_context": True},
                    )
                    actions: list[np.ndarray] = []
                    trajectory = [obs[:2].copy()]
                    plan_metric_rows: list[dict[str, float]] = []
                    times = {"planning": [], "proposal": [], "scoring": [], "step_wall": []}
                    done = False
                    info: dict[str, Any] = {}
                    limit = int(max_steps if max_steps is not None else env.preset.sim.max_steps)
                    step_index = 0
                    while not done and step_index < limit:
                        step_started = time.perf_counter()
                        if isinstance(active_generator, ProposalMixtureGenerator):
                            active_generator.omega_t = float(gate.omega_t if gate is not None else cfg.get("omega_init", cfg.get("omega_t", 1.0)))
                        previous_snapshot = env.clone_state()
                        plan = planner.plan(obs, belief, active_generator, env)
                        if model_name not in first_plans:
                            first_plans[model_name] = plan
                            if bool(cfg.get("save_visuals", True)):
                                plot_topk_rollout_overlay(plan, output_dir / "plots" / f"{model_name}_topk_overlay.png", title=f"{model_name} top-K")
                        action = planner.select_action(plan)
                        selected_score = plan.score_breakdowns[plan.selected_index]
                        metrics = proposal_metrics(plan)
                        source_metrics = _source_diagnostics(plan)
                        plan_metric_rows.append(metrics)
                        gate_diag = gate.diagnostics() if gate is not None else {
                            "omega_t": float(cfg.get("omega_init", cfg.get("omega_t", 1.0))),
                            "dbar_t": float("nan"),
                            "ell_t": float("nan"),
                        }
                        gate_omega_before = float(gate_diag.get("omega_t", np.nan))
                        gate_rho_before = _proposal_rho_for_source(cfg, proposal_source, gate_omega_before) if np.isfinite(gate_omega_before) else float("nan")
                        source_rho = _float_or_nan(source_metrics.get("rho_t"))
                        tilt_diag = tilt_belief.diagnostics(true_tilt=true_tilt) if tilt_belief is not None else {}
                        generated_diag = plan.diagnostics.get("generated_state_consistency", {}) if getattr(plan, "diagnostics", None) else {}
                        hidden_score_diag = plan.diagnostics.get("hidden_tilt_score_diagnostics", {}) if getattr(plan, "diagnostics", None) else {}
                        campaign_fields = _campaign_row_fields(campaign_scene, evaluation_manifest, cfg)
                        step_row = {
                            "model": model_name,
                            "scene_id": int(scene_id),
                            "seed": int(seed),
                            "step": int(step_index),
                            "experiment": experiment_name,
                            "variant": variant_name,
                            "benchmark_mode": benchmark_mode,
                            "ood": episode_is_ood,
                            "K": int(plan.candidates.shape[0]),
                            "H": H,
                            "selected_index": int(plan.selected_index),
                            "selected_G_total": float(selected_score.G_total),
                            "posterior_entropy": float(plan.posterior_entropy),
                            "proposal_time": float(plan.proposal_time),
                            "rollout_scoring_time": float(plan.rollout_scoring_time),
                            "planning_time": float(plan.planning_time),
                            "scoring_mode": str(plan.diagnostics.get("scoring_mode", "single_rollout")) if getattr(plan, "diagnostics", None) else "single_rollout",
                            "selection_mode": str(plan.diagnostics.get("selection_mode", selection_mode)) if getattr(plan, "diagnostics", None) else selection_mode,
                            "aif_score_used_for_selection": bool(plan.diagnostics.get("aif_score_used_for_selection", True)) if getattr(plan, "diagnostics", None) else True,
                            "hidden_tilt_hypothesis_count": int(len(hidden_score_diag.get("hypothesis_indices", []))),
                            "hidden_tilt_true_tilt_used_for_scoring": bool(hidden_score_diag.get("true_tilt_used_for_scoring", False)),
                            "proposal_source": proposal_source,
                            "selected_source": source_metrics["selected_source"],
                            "omega_t": float(source_metrics.get("omega_t", gate_diag.get("omega_t", np.nan))),
                            "rho_t": source_rho if np.isfinite(source_rho) else gate_rho_before,
                            "gate_omega_before": gate_omega_before,
                            "gate_rho_before": gate_rho_before,
                            "gate_dbar_before": float(gate_diag.get("dbar_t", np.nan)),
                            "gate_ell_before": float(gate_diag.get("ell_t", np.nan)),
                            "tilt_entropy": float(tilt_diag.get("tilt_entropy", np.nan)),
                            "true_tilt_prob": float(tilt_diag.get("true_tilt_prob", np.nan)),
                            "true_tilt_lateral": float(true_tilt[0]),
                            "true_tilt_longitudinal": float(true_tilt[1]),
                            "tilt_map_lateral": float(tilt_diag.get("MAP_tilt", [np.nan, np.nan])[0]) if tilt_diag else float("nan"),
                            "tilt_map_longitudinal": float(tilt_diag.get("MAP_tilt", [np.nan, np.nan])[1]) if tilt_diag else float("nan"),
                            "action_0": float(action[0]),
                            "action_1": float(action[1]),
                            "action_2": float(action[2]),
                            "generated_state_available": bool(generated_diag.get("generated_state_available", False)),
                            "generated_state_feature_count": int(generated_diag.get("generated_state_feature_count", 0)),
                            "selected_generated_state_available": bool(generated_diag.get("selected_generated_state_available", False)),
                            "selected_Xi": _float_or_nan(generated_diag.get("selected_Xi")),
                            "mean_candidate_Xi": _float_or_nan(generated_diag.get("mean_candidate_Xi")),
                            "min_candidate_Xi": _float_or_nan(generated_diag.get("min_candidate_Xi")),
                            "max_candidate_Xi": _float_or_nan(generated_diag.get("max_candidate_Xi")),
                            "H_step_shadow_error": _float_or_nan(generated_diag.get("H_step_shadow_error")),
                            "one_step_shadow_error": _float_or_nan(generated_diag.get("one_step_shadow_error")),
                            **campaign_fields,
                            **{f"metric_{key}": value for key, value in metrics.items()},
                            **{f"source_{key}": value for key, value in source_metrics.items()},
                        }
                        times["planning"].append(plan.planning_time)
                        times["proposal"].append(plan.proposal_time)
                        times["scoring"].append(plan.rollout_scoring_time)
                        obs_next, _, done, info = env.step(action)
                        one_step_generated_error = _one_step_generated_prediction_error(plan, obs_next)
                        step_row["one_step_generated_prediction_error"] = one_step_generated_error
                        tilt_update_diag = {}
                        if tilt_belief is not None:
                            tilt_update_diag = tilt_belief.update(
                                env_adapter=env,
                                previous_snapshot=previous_snapshot,
                                action=action,
                                observed_next_obs=obs_next,
                                true_tilt=true_tilt,
                            )
                            belief.tilt_replacement = _posterior_tilt_replacement(tilt_belief, str(cfg.get("tilt_context_mode", "posterior_mean")))
                            model_cfg["tilt_replacement"] = belief.tilt_replacement
                        gate_signals = _build_gate_signals(
                            cfg,
                            metrics=metrics,
                            source_metrics=source_metrics,
                            generated_diag=generated_diag,
                            one_step_generated_prediction_error=one_step_generated_error,
                            tilt_update_diag=tilt_update_diag,
                            is_ood_run=is_ood_run,
                        )
                        if isinstance(active_generator, ProbeAndRouteGenerator):
                            active_generator.update_lagged_signals(
                                prediction_error=one_step_generated_error,
                                belief_kl=_float_or_nan(tilt_update_diag.get("belief_update_kl")),
                                primary_regret=_float_or_nan(source_metrics.get("generator_regret_vs_fallback")),
                            )
                        for key, value in gate_signals.items():
                            if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                                step_row[f"gate_{key}"] = bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
                        if gate is not None:
                            gate.update(gate_signals)
                            gate_after_diag = gate.diagnostics()
                            gate_omega_after = float(gate_after_diag.get("omega_t", np.nan))
                            step_row["gate_omega_after"] = gate_omega_after
                            step_row["gate_rho_after"] = _proposal_rho_for_source(cfg, proposal_source, gate_omega_after) if np.isfinite(gate_omega_after) else float("nan")
                            step_row["gate_dbar_after"] = float(gate_after_diag.get("dbar_t", np.nan))
                            step_row["gate_ell_after"] = float(gate_after_diag.get("ell_t", np.nan))
                            step_row["gate_omega_delta_after_update"] = gate_omega_after - gate_omega_before
                        else:
                            step_row["gate_omega_after"] = gate_omega_before
                            step_row["gate_rho_after"] = gate_rho_before
                            step_row["gate_dbar_after"] = float(gate_diag.get("dbar_t", np.nan))
                            step_row["gate_ell_after"] = float(gate_diag.get("ell_t", np.nan))
                            step_row["gate_omega_delta_after_update"] = float("nan")
                        step_wall = time.perf_counter() - step_started
                        times["step_wall"].append(float(step_wall))
                        step_row["step_wall_clock_time"] = float(step_wall)
                        if record_per_step:
                            if record_per_step_fields:
                                missing_step_fields = [field for field in record_per_step_fields if field not in step_row]
                                if missing_step_fields:
                                    raise ValueError(f"Requested per-step fields are missing: {missing_step_fields}")
                                per_step.append({field: step_row[field] for field in record_per_step_fields})
                            else:
                                per_step.append(step_row)
                        belief.update_after_observation(obs_next, action, {**env.get_scene_context(), **info})
                        actions.append(action.copy())
                        trajectory.append(obs_next[:2].copy())
                        obs = obs_next
                        step_index += 1
                    episode = _episode_summary(actions, trajectory, obs, info, env, plan_metric_rows, times, step_limit=limit)
                    episode.update({
                        "run_id": run_id,
                        "model": model_name,
                        "experiment": experiment_name,
                        "variant": variant_name,
                        "scene_id": int(scene_id),
                        "seed": int(seed),
                        "K": int(K),
                        "H": int(H),
                        "benchmark_mode": benchmark_mode,
                        "physics_backend": env.physics_backend,
                        "ood": episode_is_ood,
                        "true_tilt_lateral": float(true_tilt[0]),
                        "true_tilt_longitudinal": float(true_tilt[1]),
                        "status": "ok",
                        **_campaign_row_fields(campaign_scene, evaluation_manifest, cfg),
                    })
                    per_episode.append(episode)
                    if record_trajectories:
                        trajectory_records.append(
                            {
                                "trajectory_xy": np.asarray(trajectory, dtype=np.float32),
                                "actions": np.asarray(actions, dtype=np.float32)
                                if actions
                                else np.zeros((0, env.action_dim), dtype=np.float32),
                                "goal_xy": np.asarray(obs[7:9], dtype=np.float32),
                                "obstacle_xy": np.asarray(obs[9:11], dtype=np.float32),
                                "obstacle_radius": float(obs[11]),
                                "run_id": run_id,
                                "model": model_name,
                                "experiment": experiment_name,
                                "variant": variant_name,
                                "benchmark_mode": benchmark_mode,
                                "split": split,
                                "preset": preset,
                                "scene_id": int(scene_id),
                                "seed": int(seed),
                                "candidate_budget": int(K),
                                "physics_backend": env.physics_backend,
                                "ood": episode_is_ood,
                                "true_tilt_lateral": float(true_tilt[0]),
                                "true_tilt_longitudinal": float(true_tilt[1]),
                                "success": bool(episode["success"]),
                                "collision": bool(episode["collision"]),
                                "fall_out": bool(episode.get("fall_out", False)),
                                "timeout": bool(episode["timeout"]),
                                "terminal_reason": str(episode.get("terminal_reason", "")),
                                "final_distance_to_goal": float(episode["final_distance_to_goal"]),
                                "episode_length": int(episode["episode_length"]),
                                **_campaign_row_fields(campaign_scene, evaluation_manifest, cfg),
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - keep other models/scenes running.
                    failures.append({"model": model_name, "scene_id": int(scene_id), "seed": int(seed), "stage": "episode", "error": repr(exc)})
                    per_episode.append({
                        "model": model_name,
                        "experiment": experiment_name,
                        "variant": variant_name,
                        "scene_id": int(scene_id),
                        "seed": int(seed),
                        "K": int(K),
                        "H": int(H),
                        "benchmark_mode": benchmark_mode,
                        "ood": episode_is_ood,
                        "status": "failed",
                        "error": repr(exc),
                        **_campaign_row_fields(campaign_scene, evaluation_manifest, cfg),
                    })

    per_step_path = output_dir / "per_step.csv"
    per_episode_path = output_dir / "per_episode.csv"
    aggregate_path = output_dir / "aggregate_metrics.json"
    config_hashes_path = output_dir / "config_hashes.json"
    episode_trajectories_path = output_dir / "episode_trajectories.npz"
    model_card_path = output_dir / "model_card.json"
    failures_path = output_dir / "failures.json"
    report_path = output_dir / "summary_report.md"
    _write_csv(per_step_path, per_step)
    _write_csv(per_episode_path, per_episode)
    np.savez_compressed(episode_trajectories_path, **_trajectory_artifacts(trajectory_records, action_dim=3))
    aggregate = _aggregate(per_episode, failures, models)
    aggregate.update(
        {
            "experiment": experiment_name,
            "variant": variant_name,
            "benchmark_mode": benchmark_mode,
            "physics_backend": str(cfg.get("physics_backend", "mujoco_rigid")),
            "ood": is_ood_run,
            "campaign_split": "" if evaluation_manifest is None else evaluation_manifest.split,
            "campaign_manifest_hash": "" if evaluation_manifest is None else evaluation_manifest.manifest_hash,
            "training_seed": -1 if cfg.get("training_seed") is None else int(cfg["training_seed"]),
            "checkpoint_id": str(cfg.get("checkpoint_id", "")),
        }
    )
    _write_json(aggregate_path, aggregate)
    if hash_records:
        save_benchmark_hashes(hash_records, config_hashes_path)
    else:
        _write_json(config_hashes_path, {})
    _write_json(
        model_card_path,
        {
            "models": model_cards,
            "benchmark_mode": benchmark_mode,
            "physics_backend": str(cfg.get("physics_backend", "mujoco_rigid")),
            "physics_params": dict(cfg.get("physics_params") or {}),
            "sim_params": dict(cfg.get("sim_params") or {}),
            "campaign_manifest_path": None if campaign_manifest_path is None else str(campaign_manifest_path),
            "primary_manifest_path": None if primary_manifest_path is None else str(primary_manifest_path),
            "campaign_manifest_hash": None if evaluation_manifest is None else evaluation_manifest.manifest_hash,
            "campaign_split": None if evaluation_manifest is None else evaluation_manifest.split,
            "training_seed": -1 if cfg.get("training_seed") is None else int(cfg["training_seed"]),
            "checkpoint_id": str(cfg.get("checkpoint_id", "")),
        },
    )
    if first_plans and bool(cfg.get("save_visuals", True)):
        plot_proposal_cloud_comparison(first_plans, output_dir / "plots" / "proposal_cloud_comparison.png")
        plot_posterior_selected_comparison(first_plans, output_dir / "plots" / "posterior_selected_comparison.png")
        plot_runtime_pareto(aggregate, output_dir / "plots" / "runtime_pareto.png")
        plot_diversity_quality(aggregate, output_dir / "plots" / "diversity_quality.png")
    _write_json(failures_path, {"failures": failures})
    report_lines = ["# Generator Benchmark Report", "", f"Models: {', '.join(models)}", f"Episodes: {len(per_episode)}", f"Failures: {len(failures)}"]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "per_step_path": str(per_step_path),
        "per_episode_path": str(per_episode_path),
        "episode_trajectories_path": str(episode_trajectories_path),
        "aggregate_metrics_path": str(aggregate_path),
        "config_hashes_path": str(config_hashes_path),
        "model_card_path": str(model_card_path),
        "failures_path": str(failures_path),
        "report_path": str(report_path),
        "aggregate": aggregate,
        "episodes": per_episode,
        "failures": failures,
        "campaign_manifest_path": None if campaign_manifest_path is None else str(campaign_manifest_path),
        "primary_manifest_path": None if primary_manifest_path is None else str(primary_manifest_path),
        "campaign_manifest_hash": None if evaluation_manifest is None else evaluation_manifest.manifest_hash,
    }
