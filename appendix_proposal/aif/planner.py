"""Shared receding-horizon Active Inference planner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any

import numpy as np

from aif.belief import BeliefState
from aif.rollout import rollout_action_batch
from aif.scoring import AIFScorer, ScoreBreakdown, policy_posterior
from aif.tilt_context import snapshot_with_tilt


@dataclass(frozen=True)
class PlanResult:
    candidates: np.ndarray
    rollouts: list[object]
    score_breakdowns: list[ScoreBreakdown]
    policy_posterior: np.ndarray
    selected_index: int
    selected_action: np.ndarray
    planning_time: float
    proposal_time: float
    rollout_scoring_time: float
    posterior_entropy: float
    route_distribution: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _empty_generated_state_diagnostics() -> dict[str, Any]:
    nan = float("nan")
    return {
        "generated_state_available": False,
        "generated_state_shape": None,
        "generated_state_timestep_mode": "none",
        "generated_state_feature_indices": [],
        "generated_state_feature_count": 0,
        "candidate_Xi": [],
        "selected_Xi": nan,
        "mean_candidate_Xi": nan,
        "min_candidate_Xi": nan,
        "max_candidate_Xi": nan,
        "one_step_shadow_error": nan,
        "H_step_shadow_error": nan,
        "selected_generated_one_step": None,
    }


def _generated_state_feature_indices(obs_dim: int, config: dict[str, Any] | None) -> np.ndarray:
    cfg = dict(config or {})
    configured = cfg.get("generated_state_consistency_indices", cfg.get("generated_state_feature_indices"))
    if configured is not None:
        indices = np.asarray(configured, dtype=np.int64).reshape(-1)
    elif bool(cfg.get("hidden_tilt_enabled", cfg.get("mask_tilt", False))) and not bool(cfg.get("oracle_tilt", False)):
        indices = np.arange(min(int(obs_dim), 7), dtype=np.int64)
    else:
        indices = np.arange(int(obs_dim), dtype=np.int64)
    if indices.size == 0:
        return indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= int(obs_dim)):
        raise ValueError(f"generated_state_consistency_indices must be within observation width {obs_dim}, got {indices.tolist()}.")
    return indices.astype(np.int64)


def _generated_state_consistency(
    proposal_batch,
    rollouts: list[object],
    selected_index: int,
    H: int,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated = getattr(proposal_batch, "observations", None)
    if generated is None:
        return _empty_generated_state_diagnostics()
    generated_obs = np.asarray(generated, dtype=np.float32)
    if generated_obs.ndim != 3:
        raise ValueError(f"Expected generated observations shaped (K, H, O) or (K, H + 1, O), got {generated_obs.shape}.")
    if generated_obs.shape[0] != len(rollouts) or generated_obs.shape[1] not in {int(H), int(H) + 1}:
        raise ValueError(
            f"Expected generated observations first axes {(len(rollouts), int(H))} or {(len(rollouts), int(H) + 1)}, "
            f"got {generated_obs.shape[:2]}."
        )
    proposal_diagnostics = getattr(proposal_batch, "diagnostics", {}) or {}
    raw_mask = proposal_diagnostics.get("generated_observation_mask")
    if raw_mask is None:
        generated_mask = np.ones((generated_obs.shape[0],), dtype=bool)
    else:
        generated_mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
        if generated_mask.shape[0] != generated_obs.shape[0]:
            raise ValueError(f"Expected generated_observation_mask length {generated_obs.shape[0]}, got {generated_mask.shape[0]}.")
    if not np.any(generated_mask):
        empty = _empty_generated_state_diagnostics()
        empty.update(
            {
                "generated_state_shape": list(generated_obs.shape),
                "generated_state_candidate_count": 0,
                "generated_state_candidate_fraction": 0.0,
                "selected_generated_state_available": False,
            }
        )
        return empty
    if not np.all(np.isfinite(generated_obs[generated_mask])):
        raise ValueError("Generated observations contain non-finite values for candidates marked available.")

    mode = "current_plus_future" if generated_obs.shape[1] == int(H) + 1 else "future_only"
    generated_future = generated_obs[:, 1:, :] if mode == "current_plus_future" else generated_obs
    rollout_future = [np.asarray(rollout.observations, dtype=np.float32)[1:] for rollout in rollouts]
    future_len = min([generated_future.shape[1], *(future.shape[0] for future in rollout_future)])
    obs_dim = min([generated_future.shape[2], *(future.shape[1] for future in rollout_future)])
    if future_len <= 0 or obs_dim <= 0:
        return _empty_generated_state_diagnostics()
    feature_indices = _generated_state_feature_indices(obs_dim, config)
    if feature_indices.size == 0:
        empty = _empty_generated_state_diagnostics()
        empty.update(
            {
                "generated_state_available": True,
                "generated_state_shape": list(generated_obs.shape),
                "generated_state_timestep_mode": mode,
                "generated_state_candidate_count": int(np.sum(generated_mask)),
                "generated_state_candidate_fraction": float(np.mean(generated_mask)),
                "selected_generated_state_available": bool(generated_mask[int(selected_index)]),
            }
        )
        return empty

    valid_indices = np.flatnonzero(generated_mask)
    valid_errors = np.stack(
        [
            generated_future[index, :future_len, :obs_dim][:, feature_indices]
            - rollout_future[index][:future_len, :obs_dim][:, feature_indices]
            for index in valid_indices.tolist()
        ],
        axis=0,
    )
    squared = valid_errors * valid_errors
    valid_Xi = np.sum(squared, axis=(1, 2)).astype(np.float32)
    valid_shadow_rmse = np.sqrt(np.mean(squared, axis=(1, 2))).astype(np.float32)
    Xi = np.full((len(rollouts),), np.nan, dtype=np.float32)
    shadow_rmse = np.full((len(rollouts),), np.nan, dtype=np.float32)
    Xi[valid_indices] = valid_Xi
    shadow_rmse[valid_indices] = valid_shadow_rmse
    selected = int(selected_index)
    selected_available = bool(generated_mask[selected]) if 0 <= selected < generated_mask.shape[0] else False
    selected_generated_one_step = generated_future[selected, 0, :obs_dim].astype(np.float32) if selected_available else None
    selected_error = (
        generated_future[selected, 0, :obs_dim][feature_indices] - rollout_future[selected][0, :obs_dim][feature_indices]
        if selected_available
        else np.full((feature_indices.shape[0],), np.nan, dtype=np.float32)
    )
    return {
        "generated_state_available": True,
        "generated_state_shape": list(generated_obs.shape),
        "generated_state_timestep_mode": mode,
        "generated_state_feature_indices": feature_indices.astype(int).tolist(),
        "generated_state_feature_count": int(feature_indices.shape[0]),
        "generated_state_candidate_count": int(np.sum(generated_mask)),
        "generated_state_candidate_fraction": float(np.mean(generated_mask)),
        "selected_generated_state_available": selected_available,
        "candidate_Xi": Xi.astype(float).tolist(),
        "selected_Xi": float(Xi[selected]),
        "mean_candidate_Xi": float(np.mean(valid_Xi)),
        "min_candidate_Xi": float(np.min(valid_Xi)),
        "max_candidate_Xi": float(np.max(valid_Xi)),
        "one_step_shadow_error": float(np.linalg.norm(selected_error)) if selected_available else float("nan"),
        "H_step_shadow_error": float(shadow_rmse[selected]),
        "selected_generated_one_step": None if selected_generated_one_step is None else selected_generated_one_step.astype(float).tolist(),
    }


def _hidden_diagnostic_tilt(belief: BeliefState, tilt_belief, mode: str) -> np.ndarray:
    key = str(mode)
    if key == "map":
        return np.asarray(tilt_belief.map_tilt, dtype=np.float32).reshape(2)
    if key == "posterior_mean":
        probs = np.asarray(tilt_belief.probabilities, dtype=np.float32).reshape(-1)
        return np.sum(tilt_belief.tilt_grid * probs[:, None], axis=0).astype(np.float32)
    return np.asarray(belief.tilt_replacement, dtype=np.float32).reshape(2)


def _select_candidate_index(
    *,
    selection_mode: str,
    policy_posterior: np.ndarray,
    proposal_log_prob: np.ndarray | None,
) -> tuple[int, bool]:
    key = str(selection_mode).strip().lower()
    if key in {"", "aif", "aif_score", "map_first_action", "posterior_map"}:
        return int(np.argmax(policy_posterior)), True
    if key in {"generator_first", "generator_only", "generator_only_first", "first_sample"}:
        return 0, False
    if key in {"generator_logprob", "generator_likelihood", "max_log_prob"}:
        if proposal_log_prob is None or proposal_log_prob.size <= 0 or not np.any(np.isfinite(proposal_log_prob)):
            return 0, False
        scores = np.where(np.isfinite(proposal_log_prob), proposal_log_prob, -np.inf)
        return int(np.argmax(scores)), False
    raise ValueError(f"Unknown selection_mode: {selection_mode!r}.")


class AIFPlanner:
    """Generator-agnostic planner that scores candidates through shared G."""

    def __init__(
        self,
        *,
        K: int = 64,
        H: int = 16,
        gamma_t: float = 2.0,
        execution_window: int = 1,
        scorer: AIFScorer | None = None,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.K = int(K)
        self.H = int(H)
        self.gamma_t = float(gamma_t)
        self.execution_window = int(execution_window)
        self.scorer = scorer or AIFScorer(config)
        self.seed = seed
        self.config = dict(config or {})
        self._replan_index = 0
        self.belief: BeliefState | None = None

    def _proposal_seed(self) -> int | None:
        if self.seed is None:
            return None
        return int(np.random.SeedSequence((int(self.seed), int(self._replan_index), 31)).generate_state(1)[0])

    def plan(self, obs: np.ndarray, belief: BeliefState, generator, env_adapter) -> PlanResult:
        started = time.perf_counter()
        if bool(self.config.get("use_generator_context", self.config.get("generator_context", False))):
            context = belief.to_generator_context(
                env_adapter.get_scene_context(),
                normalizer=self.config.get("context_normalizer"),
                history_len=int(self.config.get("history_len", 4)),
            )
        else:
            context = belief.to_context_vector()
        if bool(self.config.get("outcome_conditioning", self.config.get("append_success_outcome", False))):
            from data.archive_dataset import outcome_one_hot
            from generators.utils import context_to_vector

            outcome = str(self.config.get("control_outcome_label", "success"))
            context = np.concatenate([context_to_vector(context), outcome_one_hot(outcome)], axis=0).astype(np.float32)
        low, high = env_adapter.get_action_bounds()
        proposal_seed = self._proposal_seed()
        proposal_started = time.perf_counter()
        if hasattr(generator, "propose_with_aif"):
            proposal_batch = generator.propose_with_aif(
                context,
                self.K,
                self.H,
                env_adapter.action_dim,
                (low, high),
                scorer=self.scorer,
                belief=belief,
                env_adapter=env_adapter,
                seed=proposal_seed,
            )
        else:
            proposal_batch = generator.propose(
                context,
                self.K,
                self.H,
                env_adapter.action_dim,
                (low, high),
                seed=proposal_seed,
            )
        proposal_time = time.perf_counter() - proposal_started
        if getattr(proposal_batch, "diagnostics", None) and "proposal_time" in proposal_batch.diagnostics:
            proposal_time = float(proposal_batch.diagnostics["proposal_time"])
        candidates = np.asarray(proposal_batch.actions, dtype=np.float32)
        if candidates.shape != (self.K, self.H, env_adapter.action_dim):
            raise ValueError(f"Expected candidates {(self.K, self.H, env_adapter.action_dim)}, got {candidates.shape}.")
        candidates = np.clip(candidates, low.reshape(1, 1, -1), high.reshape(1, 1, -1)).astype(np.float32)

        rollout_started = time.perf_counter()
        snapshot = env_adapter.clone_state()
        hidden_tilt_belief = self.config.get("tilt_belief") if bool(self.config.get("hidden_tilt_enabled", False)) else None
        hidden_scoring = hidden_tilt_belief is not None
        diagnostic_tilt = None
        diagnostic_snapshot = snapshot
        if hidden_scoring:
            diagnostic_tilt = _hidden_diagnostic_tilt(
                belief,
                hidden_tilt_belief,
                str(self.config.get("hidden_diagnostic_tilt_mode", self.config.get("tilt_context_mode", "posterior_mean"))),
            )
            diagnostic_snapshot = snapshot_with_tilt(snapshot, diagnostic_tilt)
        rollouts = rollout_action_batch(env_adapter, diagnostic_snapshot, candidates)
        epistemic_diagnostics: dict[str, Any] = {}
        epistemic_values = None
        if bool(self.config.get("epistemic_enabled", False)) and hidden_tilt_belief is not None:
            from aif.epistemic import adaptive_epistemic_beta, epistemic_values_for_candidates

            values, epistemic_diagnostics = epistemic_values_for_candidates(
                candidates=candidates,
                env_adapter=env_adapter,
                tilt_belief=hidden_tilt_belief,
                snapshot=snapshot,
                approximation=str(self.config.get("epistemic_approximation", "one_step_epistemic")),
                horizon_steps=self.config.get("epistemic_horizon_steps"),
                hypothesis_mode=str(self.config.get("epistemic_tilt_mode", self.config.get("hidden_tilt_rollout_mode", "exact_grid"))),
                hypothesis_top_m=self.config.get("epistemic_top_m", self.config.get("hidden_tilt_top_m")),
                seed=proposal_seed,
            )
            epistemic_values = np.asarray(values, dtype=np.float32).reshape(-1)
            entropy = float(epistemic_diagnostics.get("tilt_entropy", 0.0))
            beta = adaptive_epistemic_beta(
                entropy,
                beta_min=float(self.config.get("epistemic_beta_min", self.config.get("beta_min", 0.0))),
                beta_max=float(self.config.get("epistemic_beta_max", self.config.get("beta_max", 1.0))),
                theta=float(self.config.get("epistemic_theta", 1.0)),
                temperature=float(self.config.get("epistemic_temperature", 0.2)),
            )
            self.config["w_epistemic"] = float(self.config.get("w_epistemic", beta))
            for rollout, value in zip(rollouts, values):
                rollout.raw_simulator_info["epistemic_value"] = float(value)
            epistemic_diagnostics["beta"] = float(beta)
        scores = self.scorer.score_batch(rollouts, belief, config=self.config)
        hidden_score_diagnostics: dict[str, Any] = {}
        if hidden_scoring:
            from aif.tilt_hypothesis_bank import evaluate_candidates_expected_tilt

            expected_G, hidden_score_diagnostics = evaluate_candidates_expected_tilt(
                candidates=candidates,
                env_adapter=env_adapter,
                scorer=self.scorer,
                belief=belief,
                tilt_belief=hidden_tilt_belief,
                snapshot=snapshot,
                mode=str(self.config.get("hidden_tilt_rollout_mode", "exact_grid")),
                m=self.config.get("hidden_tilt_top_m"),
                seed=proposal_seed,
                score_config=self.config,
                epistemic_values=epistemic_values,
            )
            live_diagnostic_G = [float(score.G_total) for score in scores]
            scores = [
                replace(
                    score,
                    G_total=float(expected),
                    diagnostics={
                        **dict(score.diagnostics),
                        "diagnostic_rollout_G_total": float(score.G_total),
                        "hidden_tilt_expected_G_total": float(expected),
                    },
                )
                for score, expected in zip(scores, expected_G)
            ]
            hidden_score_diagnostics.update(
                {
                    "scoring_mode": "hidden_tilt_expected",
                    "diagnostic_rollout_G_total": live_diagnostic_G,
                    "expected_G_total": expected_G.astype(float).tolist(),
                    "diagnostic_rollout_tilt": None if diagnostic_tilt is None else diagnostic_tilt.astype(float).tolist(),
                    "true_tilt_used_for_scoring": False,
                }
            )
        rollout_scoring_time = time.perf_counter() - rollout_started
        G = np.asarray([score.G_total for score in scores], dtype=np.float32)
        log_E = None
        raw_log_prob = getattr(proposal_batch, "log_prob", None)
        proposal_log_prob = None if raw_log_prob is None else np.asarray(raw_log_prob, dtype=np.float32).reshape(-1)
        generator_prior_used = False
        if bool(self.config.get("generator_prior_enabled", False)) and raw_log_prob is not None:
            log_E = np.asarray(raw_log_prob, dtype=np.float32).reshape(-1)
            clip_value = self.config.get("generator_prior_logprob_clip", self.config.get("log_prob_clip"))
            if clip_value is not None:
                log_E = np.clip(log_E, -float(clip_value), float(clip_value))
            prior_weight = float(self.config.get("generator_prior_weight", self.config.get("lambda_R", 1.0)))
            prior_temperature = float(self.config.get("generator_prior_temperature", self.config.get("omega_t", 1.0)))
            log_E = prior_weight * prior_temperature * log_E
            generator_prior_used = True
        Q_pi = policy_posterior(G, self.gamma_t, log_E=log_E)
        selection_mode = str(self.config.get("selection_mode", self.config.get("controller_mode", "aif_score")))
        selected_index, aif_score_used_for_selection = _select_candidate_index(
            selection_mode=selection_mode,
            policy_posterior=Q_pi,
            proposal_log_prob=proposal_log_prob,
        )
        selected_action = candidates[selected_index, 0].copy()
        generated_state_diagnostics = _generated_state_consistency(proposal_batch, rollouts, selected_index, self.H, config=self.config)
        entropy = float(-np.sum(Q_pi * np.log(np.clip(Q_pi, 1e-12, 1.0))))
        route_distribution = self._route_distribution(rollouts, Q_pi)
        planning_time = time.perf_counter() - started
        self._replan_index += 1
        return PlanResult(
            candidates=candidates,
            rollouts=rollouts,
            score_breakdowns=scores,
            policy_posterior=Q_pi,
            selected_index=selected_index,
            selected_action=selected_action.astype(np.float32),
            planning_time=float(planning_time),
            proposal_time=float(proposal_time),
            rollout_scoring_time=float(rollout_scoring_time),
            posterior_entropy=entropy,
            route_distribution=route_distribution,
            diagnostics={
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "G_total": G.tolist(),
                "selected_G_total": float(G[selected_index]),
                "proposal_seed": proposal_seed,
                "proposal_diagnostics": getattr(proposal_batch, "diagnostics", {}),
                "generator_prior_used": generator_prior_used,
                "generator_prior_enabled": bool(self.config.get("generator_prior_enabled", False)),
                "proposal_log_prob": None if proposal_log_prob is None else proposal_log_prob.tolist(),
                "selected_log_prob": None if proposal_log_prob is None else float(proposal_log_prob[selected_index]),
                "selected_route": rollouts[selected_index].route_label,
                "selection_mode": selection_mode,
                "aif_score_used_for_selection": bool(aif_score_used_for_selection),
                "epistemic_diagnostics": epistemic_diagnostics,
                "generated_state_consistency": generated_state_diagnostics,
                "scoring_mode": "hidden_tilt_expected" if hidden_scoring else "single_rollout",
                "hidden_tilt_score_diagnostics": hidden_score_diagnostics,
            },
        )

    def select_action(self, plan_result: PlanResult, mode: str = "map_first_action") -> np.ndarray:
        if mode != "map_first_action":
            raise ValueError("Stage 1 main controller supports only map_first_action.")
        return plan_result.selected_action.copy()

    def step(self, obs: np.ndarray, env_adapter, generator) -> tuple[np.ndarray, dict[str, Any]]:
        if self.belief is None:
            self.belief = BeliefState.from_observation(obs, env_adapter.get_scene_context(), self.config)
        plan = self.plan(obs, self.belief, generator, env_adapter)
        action = self.select_action(plan)
        return action, {"plan": plan}

    def _route_distribution(self, rollouts: list[object], Q_pi: np.ndarray) -> dict[str, float]:
        labels = ["left", "right", "center", "invalid"]
        route_distribution = {label: 0.0 for label in labels}
        for rollout, mass in zip(rollouts, Q_pi):
            label = rollout.route_label if rollout.route_label in route_distribution else "invalid"
            route_distribution[label] += float(mass)
        return route_distribution
