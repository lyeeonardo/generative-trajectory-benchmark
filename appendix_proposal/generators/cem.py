"""CEM-AIF proposal baseline using the shared AIF scorer."""

from __future__ import annotations

import time

import numpy as np

from generators.base import ProposalBatch, ProposalGenerator


class CEMGenerator(ProposalGenerator):
    def __init__(
        self,
        *,
        iterations: int = 2,
        population: int = 64,
        elites: int = 8,
        elite_frac: float | None = None,
        init_std: tuple[float, float, float] = (0.25, 0.25, 1.25),
        min_std: tuple[float, float, float] = (0.02, 0.02, 0.10),
        smoothing_alpha: float = 1.0,
        action_squash: str = "clip",
        risk_mode: str = "mean",
        cvar_alpha: float = 0.25,
        risk_beta: float = 1.0,
    ) -> None:
        self.iterations = int(iterations)
        self.population = int(population)
        self.elite_frac = None if elite_frac is None else float(elite_frac)
        self.elites = int(elites)
        self.init_std = np.asarray(init_std, dtype=np.float32)
        self.min_std = np.asarray(min_std, dtype=np.float32)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.action_squash = str(action_squash)
        self.risk_mode = str(risk_mode)
        self.cvar_alpha = float(cvar_alpha)
        self.risk_beta = float(risk_beta)
        if self.risk_mode not in {"mean", "worst", "cvar", "mean_std"}:
            raise ValueError("risk_mode must be mean, worst, cvar, or mean_std")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must lie in (0, 1]")
        self.name = "cem_aif"
        self.supports_log_prob = False
        self.supports_guidance = False
        self.is_learned = False
        self.is_stochastic = True

    def propose(
        self,
        context: np.ndarray,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        del context
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        low, high = action_bounds
        std = np.broadcast_to(self.init_std.reshape(1, 1, action_dim), (K, H, action_dim)).copy()
        mean = np.zeros((K, H, action_dim), dtype=np.float32)
        actions = rng.normal(mean, std).astype(np.float32)
        actions = np.clip(actions, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
        elapsed = time.perf_counter() - started
        return ProposalBatch(actions=actions, diagnostics={"proposal_time": elapsed}, sample_time_sec=elapsed)

    def propose_with_aif(
        self,
        context: np.ndarray,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        *,
        scorer,
        belief,
        env_adapter,
        seed: int | None = None,
    ) -> ProposalBatch:
        del context
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        low, high = action_bounds
        population = max(self.population, K)
        elites = int(np.ceil(population * self.elite_frac)) if self.elite_frac is not None else self.elites
        elites = max(1, min(elites, population))
        mean = np.zeros((H, action_dim), dtype=np.float32)
        std = np.broadcast_to(self.init_std.reshape(1, action_dim), (H, action_dim)).copy()
        min_std = np.broadcast_to(self.min_std.reshape(1, action_dim), (H, action_dim)).copy()
        best_actions: np.ndarray | None = None
        best_scores: np.ndarray | None = None
        elite_score_mean = 0.0
        elite_score_std = 0.0
        covariance_trace = float(np.sum(std * std))
        action_distribution_entropy = 0.0
        initial_best_score: float | None = None
        iterations = max(self.iterations, 1)
        for iteration_index in range(iterations):
            samples = rng.normal(mean[None, :, :], std[None, :, :], size=(population, H, action_dim)).astype(np.float32)
            if self.action_squash == "tanh":
                unit = np.tanh(samples)
                samples = low.reshape(1, 1, -1) + 0.5 * (unit + 1.0) * (high - low).reshape(1, 1, -1)
            else:
                samples = np.clip(samples, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
            snapshot = env_adapter.clone_state()
            scorer_config = getattr(scorer, "config", {}) or {}
            tilt_belief = scorer_config.get("tilt_belief") if bool(scorer_config.get("hidden_tilt_enabled", False)) else None
            if tilt_belief is not None:
                from aif.tilt_hypothesis_bank import evaluate_candidates_tilt_distribution

                score_matrix, hypothesis_probabilities, hidden_tilt_diag = evaluate_candidates_tilt_distribution(
                    candidates=samples,
                    env_adapter=env_adapter,
                    scorer=scorer,
                    belief=belief,
                    tilt_belief=tilt_belief,
                    snapshot=snapshot,
                    mode=str(scorer_config.get("hidden_tilt_rollout_mode", "exact_grid")),
                    m=scorer_config.get("hidden_tilt_top_m"),
                    seed=seed,
                    score_config=scorer_config,
                )
                scores = _aggregate_tilt_risk(
                    score_matrix,
                    hypothesis_probabilities,
                    mode=self.risk_mode,
                    cvar_alpha=self.cvar_alpha,
                    risk_beta=self.risk_beta,
                )
            else:
                hidden_tilt_diag = {}
                rollouts = [env_adapter.rollout_from_state(snapshot, candidate) for candidate in samples]
                scores = np.asarray([scorer.score_rollout(rollout, belief).G_total for rollout in rollouts], dtype=np.float32)
            if iteration_index == 0:
                initial_best_score = float(np.min(scores))
            elite_indices = np.argsort(scores)[:elites]
            elites_stack = samples[elite_indices]
            elite_scores = scores[elite_indices]
            elite_mean = elites_stack.mean(axis=0)
            elite_std = np.maximum(elites_stack.std(axis=0), min_std)
            alpha = self.smoothing_alpha
            mean = ((1.0 - alpha) * mean + alpha * elite_mean).astype(np.float32)
            std = np.maximum(((1.0 - alpha) * std + alpha * elite_std).astype(np.float32), min_std)
            selected = np.argsort(scores)[:K]
            best_actions = samples[selected]
            best_scores = scores[selected]
            elite_score_mean = float(np.mean(elite_scores))
            elite_score_std = float(np.std(elite_scores))
            covariance_trace = float(np.sum(std * std))
            action_distribution_entropy = float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * np.maximum(std * std, 1e-8))))
        assert best_actions is not None
        final_best_score = float(np.min(best_scores)) if best_scores is not None else float("inf")
        diagnostics = {
            "proposal_time": time.perf_counter() - started,
            "cem_iterations": iterations,
            "cem_population": population,
            "cem_elites": elites,
            "cem_elite_frac": float(elites / population),
            "elite_score_mean": elite_score_mean,
            "elite_score_std": elite_score_std,
            "covariance_trace": covariance_trace,
            "action_distribution_entropy": action_distribution_entropy,
            "initial_best_score": initial_best_score,
            "final_best_score": final_best_score,
            "shared_scorer_used": True,
            "hidden_tilt_expected_scoring": bool((getattr(scorer, "config", {}) or {}).get("hidden_tilt_enabled", False)),
            "hidden_tilt_hypothesis_count": int(len(hidden_tilt_diag.get("hypothesis_indices", []))) if "hidden_tilt_diag" in locals() else 0,
            "true_tilt_used_for_scoring": False if bool((getattr(scorer, "config", {}) or {}).get("hidden_tilt_enabled", False)) else None,
            "risk_mode": self.risk_mode,
            "cvar_alpha": self.cvar_alpha,
            "risk_beta": self.risk_beta,
        }
        return ProposalBatch(actions=best_actions.astype(np.float32), model_score=best_scores, raw_model_outputs=best_scores, diagnostics=diagnostics, sample_time_sec=float(diagnostics["proposal_time"]))


def _aggregate_tilt_risk(
    score_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    mode: str,
    cvar_alpha: float,
    risk_beta: float,
) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if scores.ndim != 2 or scores.shape[1] != probs.shape[0]:
        raise ValueError("score_matrix and probabilities have incompatible shapes")
    probs = probs / max(float(np.sum(probs)), 1e-12)
    mean = scores @ probs
    if mode == "mean":
        return mean.astype(np.float32)
    if mode == "worst":
        return np.max(scores, axis=1).astype(np.float32)
    if mode == "mean_std":
        variance = np.sum(probs[None, :] * (scores - mean[:, None]) ** 2, axis=1)
        return (mean + float(risk_beta) * np.sqrt(np.maximum(variance, 0.0))).astype(np.float32)
    if mode != "cvar":
        raise ValueError(f"Unknown risk mode {mode!r}")
    alpha = float(cvar_alpha)
    result = np.zeros((scores.shape[0],), dtype=np.float64)
    for row_index, row in enumerate(scores):
        order = np.argsort(-row)
        remaining = alpha
        weighted = 0.0
        for column in order:
            mass = min(float(probs[column]), remaining)
            weighted += mass * float(row[column])
            remaining -= mass
            if remaining <= 1e-12:
                break
        result[row_index] = weighted / alpha
    return result.astype(np.float32)
