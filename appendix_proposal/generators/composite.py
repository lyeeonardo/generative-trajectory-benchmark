"""Composite proposal sources used by the gate-improvement campaign."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from generators.base import ProposalBatch, ProposalGenerator


def _call_generator(
    generator: ProposalGenerator,
    *,
    context,
    count: int,
    H: int,
    action_dim: int,
    action_bounds: tuple[np.ndarray, np.ndarray],
    seed: int | None,
    scorer=None,
    belief=None,
    env_adapter=None,
) -> ProposalBatch:
    if scorer is not None and hasattr(generator, "propose_with_aif"):
        return generator.propose_with_aif(
            context,
            count,
            H,
            action_dim,
            action_bounds,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
            seed=seed,
        )
    return generator.propose(context, count, H, action_dim, action_bounds, seed=seed)


def _batch_with_diagnostics(batch: ProposalBatch, diagnostics: dict[str, Any]) -> ProposalBatch:
    return ProposalBatch(
        actions=np.asarray(batch.actions, dtype=np.float32),
        observations=batch.observations,
        log_prob=batch.log_prob,
        model_score=batch.model_score,
        diagnostics={**dict(batch.diagnostics or {}), **diagnostics},
        sample_time_sec=float(batch.sample_time_sec),
        raw_model_outputs=batch.raw_model_outputs,
    )


def _allocate(total: int, count: int) -> list[int]:
    allocation = [total // count] * count
    for index in range(total % count):
        allocation[index] += 1
    return allocation


def _concat_batches(batches: Sequence[tuple[str, ProposalBatch]], *, elapsed: float) -> ProposalBatch:
    actions = np.concatenate([np.asarray(batch.actions, dtype=np.float32) for _, batch in batches], axis=0)
    observations = None
    if all(batch.observations is not None for _, batch in batches):
        shapes = {tuple(np.asarray(batch.observations).shape[1:]) for _, batch in batches}
        if len(shapes) == 1:
            observations = np.concatenate([np.asarray(batch.observations, dtype=np.float32) for _, batch in batches], axis=0)
    log_prob = None
    if all(batch.log_prob is not None for _, batch in batches):
        log_prob = np.concatenate([np.asarray(batch.log_prob, dtype=np.float32).reshape(-1) for _, batch in batches])
    labels = [label for label, batch in batches for _ in range(np.asarray(batch.actions).shape[0])]
    return ProposalBatch(
        actions=actions,
        observations=observations,
        log_prob=log_prob,
        diagnostics={
            "proposal_time": elapsed,
            "candidate_sources": labels,
            "source_counts": {label: int(np.asarray(batch.actions).shape[0]) for label, batch in batches},
        },
        sample_time_sec=elapsed,
    )


class EnsembleGenerator(ProposalGenerator):
    """Allocate K equally across a fixed set of proposal generators."""

    def __init__(self, generators: Sequence[ProposalGenerator], *, name: str = "ensemble_aif") -> None:
        if len(generators) < 2:
            raise ValueError("EnsembleGenerator requires at least two sources")
        self.generators = tuple(generators)
        self.name = str(name)
        self.is_learned = any(generator.is_learned for generator in generators)
        self.is_stochastic = any(generator.is_stochastic for generator in generators)

    def _propose(self, context, K, H, action_dim, action_bounds, *, seed, scorer=None, belief=None, env_adapter=None):
        started = time.perf_counter()
        batches: list[tuple[str, ProposalBatch]] = []
        for index, (generator, count) in enumerate(zip(self.generators, _allocate(int(K), len(self.generators)))):
            if count <= 0:
                continue
            source_seed = None if seed is None else int(seed) + 104_729 * index
            batch = _call_generator(
                generator,
                context=context,
                count=count,
                H=H,
                action_dim=action_dim,
                action_bounds=action_bounds,
                seed=source_seed,
                scorer=scorer,
                belief=belief,
                env_adapter=env_adapter,
            )
            batches.append((generator.name, batch))
        return _concat_batches(batches, elapsed=time.perf_counter() - started)

    def propose(self, context, K, H, action_dim, action_bounds, seed=None) -> ProposalBatch:
        return self._propose(context, K, H, action_dim, action_bounds, seed=seed)

    def propose_with_aif(self, context, K, H, action_dim, action_bounds, *, scorer, belief, env_adapter, seed=None) -> ProposalBatch:
        return self._propose(
            context,
            K,
            H,
            action_dim,
            action_bounds,
            seed=seed,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
        )

    def diagnostics(self) -> dict[str, object]:
        return {**super().diagnostics(), "sources": [generator.diagnostics() for generator in self.generators]}


def _score_pool(pool: np.ndarray, *, scorer, belief, env_adapter, seed: int | None, risk_mode: str, cvar_alpha: float, risk_beta: float) -> np.ndarray:
    snapshot = env_adapter.clone_state()
    scorer_config = getattr(scorer, "config", {}) or {}
    tilt_belief = scorer_config.get("tilt_belief") if bool(scorer_config.get("hidden_tilt_enabled", False)) else None
    if tilt_belief is None:
        rollouts = [env_adapter.rollout_from_state(snapshot, candidate) for candidate in pool]
        return np.asarray([scorer.score_rollout(rollout, belief).G_total for rollout in rollouts], dtype=np.float32)
    from aif.tilt_hypothesis_bank import evaluate_candidates_tilt_distribution
    from generators.cem import _aggregate_tilt_risk

    matrix, probabilities, _ = evaluate_candidates_tilt_distribution(
        candidates=pool,
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
    return _aggregate_tilt_risk(
        matrix,
        probabilities,
        mode=risk_mode,
        cvar_alpha=cvar_alpha,
        risk_beta=risk_beta,
    )


class RepairGenerator(ProposalGenerator):
    """AIF-score local perturbations around candidates from a learned source."""

    def __init__(
        self,
        base_generator: ProposalGenerator,
        *,
        base_fraction: float = 0.25,
        population_multiplier: float = 4.0,
        noise_std: tuple[float, float, float] = (0.08, 0.08, 0.40),
        risk_mode: str = "cvar",
        cvar_alpha: float = 0.25,
        risk_beta: float = 1.0,
    ) -> None:
        self.base_generator = base_generator
        self.base_fraction = float(base_fraction)
        self.population_multiplier = float(population_multiplier)
        self.noise_std = np.asarray(noise_std, dtype=np.float32)
        self.risk_mode = str(risk_mode)
        self.cvar_alpha = float(cvar_alpha)
        self.risk_beta = float(risk_beta)
        self.name = "repair_aif"
        self.is_learned = base_generator.is_learned
        self.is_stochastic = True

    def _candidate_pool(self, context, K, H, action_dim, action_bounds, seed) -> tuple[np.ndarray, ProposalBatch]:
        rng = np.random.default_rng(seed)
        base_count = max(1, min(int(K), int(math.ceil(float(K) * self.base_fraction))))
        base = self.base_generator.propose(context, base_count, H, action_dim, action_bounds, seed=seed)
        base_actions = np.asarray(base.actions, dtype=np.float32)
        population = max(int(K), int(math.ceil(float(K) * self.population_multiplier)))
        indices = rng.integers(0, base_actions.shape[0], size=population)
        pool = base_actions[indices].copy()
        noise = rng.normal(0.0, self.noise_std.reshape(1, 1, -1), size=pool.shape).astype(np.float32)
        pool = pool + noise
        pool[: min(base_actions.shape[0], population)] = base_actions[: min(base_actions.shape[0], population)]
        low, high = action_bounds
        pool = np.clip(pool, low.reshape(1, 1, -1), high.reshape(1, 1, -1)).astype(np.float32)
        return pool, base

    def propose(self, context, K, H, action_dim, action_bounds, seed=None) -> ProposalBatch:
        started = time.perf_counter()
        pool, _ = self._candidate_pool(context, K, H, action_dim, action_bounds, seed)
        return ProposalBatch(
            actions=pool[: int(K)],
            diagnostics={"proposal_time": time.perf_counter() - started, "repair_scored": False},
        )

    def propose_with_aif(self, context, K, H, action_dim, action_bounds, *, scorer, belief, env_adapter, seed=None) -> ProposalBatch:
        started = time.perf_counter()
        pool, _ = self._candidate_pool(context, K, H, action_dim, action_bounds, seed)
        scores = _score_pool(
            pool,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
            seed=seed,
            risk_mode=self.risk_mode,
            cvar_alpha=self.cvar_alpha,
            risk_beta=self.risk_beta,
        )
        selected = np.argsort(scores)[: int(K)]
        elapsed = time.perf_counter() - started
        return ProposalBatch(
            actions=pool[selected],
            model_score=scores[selected],
            diagnostics={
                "proposal_time": elapsed,
                "repair_scored": True,
                "repair_population": int(pool.shape[0]),
                "repair_base_model": self.base_generator.name,
                "risk_mode": self.risk_mode,
                "cvar_alpha": self.cvar_alpha,
            },
            sample_time_sec=elapsed,
        )

    def diagnostics(self) -> dict[str, object]:
        return {**super().diagnostics(), "base": self.base_generator.diagnostics(), "risk_mode": self.risk_mode}


class OracleSourceSelectorGenerator(ProposalGenerator):
    """Diagnostic upper bound that chooses a source using true rollouts.

    This deliberately spends K candidates per source and is never a deployable
    comparator.  It establishes whether source complementarity is large enough
    for a learned budget-neutral router to be identifiable.
    """

    def __init__(self, generators: Sequence[ProposalGenerator], *, source_labels: Sequence[str] | None = None) -> None:
        if len(generators) < 2:
            raise ValueError("OracleSourceSelectorGenerator requires at least two sources")
        self.generators = tuple(generators)
        self.source_labels = tuple(source_labels or (generator.name for generator in generators))
        if len(self.source_labels) != len(self.generators) or len(set(self.source_labels)) != len(self.source_labels):
            raise ValueError("Oracle source labels must be unique and aligned with generators")
        self.name = "oracle_source_selector_aif"
        self.is_stochastic = any(generator.is_stochastic for generator in generators)

    @staticmethod
    def _source_key(batch: ProposalBatch, env_adapter) -> tuple[tuple[float, ...], int]:
        snapshot = env_adapter.clone_state()
        best_key: tuple[float, ...] | None = None
        best_index = 0
        for index, candidate in enumerate(np.asarray(batch.actions, dtype=np.float32)):
            rollout = env_adapter.rollout_from_state(snapshot, candidate)
            key = (
                0.0 if rollout.success else 1.0,
                1.0 if rollout.fall_out else 0.0,
                1.0 if rollout.collision else 0.0,
                float(rollout.final_distance_to_goal),
                float(rollout.path_length),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        assert best_key is not None
        return best_key, best_index

    def propose(self, context, K, H, action_dim, action_bounds, seed=None) -> ProposalBatch:
        raise RuntimeError("oracle_source_selector_aif requires scorer, belief, and environment access")

    def propose_with_aif(self, context, K, H, action_dim, action_bounds, *, scorer, belief, env_adapter, seed=None) -> ProposalBatch:
        started = time.perf_counter()
        candidates: list[tuple[tuple[float, ...], str, ProposalBatch, int]] = []
        for index, (generator, source_label) in enumerate(zip(self.generators, self.source_labels)):
            source_seed = None if seed is None else int(seed) + 1_000_003 * index
            batch = _call_generator(
                generator,
                context=context,
                count=int(K),
                H=H,
                action_dim=action_dim,
                action_bounds=action_bounds,
                seed=source_seed,
                scorer=scorer,
                belief=belief,
                env_adapter=env_adapter,
            )
            key, best_index = self._source_key(batch, env_adapter)
            candidates.append((key, source_label, batch, best_index))
        candidates.sort(key=lambda row: row[0])
        key, source_name, selected, best_index = candidates[0]
        elapsed = time.perf_counter() - started
        diagnostics = {
            "proposal_time": elapsed,
            "oracle_selected_source": source_name,
            "oracle_selected_source_best_index": int(best_index),
            "oracle_selected_source_key": list(key),
            "oracle_source_keys": {name: list(source_key) for source_key, name, _, _ in candidates},
            "oracle_source_count": len(candidates),
            "oracle_true_rollouts_used": True,
            "candidate_sources": [source_name] * int(np.asarray(selected.actions).shape[0]),
            "rho_t": 1.0 if source_name == "generator" else (0.0 if source_name == "fallback" else float("nan")),
        }
        return _batch_with_diagnostics(selected, diagnostics)

    def diagnostics(self) -> dict[str, object]:
        return {
            **super().diagnostics(),
            "sources": [generator.diagnostics() for generator in self.generators],
            "source_labels": list(self.source_labels),
        }
