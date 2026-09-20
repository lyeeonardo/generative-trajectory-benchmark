"""Reliability-weighted proposal mixtures for Experiment."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from generators.base import ProposalBatch, ProposalGenerator


def rho_from_omega(
    omega_t: float,
    omega_min: float = 0.0,
    omega_max: float = 1.0,
    *,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
) -> float:
    denom = max(float(omega_max) - float(omega_min), 1e-12)
    normalized = float(np.clip((float(omega_t) - float(omega_min)) / denom, 0.0, 1.0))
    return float(float(rho_min) + (float(rho_max) - float(rho_min)) * normalized)


def allocate_equal_k(K: int, rho_t: float) -> tuple[int, int]:
    total = int(K)
    primary = int(np.floor(np.clip(float(rho_t), 0.0, 1.0) * total))
    fallback = total - primary
    return primary, fallback


def proposal_mixture_entropy(sources: list[str] | np.ndarray) -> float:
    labels = np.asarray(sources, dtype=object).reshape(-1)
    if labels.size == 0:
        return 0.0
    entropy = 0.0
    for label in sorted(set(labels.tolist())):
        p = float(np.mean(labels == label))
        entropy -= p * float(np.log(max(p, 1e-12)))
    return float(entropy)


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
) -> ProposalBatch | None:
    if int(count) <= 0:
        return None
    if scorer is not None and hasattr(generator, "propose_with_aif"):
        return generator.propose_with_aif(
            context,
            int(count),
            int(H),
            int(action_dim),
            action_bounds,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
            seed=seed,
        )
    return generator.propose(context, int(count), int(H), int(action_dim), action_bounds, seed=seed)


def mix_proposals(
    *,
    generator: ProposalGenerator,
    fallback_generator: ProposalGenerator,
    context,
    K: int,
    H: int,
    action_dim: int,
    action_bounds: tuple[np.ndarray, np.ndarray],
    omega_t: float,
    omega_min: float = 0.0,
    omega_max: float = 1.0,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    benchmark_mode: str = "equal_k",
    seed: int | None = None,
    scorer=None,
    belief=None,
    env_adapter=None,
) -> ProposalBatch:
    """Return exactly K mixed candidates in equal-K mode."""

    started = time.perf_counter()
    rho_t = rho_from_omega(omega_t, omega_min=omega_min, omega_max=omega_max, rho_min=rho_min, rho_max=rho_max)
    K_generator, K_fallback = allocate_equal_k(K, rho_t)
    if str(benchmark_mode) == "equal_time":
        # Smoke implementation uses the same deterministic allocation while
        # recording that candidate counts are actual, measured outputs.
        K_generator, K_fallback = allocate_equal_k(K, rho_t)
    elif str(benchmark_mode) != "equal_k":
        raise ValueError("benchmark_mode must be equal_k or equal_time.")
    primary = _call_generator(
        generator,
        context=context,
        count=K_generator,
        H=H,
        action_dim=action_dim,
        action_bounds=action_bounds,
        seed=seed,
        scorer=scorer,
        belief=belief,
        env_adapter=env_adapter,
    )
    fallback = _call_generator(
        fallback_generator,
        context=context,
        count=K_fallback,
        H=H,
        action_dim=action_dim,
        action_bounds=action_bounds,
        seed=None if seed is None else int(seed) + 1000003,
        scorer=scorer,
        belief=belief,
        env_adapter=env_adapter,
    )
    batch_records: list[tuple[str, np.ndarray, ProposalBatch]] = []
    observation_shape: tuple[int, int] | None = None
    for label, batch in (("generator", primary), ("fallback", fallback)):
        if batch is None:
            continue
        actions = np.asarray(batch.actions, dtype=np.float32)
        batch_records.append((label, actions, batch))
        if batch.observations is not None:
            observations = np.asarray(batch.observations, dtype=np.float32)
            if observations.ndim != 3:
                raise ValueError(f"Expected mixture observations shaped (K, H, O) or (K, H + 1, O), got {observations.shape}.")
            if observations.shape[0] != actions.shape[0]:
                raise ValueError(f"Expected observation batch size {actions.shape[0]}, got {observations.shape[0]}.")
            tail_shape = (int(observations.shape[1]), int(observations.shape[2]))
            if observation_shape is None:
                observation_shape = tail_shape
            elif observation_shape != tail_shape:
                raise ValueError(f"Mixture observation shapes must match after K axis, got {observation_shape} and {tail_shape}.")

    actions_parts = []
    observation_parts = []
    observation_mask_parts = []
    source_labels: list[str] = []
    log_prob_parts = []
    for label, actions, batch in batch_records:
        actions_parts.append(actions)
        source_labels.extend([label] * actions.shape[0])
        if batch.log_prob is not None:
            log_prob_parts.append(np.asarray(batch.log_prob, dtype=np.float32).reshape(-1))
        if observation_shape is not None:
            if batch.observations is None:
                observations = np.zeros((actions.shape[0], observation_shape[0], observation_shape[1]), dtype=np.float32)
                observation_mask = np.zeros((actions.shape[0],), dtype=bool)
            else:
                observations = np.asarray(batch.observations, dtype=np.float32)
                observation_mask = np.ones((actions.shape[0],), dtype=bool)
            observation_parts.append(observations)
            observation_mask_parts.append(observation_mask)
    actions = np.concatenate(actions_parts, axis=0) if actions_parts else np.zeros((0, H, action_dim), dtype=np.float32)
    if actions.shape[0] != int(K):
        raise ValueError(f"Proposal mixture expected {K} candidates, got {actions.shape[0]}.")
    log_prob = np.concatenate(log_prob_parts, axis=0) if log_prob_parts and sum(part.size for part in log_prob_parts) == int(K) else None
    observations = np.concatenate(observation_parts, axis=0) if observation_parts else None
    observation_mask = np.concatenate(observation_mask_parts, axis=0) if observation_mask_parts else None
    elapsed = time.perf_counter() - started
    diagnostics = {
        "omega_t": float(omega_t),
        "rho_t": float(rho_t),
        "rho_min": float(rho_min),
        "rho_max": float(rho_max),
        "benchmark_mode": str(benchmark_mode),
        "K_generator": int(K_generator),
        "K_fallback": int(K_fallback),
        "candidate_sources": source_labels,
        "best_generator_candidate_score": float("nan"),
        "best_fallback_candidate_score": float("nan"),
        "selected_source": "unselected_by_mixture",
        "proposal_mixture_entropy": proposal_mixture_entropy(source_labels),
        "proposal_time": float(elapsed),
    }
    if observation_mask is not None:
        diagnostics["generated_observation_mask"] = observation_mask.astype(bool).tolist()
        diagnostics["generated_observation_available_count"] = int(np.sum(observation_mask))
        diagnostics["generated_observation_candidate_fraction"] = float(np.mean(observation_mask)) if observation_mask.size else 0.0
    return ProposalBatch(actions=actions, observations=observations, log_prob=log_prob, diagnostics=diagnostics, sample_time_sec=elapsed)


class ProposalMixtureGenerator(ProposalGenerator):
    name = "proposal_mixture"
    supports_log_prob = False
    supports_guidance = False
    is_learned = False
    is_stochastic = True

    def __init__(
        self,
        generator: ProposalGenerator,
        fallback_generator: ProposalGenerator,
        *,
        omega_t: float = 1.0,
        omega_min: float = 0.0,
        omega_max: float = 1.0,
        rho_min: float = 0.0,
        rho_max: float = 1.0,
        benchmark_mode: str = "equal_k",
    ) -> None:
        self.generator = generator
        self.fallback_generator = fallback_generator
        self.omega_t = float(omega_t)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.benchmark_mode = str(benchmark_mode)
        self.name = f"mixture_{generator.name}_{fallback_generator.name}"

    def propose(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        return mix_proposals(
            generator=self.generator,
            fallback_generator=self.fallback_generator,
            context=context,
            K=K,
            H=H,
            action_dim=action_dim,
            action_bounds=action_bounds,
            omega_t=self.omega_t,
            omega_min=self.omega_min,
            omega_max=self.omega_max,
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            benchmark_mode=self.benchmark_mode,
            seed=seed,
        )

    def propose_with_aif(
        self,
        context,
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
        return mix_proposals(
            generator=self.generator,
            fallback_generator=self.fallback_generator,
            context=context,
            K=K,
            H=H,
            action_dim=action_dim,
            action_bounds=action_bounds,
            omega_t=self.omega_t,
            omega_min=self.omega_min,
            omega_max=self.omega_max,
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            benchmark_mode=self.benchmark_mode,
            seed=seed,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
        )

    def diagnostics(self) -> dict[str, object]:
        base = super().diagnostics()
        base.update(
            {
                "primary": self.generator.diagnostics(),
                "fallback": self.fallback_generator.diagnostics(),
                "omega_t": float(self.omega_t),
                "rho_t": rho_from_omega(
                    self.omega_t,
                    omega_min=self.omega_min,
                    omega_max=self.omega_max,
                    rho_min=self.rho_min,
                    rho_max=self.rho_max,
                ),
                "benchmark_mode": self.benchmark_mode,
            }
        )
        return base
