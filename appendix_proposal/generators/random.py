"""Random-shooting proposal baseline for AIF scoring."""

from __future__ import annotations

import time

import numpy as np

from generators.base import ProposalBatch, ProposalGenerator


class RandomShootingGenerator(ProposalGenerator):
    """Samples random action chunks; AIFPlanner handles rollout, scoring, and selection."""

    def __init__(self, *, action_std: tuple[float, float, float] = (0.35, 0.35, 1.5)) -> None:
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.name = "random_shooting_aif"
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
        std = np.broadcast_to(self.action_std.reshape(1, 1, action_dim), (K, H, action_dim))
        actions = rng.normal(0.0, std, size=(K, H, action_dim)).astype(np.float32)
        actions = np.clip(actions, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
        elapsed = time.perf_counter() - started
        return ProposalBatch(actions=actions, diagnostics={"proposal_time": elapsed}, sample_time_sec=elapsed)
