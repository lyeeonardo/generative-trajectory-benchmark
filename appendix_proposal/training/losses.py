"""Loss helpers for the Stage 1 CVAE."""

from __future__ import annotations


def kl_warmup_beta(base_beta: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(base_beta)
    return float(base_beta) * min(max(float(step) / float(warmup_steps), 0.0), 1.0)
