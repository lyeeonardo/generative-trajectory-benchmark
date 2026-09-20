"""Small training schedules shared by Generator trainers."""

from __future__ import annotations


def linear_warmup(final_value: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(final_value)
    return float(final_value) * min(max(int(step), 0) / float(warmup_steps), 1.0)
