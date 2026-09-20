"""Common Generator proposal interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProposalBatch:
    actions: Any
    observations: Any | None = None
    log_prob: np.ndarray | None = None
    model_score: np.ndarray | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    sample_time_sec: float = 0.0
    raw_model_outputs: object | None = None

    def __post_init__(self) -> None:
        diagnostics = dict(self.diagnostics or {})
        sample_time = float(self.sample_time_sec)
        if sample_time <= 0.0:
            sample_time = float(diagnostics.get("sample_time_sec", diagnostics.get("proposal_time", 0.0)))
        diagnostics.setdefault("sample_time_sec", sample_time)
        diagnostics.setdefault("proposal_time", sample_time)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "sample_time_sec", sample_time)
        if self.model_score is None and self.raw_model_outputs is not None:
            object.__setattr__(self, "model_score", np.asarray(self.raw_model_outputs, dtype=np.float32))
        elif self.raw_model_outputs is None and self.model_score is not None:
            object.__setattr__(self, "raw_model_outputs", self.model_score)


class ProposalGenerator(abc.ABC):
    name: str = "proposal_generator"
    supports_log_prob: bool = False
    supports_guidance: bool = False
    is_learned: bool = False
    is_stochastic: bool = True

    def load(
        self,
        checkpoint_path: str | Path,
        normalization: object | None = None,
        config: dict[str, Any] | None = None,
    ) -> "ProposalGenerator":
        del checkpoint_path
        del normalization
        del config
        return self

    @abc.abstractmethod
    def propose(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        raise NotImplementedError

    def log_prob(self, context, actions) -> np.ndarray | None:
        del context
        del actions
        return None

    def diagnostics(self) -> dict[str, object]:
        return {
            "name": self.name,
            "supports_log_prob": bool(self.supports_log_prob),
            "supports_guidance": bool(self.supports_guidance),
            "is_learned": bool(self.is_learned),
            "is_stochastic": bool(self.is_stochastic),
        }
