"""CPU-safe prototype generators for Generator contract verification.

These classes satisfy the proposal interface for model families whose full
training algorithms are implemented in later Generator steps. They are not used to
claim model-specific benchmark performance.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from generators.base import ProposalBatch, ProposalGenerator
from generators.utils import clip_actions, context_to_vector


class ContextGaussianPrototypeGenerator(ProposalGenerator):
    def __init__(
        self,
        *,
        name: str,
        supports_log_prob: bool = False,
        supports_guidance: bool = False,
        is_learned: bool = True,
        hidden_seed: int = 0,
        base_std: tuple[float, float, float] = (0.30, 0.30, 1.25),
    ) -> None:
        self.name = str(name)
        self.supports_log_prob = bool(supports_log_prob)
        self.supports_guidance = bool(supports_guidance)
        self.is_learned = bool(is_learned)
        self.is_stochastic = True
        self.hidden_seed = int(hidden_seed)
        self.base_std = np.asarray(base_std, dtype=np.float32)
        self._projection: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    def _ensure_params(self, context_dim: int, H: int, action_dim: int) -> None:
        flat_dim = int(H) * int(action_dim)
        if self._projection is not None and self._projection.shape == (context_dim, flat_dim):
            return
        rng = np.random.default_rng(self.hidden_seed)
        self._projection = rng.normal(0.0, 0.04, size=(context_dim, flat_dim)).astype(np.float32)
        self._bias = rng.normal(0.0, 0.02, size=(flat_dim,)).astype(np.float32)

    def load(
        self,
        checkpoint_path: str | Path,
        normalization: object | None = None,
        config: dict[str, Any] | None = None,
    ) -> "ContextGaussianPrototypeGenerator":
        del normalization
        del config
        path = Path(checkpoint_path)
        if path.exists() and path.stat().st_size > 0:
            try:
                payload = np.load(path, allow_pickle=True)
                if "projection" in payload:
                    self._projection = np.asarray(payload["projection"], dtype=np.float32)
                if "bias" in payload:
                    self._bias = np.asarray(payload["bias"], dtype=np.float32)
            except Exception:
                # Later model-specific checkpoints are handled by their own classes.
                pass
        return self

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        projection = np.zeros((1, 1), dtype=np.float32) if self._projection is None else self._projection
        bias = np.zeros((1,), dtype=np.float32) if self._bias is None else self._bias
        np.savez_compressed(path, projection=projection, bias=bias, name=np.asarray([self.name]))

    def propose(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        started = time.perf_counter()
        vector = context_to_vector(context)
        self._ensure_params(vector.shape[0], H, action_dim)
        assert self._projection is not None and self._bias is not None
        mean = np.tanh(vector @ self._projection + self._bias).reshape(1, H, action_dim)
        rng = np.random.default_rng(seed)
        std = self.base_std.reshape(1, 1, action_dim)
        actions = mean + rng.normal(0.0, std, size=(K, H, action_dim)).astype(np.float32)
        actions = clip_actions(actions, action_bounds)
        log_prob = None
        if self.supports_log_prob:
            residual = actions - mean
            var = np.maximum(std * std, 1e-6)
            log_prob = -0.5 * np.sum((residual * residual) / var + np.log(2.0 * np.pi * var), axis=(1, 2))
            log_prob = np.asarray(log_prob, dtype=np.float32)
        return ProposalBatch(
            actions=actions,
            log_prob=log_prob,
            diagnostics={"prototype_generator": True, "model_family": self.name},
            sample_time_sec=time.perf_counter() - started,
        )

    def log_prob(self, context, actions) -> np.ndarray | None:
        if not self.supports_log_prob:
            return None
        action_array = np.asarray(actions, dtype=np.float32)
        vector = context_to_vector(context)
        self._ensure_params(vector.shape[0], action_array.shape[1], action_array.shape[2])
        assert self._projection is not None and self._bias is not None
        mean = np.tanh(vector @ self._projection + self._bias).reshape(1, action_array.shape[1], action_array.shape[2])
        std = self.base_std.reshape(1, 1, action_array.shape[2])
        var = np.maximum(std * std, 1e-6)
        return np.asarray(-0.5 * np.sum(((action_array - mean) ** 2) / var + np.log(2.0 * np.pi * var), axis=(1, 2)), dtype=np.float32)
