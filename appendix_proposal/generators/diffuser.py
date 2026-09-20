"""Diffuser-AIF proposals over action chunks or state-action futures.

The default mode is action-only. In ``state_action`` mode the model learns a
joint future sequence ``(obs_{t+1:t+H}, action_{t:t+H-1})`` and returns the
generated observations alongside executable actions. Shared AIF planning still
scores the returned actions through simulator rollouts for a fair controller
comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data.normalization import NormalizationStats
from generators.base import ProposalBatch, ProposalGenerator
from generators.diffusion_policy import DiffusionPolicyConfig, DiffusionPolicyGenerator
from generators.utils import context_to_vector


@dataclass(frozen=True)
class DiffuserConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0
    diffusion_steps: int = 64
    sample_steps: int = 16
    beta_start: float = 1e-4
    beta_end: float = 0.02
    context_dropout: float = 0.0
    backbone: str = "mlp"
    mode: str = "action_only"
    state_dim: int = 14
    state_unit_clip: float = 5.0


class DiffuserGenerator(nn.Module, ProposalGenerator):
    """Diffuser-style proposal generator with diagnostic state-action support."""

    def __init__(
        self,
        config: DiffuserConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if config.mode not in {"action_only", "state_action"}:
            raise ValueError("Diffuser mode must be 'action_only' or 'state_action'.")
        self.config = config
        self.normalizer = normalizer
        self.name = "diffuser_aif"
        self.supports_log_prob = False
        self.supports_guidance = True
        self.is_learned = True
        self.is_stochastic = True
        model_action_dim = config.action_dim if config.mode == "action_only" else config.state_dim + config.action_dim
        self.action_model = DiffusionPolicyGenerator(
            DiffusionPolicyConfig(
                context_dim=config.context_dim,
                horizon=config.horizon,
                action_dim=model_action_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                diffusion_steps=config.diffusion_steps,
                sample_steps=config.sample_steps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                context_dropout=config.context_dropout,
                backbone=config.backbone,
            ),
            normalizer=normalizer if config.mode == "action_only" else None,
            device=device,
        )
        self.action_model.name = "diffuser_aif_action_model" if config.mode == "action_only" else "diffuser_aif_state_action_model"

    @property
    def device(self) -> torch.device:
        return self.action_model.device

    def loss(self, context: torch.Tensor, target_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.action_model.loss(context, target_unit)
        metric_name = "action_denoising_loss" if self.config.mode == "action_only" else "state_action_denoising_loss"
        losses[metric_name] = losses["loss"].detach()
        return losses

    def propose(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        if int(H) != self.config.horizon or int(action_dim) != self.config.action_dim:
            raise ValueError(
                f"Diffuser configured for H={self.config.horizon}, action_dim={self.config.action_dim}; "
                f"got H={H}, action_dim={action_dim}."
            )
        started = time.perf_counter()
        if self.config.mode == "state_action":
            sequence_unit = self.action_model.sample_unit(self._normalized_context(context), K, seed=seed)
            observations, actions = self.state_action_unit_to_outputs(sequence_unit, action_bounds=action_bounds)
            elapsed = time.perf_counter() - started
            diagnostics = {
                "mode": self.config.mode,
                "action_generator": "ddpm_state_action_chunk",
                "denoising_steps": int(self.config.sample_steps),
                "diffusion_steps": int(self.config.diffusion_steps),
                "generated_states_learned": True,
                "generated_states_ignored_by_scorer": True,
                "shared_simulator_rollout_required": True,
                "state_dim": int(self.config.state_dim),
                "state_action_dim": int(self.config.state_dim + self.config.action_dim),
            }
            return ProposalBatch(
                actions=actions,
                observations=observations,
                model_score=np.zeros((int(K),), dtype=np.float32),
                diagnostics=diagnostics,
                sample_time_sec=elapsed,
                raw_model_outputs={
                    "generated_state_action_unit": sequence_unit,
                    "generated_states": observations,
                    "generated_actions": actions,
                    "used_for_scoring": False,
                    "learned_state_action": True,
                },
            )

        action_batch = self.action_model.propose(context, K, H, action_dim, action_bounds, seed=seed)
        elapsed = time.perf_counter() - started
        diagnostics = dict(action_batch.diagnostics)
        diagnostics.update(
            {
                "mode": self.config.mode,
                "action_generator": "ddpm_action_chunk",
                "shared_simulator_rollout_required": True,
            }
        )
        return ProposalBatch(
            actions=action_batch.actions,
            observations=None,
            log_prob=action_batch.log_prob,
            model_score=action_batch.model_score,
            diagnostics=diagnostics,
            sample_time_sec=max(float(action_batch.sample_time_sec), elapsed),
            raw_model_outputs=action_batch.raw_model_outputs,
        )

    def _normalized_context(self, context) -> np.ndarray:
        vector = context_to_vector(context)
        if vector.shape[0] > self.config.context_dim:
            vector = vector[: self.config.context_dim]
        elif vector.shape[0] < self.config.context_dim:
            padded = np.zeros((self.config.context_dim,), dtype=np.float32)
            padded[: vector.shape[0]] = vector
            vector = padded
        if self.normalizer is not None and self.normalizer.context_mean.shape[0] == vector.shape[0]:
            return self.normalizer.normalize_context(vector)
        return vector.astype(np.float32)

    def state_action_to_unit(self, state_action_seq: np.ndarray) -> np.ndarray:
        sequence = np.asarray(state_action_seq, dtype=np.float32)
        state_dim = int(self.config.state_dim)
        action_dim = int(self.config.action_dim)
        expected = state_dim + action_dim
        if sequence.shape[-1] != expected:
            raise ValueError(f"Expected state-action width {expected}, got {sequence.shape[-1]}.")
        obs = sequence[..., :state_dim]
        actions = sequence[..., state_dim:]
        if self.normalizer is None:
            obs_unit = np.clip(obs, -1.0, 1.0)
            low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
            high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
            span = np.maximum(high - low, 1e-6)
            action_unit = np.clip(2.0 * (actions - low) / span - 1.0, -1.0, 1.0)
        else:
            clip = max(float(self.config.state_unit_clip), 1e-6)
            obs_unit = np.clip(self.normalizer.normalize_obs(obs), -clip, clip) / clip
            action_unit = self.normalizer.action_to_unit(actions)
        return np.concatenate([obs_unit, action_unit], axis=-1).astype(np.float32)

    def state_action_unit_to_outputs(
        self,
        sequence_unit: np.ndarray,
        *,
        action_bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        unit = np.asarray(sequence_unit, dtype=np.float32)
        state_dim = int(self.config.state_dim)
        action_dim = int(self.config.action_dim)
        expected = state_dim + action_dim
        if unit.shape[-1] != expected:
            raise ValueError(f"Expected state-action unit width {expected}, got {unit.shape[-1]}.")
        obs_unit = unit[..., :state_dim]
        action_unit = unit[..., state_dim:]
        if self.normalizer is None:
            observations = obs_unit.astype(np.float32)
            low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
            high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
            actions = low + 0.5 * (np.clip(action_unit, -1.0, 1.0) + 1.0) * (high - low)
        else:
            observations = self.normalizer.denormalize_obs(obs_unit * max(float(self.config.state_unit_clip), 1e-6))
            actions = self.normalizer.unit_to_action(action_unit)
        if action_bounds is not None:
            low, high = action_bounds
            actions = np.clip(actions, np.asarray(low, dtype=np.float32).reshape(1, 1, -1), np.asarray(high, dtype=np.float32).reshape(1, 1, -1))
        return observations.astype(np.float32), actions.astype(np.float32)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "normalizer": None if self.normalizer is None else self.normalizer.to_dict(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    def load(
        self,
        checkpoint_path: str | Path,
        normalization: object | None = None,
        config: dict[str, Any] | None = None,
    ) -> "DiffuserGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "DiffuserGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = DiffuserConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
