"""Conditional RealNVP NormalizingFlow-AIF over bounded action chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.normalization import NormalizationStats
from generators.base import ProposalBatch, ProposalGenerator
from generators.utils import clip_actions, context_to_vector


@dataclass(frozen=True)
class NormalizingFlowConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    hidden_dim: int = 128
    num_layers: int = 2
    num_coupling_layers: int = 4
    scale_clip: float = 2.0
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    action_squash_eps: float = 1e-5


class _CouplingNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(max(int(num_layers), 1)):
            layers.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AffineCoupling(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_dim: int, num_layers: int, mask: torch.Tensor, scale_clip: float) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float().view(1, -1))
        self.scale_clip = float(scale_clip)
        self.net = _CouplingNet(dim + context_dim, hidden_dim, 2 * dim, num_layers)

    def _scale_shift(self, masked: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([masked, context.float()], dim=-1))
        shift, log_scale = raw.chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        log_scale = torch.tanh(log_scale) * self.scale_clip * inv_mask
        shift = shift * inv_mask
        return log_scale, shift

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = z * self.mask
        log_scale, shift = self._scale_shift(masked, context)
        y = masked + (1.0 - self.mask) * (z * torch.exp(log_scale) + shift)
        log_det = torch.sum(log_scale, dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = y * self.mask
        log_scale, shift = self._scale_shift(masked, context)
        z = masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        log_det = -torch.sum(log_scale, dim=-1)
        return z, log_det


class NormalizingFlowGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: NormalizingFlowConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.name = "normalizing_flow_aif"
        self.supports_log_prob = True
        self.supports_guidance = False
        self.is_learned = True
        self.is_stochastic = True
        self.default_action_low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
        self.default_action_high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
        self.flat_dim = int(config.horizon * config.action_dim)
        self.base_mean = nn.Linear(config.context_dim, self.flat_dim)
        self.base_log_std = nn.Linear(config.context_dim, self.flat_dim)
        masks = []
        for layer in range(int(config.num_coupling_layers)):
            pattern = (torch.arange(self.flat_dim) + layer) % 2
            masks.append((pattern == 0).float())
        self.couplings = nn.ModuleList(
            [
                _AffineCoupling(
                    self.flat_dim,
                    config.context_dim,
                    config.hidden_dim,
                    config.num_layers,
                    mask,
                    config.scale_clip,
                )
                for mask in masks
            ]
        )
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _bounds(self, action_bounds: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
        if action_bounds is None:
            return self.default_action_low, self.default_action_high
        return np.asarray(action_bounds[0], dtype=np.float32), np.asarray(action_bounds[1], dtype=np.float32)

    def _context_np(self, context) -> np.ndarray:
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

    def _base_params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.base_mean(context.float())
        log_std = self.base_log_std(context.float()).clamp(self.config.min_log_std, self.config.max_log_std)
        return mean, log_std

    def _forward_flow(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = z
        log_det = torch.zeros((z.shape[0],), device=z.device, dtype=z.dtype)
        for coupling in self.couplings:
            y, delta = coupling(y, context)
            log_det = log_det + delta
        return y, log_det

    def _inverse_flow(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = y
        log_det = torch.zeros((y.shape[0],), device=y.device, dtype=y.dtype)
        for coupling in reversed(self.couplings):
            z, delta = coupling.inverse(z, context)
            log_det = log_det + delta
        return z, log_det

    def _unit_to_action(self, unit_actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if self.normalizer is not None:
            actions = self.normalizer.unit_to_action(unit_actions)
        else:
            low, high = self._bounds(action_bounds)
            unit = np.clip(np.asarray(unit_actions, dtype=np.float32), -1.0, 1.0)
            actions = low + 0.5 * (unit + 1.0) * (high - low)
        if action_bounds is not None:
            actions = clip_actions(actions, action_bounds)
        return np.asarray(actions, dtype=np.float32)

    def _action_to_unit(self, actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if self.normalizer is not None:
            return self.normalizer.action_to_unit(actions)
        low, high = self._bounds(action_bounds)
        span = np.maximum(high - low, 1e-6)
        return np.clip(2.0 * (np.asarray(actions, dtype=np.float32) - low) / span - 1.0, -1.0, 1.0).astype(np.float32)

    def _pre_tanh(self, unit: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.action_squash_eps)
        clipped = torch.clamp(unit, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

    def _log_prob_pre_tanh(self, context: torch.Tensor, pre_tanh: torch.Tensor) -> torch.Tensor:
        flat_y = pre_tanh.reshape(pre_tanh.shape[0], -1)
        z, inv_log_det = self._inverse_flow(flat_y, context.float())
        mean, log_std = self._base_params(context.float())
        base = -0.5 * (((z - mean) / torch.exp(log_std)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        return torch.sum(base, dim=-1) + inv_log_det

    def log_prob_unit(self, context: torch.Tensor, action_unit: torch.Tensor) -> torch.Tensor:
        unit = torch.clamp(action_unit.float(), -1.0 + float(self.config.action_squash_eps), 1.0 - float(self.config.action_squash_eps))
        pre_tanh = self._pre_tanh(unit)
        log_p_y = self._log_prob_pre_tanh(context.float(), pre_tanh)
        tanh_log_det = torch.sum(torch.log(torch.clamp(1.0 - unit.reshape(unit.shape[0], -1) ** 2, min=1e-8)), dim=-1)
        return log_p_y - tanh_log_det

    def loss(self, context: torch.Tensor, action_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        log_prob = self.log_prob_unit(context.float(), action_unit.float())
        nll = -torch.mean(log_prob)
        return {"loss": nll, "nll": nll.detach(), "mean_log_prob": torch.mean(log_prob.detach())}

    @torch.no_grad()
    def sample_unit(self, context, K: int, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(context_np).float().to(self.device).repeat(int(K), 1)
        mean, log_std = self._base_params(context_t)
        eps = torch.randn(mean.shape, generator=generator, device=self.device)
        z = mean + eps * torch.exp(log_std)
        pre_tanh, _ = self._forward_flow(z, context_t)
        unit = torch.tanh(pre_tanh).view(int(K), self.config.horizon, self.config.action_dim)
        log_prob = self.log_prob_unit(context_t, unit)
        return unit.cpu().numpy().astype(np.float32), log_prob.cpu().numpy().astype(np.float32)

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
                f"NormalizingFlow configured for H={self.config.horizon}, action_dim={self.config.action_dim}; "
                f"got H={H}, action_dim={action_dim}."
            )
        started = time.perf_counter()
        unit, log_prob = self.sample_unit(context, K, seed=seed)
        actions = self._unit_to_action(unit, action_bounds=action_bounds)
        elapsed = time.perf_counter() - started
        return ProposalBatch(
            actions=actions,
            log_prob=log_prob,
            diagnostics={
                "flow_type": "conditional_realnvp",
                "num_coupling_layers": int(self.config.num_coupling_layers),
                "log_prob_exact": True,
                "action_density_space": "tanh_unit_action",
            },
            sample_time_sec=elapsed,
        )

    def log_prob(self, context, actions) -> np.ndarray | None:
        action_unit = self._action_to_unit(np.asarray(actions, dtype=np.float32), action_bounds=None)
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(np.repeat(context_np, action_unit.shape[0], axis=0)).float().to(self.device)
        action_t = torch.from_numpy(action_unit).float().to(self.device)
        with torch.no_grad():
            values = self.log_prob_unit(context_t, action_t)
        return values.cpu().numpy().astype(np.float32)

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
    ) -> "NormalizingFlowGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "NormalizingFlowGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = NormalizingFlowConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
