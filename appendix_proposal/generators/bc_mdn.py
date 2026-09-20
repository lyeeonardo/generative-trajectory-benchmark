"""Gaussian / MDN behavior-cloning proposal generator for Generator."""

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
class BCMDNConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    variant: str = "mdn"
    hidden_dim: int = 128
    num_components: int = 5
    dropout: float = 0.0
    layer_norm: bool = False
    min_log_std: float = -4.0
    max_log_std: float = 1.0


class _ContextEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, *, dropout: float, layer_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCMDNGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: BCMDNConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.name = "bc_mdn_aif"
        self.supports_log_prob = True
        self.supports_guidance = False
        self.is_learned = True
        self.is_stochastic = True
        self.default_action_low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
        self.default_action_high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
        self._last_component_indices: np.ndarray | None = None
        self.encoder = _ContextEncoder(
            config.context_dim,
            config.hidden_dim,
            dropout=float(config.dropout),
            layer_norm=bool(config.layer_norm),
        )
        flat = config.horizon * config.action_dim
        if config.variant == "gaussian":
            self.mean_head = nn.Linear(config.hidden_dim, flat)
            self.log_std_head = nn.Linear(config.hidden_dim, flat)
            self.mdn_head = None
        elif config.variant == "mdn":
            out_dim = config.num_components + 2 * config.num_components * flat
            self.mdn_head = nn.Linear(config.hidden_dim, out_dim)
            self.mean_head = None
            self.log_std_head = None
        else:
            raise ValueError("BCMDNConfig.variant must be 'gaussian' or 'mdn'.")
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _normalize_context_np(self, context) -> np.ndarray:
        vector = context_to_vector(context)
        if self.normalizer is not None and self.normalizer.context_mean.shape[0] == vector.shape[0]:
            return self.normalizer.normalize_context(vector)
        return vector.astype(np.float32)

    def _action_to_unit(self, actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if self.normalizer is not None:
            return self.normalizer.action_to_unit(actions)
        low, high = self._bounds(action_bounds)
        span = np.maximum(high - low, 1e-6)
        return np.clip(2.0 * (np.asarray(actions, dtype=np.float32) - low) / span - 1.0, -1.0, 1.0)

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

    def _bounds(self, action_bounds: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
        if action_bounds is None:
            return self.default_action_low, self.default_action_high
        return np.asarray(action_bounds[0], dtype=np.float32), np.asarray(action_bounds[1], dtype=np.float32)

    def _params(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(context)
        flat = self.config.horizon * self.config.action_dim
        if self.config.variant == "gaussian":
            assert self.mean_head is not None and self.log_std_head is not None
            mean = self.mean_head(encoded).view(-1, self.config.horizon, self.config.action_dim)
            log_std = self.log_std_head(encoded).view(-1, self.config.horizon, self.config.action_dim)
            return {"mean": mean, "log_std": log_std.clamp(self.config.min_log_std, self.config.max_log_std)}
        assert self.mdn_head is not None
        raw = self.mdn_head(encoded)
        logits = raw[:, : self.config.num_components]
        rest = raw[:, self.config.num_components :]
        means_raw, log_stds_raw = rest.chunk(2, dim=-1)
        means = means_raw.view(-1, self.config.num_components, self.config.horizon, self.config.action_dim)
        log_stds = log_stds_raw.view(-1, self.config.num_components, self.config.horizon, self.config.action_dim)
        return {
            "logits": logits,
            "means": means,
            "log_stds": log_stds.clamp(self.config.min_log_std, self.config.max_log_std),
        }

    def loss(self, context: torch.Tensor, action_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        params = self._params(context.float())
        if self.config.variant == "gaussian":
            mean = params["mean"]
            log_std = params["log_std"]
            var = torch.exp(2.0 * log_std)
            nll = 0.5 * (((action_unit - mean) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi))
            loss = nll.sum(dim=(1, 2)).mean()
            entropy = torch.mean(log_std.detach())
            return {"loss": loss, "nll": loss.detach(), "mixture_entropy": entropy}
        logits = params["logits"]
        means = params["means"]
        log_stds = params["log_stds"]
        target = action_unit[:, None, :, :]
        var = torch.exp(2.0 * log_stds)
        log_component = -0.5 * (((target - means) ** 2) / var + 2.0 * log_stds + np.log(2.0 * np.pi)).sum(dim=(2, 3))
        log_mix = F.log_softmax(logits, dim=-1)
        log_prob = torch.logsumexp(log_mix + log_component, dim=-1)
        loss = -log_prob.mean()
        probs = torch.softmax(logits.detach(), dim=-1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=-1).mean()
        return {"loss": loss, "nll": loss.detach(), "mixture_entropy": entropy}

    @torch.no_grad()
    def sample_unit(self, context, K: int, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        context_np = self._normalize_context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(context_np).float().to(self.device)
        context_t = context_t.repeat(int(K), 1)
        params = self._params(context_t)
        if self.config.variant == "gaussian":
            mean = params["mean"]
            std = torch.exp(params["log_std"])
            eps = torch.randn(mean.shape, generator=generator, device=self.device)
            unit = torch.clamp(mean + eps * std, -1.0, 1.0)
            log_prob = self._gaussian_log_prob(unit, mean, params["log_std"]).cpu().numpy()
            return unit.cpu().numpy(), log_prob.astype(np.float32), None
        logits = params["logits"]
        probs = torch.softmax(logits, dim=-1)
        component = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(-1)
        gather_index = component.view(-1, 1, 1, 1).expand(-1, 1, self.config.horizon, self.config.action_dim)
        mean = torch.gather(params["means"], 1, gather_index).squeeze(1)
        log_std = torch.gather(params["log_stds"], 1, gather_index).squeeze(1)
        std = torch.exp(log_std)
        eps = torch.randn(mean.shape, generator=generator, device=self.device)
        unit = torch.clamp(mean + eps * std, -1.0, 1.0)
        log_prob = self._mdn_log_prob(unit, params).cpu().numpy()
        components = component.cpu().numpy().astype(np.int64)
        return unit.cpu().numpy(), log_prob.astype(np.float32), components

    def _gaussian_log_prob(self, action_unit: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        return -0.5 * (((action_unit - mean) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi)).sum(dim=(1, 2))

    def _mdn_log_prob(self, action_unit: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        target = action_unit[:, None, :, :]
        var = torch.exp(2.0 * params["log_stds"])
        log_component = -0.5 * (((target - params["means"]) ** 2) / var + 2.0 * params["log_stds"] + np.log(2.0 * np.pi)).sum(dim=(2, 3))
        log_mix = F.log_softmax(params["logits"], dim=-1)
        return torch.logsumexp(log_mix + log_component, dim=-1)

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
            raise ValueError(f"BC/MDN configured for H={self.config.horizon}, action_dim={self.config.action_dim}; got H={H}, action_dim={action_dim}.")
        started = time.perf_counter()
        unit, log_prob, components = self.sample_unit(context, K, seed=seed)
        actions = self._unit_to_action(unit, action_bounds=action_bounds)
        self._last_component_indices = components
        diagnostics = {
            "variant": self.config.variant,
            "mixture_components": self.config.num_components if self.config.variant == "mdn" else 1,
        }
        if components is not None:
            diagnostics["component_indices"] = components.tolist()
            diagnostics["component_usage"] = {str(i): int(np.sum(components == i)) for i in range(self.config.num_components)}
        elapsed = time.perf_counter() - started
        return ProposalBatch(actions=actions, log_prob=log_prob, diagnostics=diagnostics, sample_time_sec=elapsed)

    def log_prob(self, context, actions) -> np.ndarray | None:
        action_unit = self._action_to_unit(np.asarray(actions, dtype=np.float32), action_bounds=None)
        context_np = self._normalize_context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(np.repeat(context_np, action_unit.shape[0], axis=0)).float().to(self.device)
        action_t = torch.from_numpy(action_unit).float().to(self.device)
        with torch.no_grad():
            params = self._params(context_t)
            if self.config.variant == "gaussian":
                values = self._gaussian_log_prob(action_t, params["mean"], params["log_std"])
            else:
                values = self._mdn_log_prob(action_t, params)
        return values.cpu().numpy().astype(np.float32)

    def diagnostics(self) -> dict[str, object]:
        base = super().diagnostics()
        base.update({"variant": self.config.variant, "num_components": self.config.num_components})
        if self._last_component_indices is not None:
            base["last_component_usage"] = {str(i): int(np.sum(self._last_component_indices == i)) for i in range(self.config.num_components)}
        return base

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
    ) -> "BCMDNGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "BCMDNGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = BCMDNConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
