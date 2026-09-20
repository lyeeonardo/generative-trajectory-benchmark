"""Autoregressive Transformer-AIF over bounded continuous action chunks."""

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
class TransformerConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    num_components: int = 5
    variant: str = "gmm"
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    action_squash_eps: float = 1e-5
    temperature: float = 1.0


class TransformerActionGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if config.variant not in {"gmm", "gaussian"}:
            raise ValueError("TransformerConfig.variant must be 'gmm' or 'gaussian'.")
        if int(config.hidden_dim) % int(config.num_heads) != 0:
            raise ValueError("Transformer hidden_dim must be divisible by num_heads.")
        self.config = config
        self.normalizer = normalizer
        self.name = "transformer_aif"
        self.supports_log_prob = True
        self.supports_guidance = False
        self.is_learned = True
        self.is_stochastic = True
        self.default_action_low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
        self.default_action_high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
        self.context_encoder = nn.Sequential(nn.Linear(config.context_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        if config.variant == "gaussian":
            self.head = nn.Linear(config.hidden_dim, 2 * config.action_dim)
        else:
            self.head = nn.Linear(config.hidden_dim, config.num_components + 2 * config.num_components * config.action_dim)
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

    def _causal_mask(self, H: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((int(H), int(H)), device=device, dtype=torch.bool), diagonal=1)

    def _params(self, context: torch.Tensor, previous_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        B, H, _ = previous_unit.shape
        context_token = self.context_encoder(context.float()).unsqueeze(1)
        tokens = self.action_embed(previous_unit.float()) + self.position[:, :H, :] + context_token
        encoded = self.transformer(tokens, mask=self._causal_mask(H, tokens.device))
        raw = self.head(encoded)
        if self.config.variant == "gaussian":
            mean, log_std = raw.chunk(2, dim=-1)
            return {"mean": mean, "log_std": log_std.clamp(self.config.min_log_std, self.config.max_log_std)}
        logits = raw[:, :, : self.config.num_components]
        rest = raw[:, :, self.config.num_components :]
        means_raw, log_stds_raw = rest.chunk(2, dim=-1)
        means = means_raw.view(B, H, self.config.num_components, self.config.action_dim)
        log_stds = log_stds_raw.view(B, H, self.config.num_components, self.config.action_dim)
        return {"logits": logits, "means": means, "log_stds": log_stds.clamp(self.config.min_log_std, self.config.max_log_std)}

    def _teacher_inputs(self, action_unit: torch.Tensor) -> torch.Tensor:
        previous = torch.zeros_like(action_unit)
        if action_unit.shape[1] > 1:
            previous[:, 1:, :] = action_unit[:, :-1, :]
        return previous

    def _step_log_prob(self, y: torch.Tensor, unit: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        tanh_log_det = torch.sum(torch.log(torch.clamp(1.0 - unit.float() ** 2, min=1e-8)), dim=-1)
        if self.config.variant == "gaussian":
            mean = params["mean"]
            log_std = params["log_std"]
            log_p_y = -0.5 * (((y - mean) / torch.exp(log_std)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi)).sum(dim=-1)
            return log_p_y - tanh_log_det
        target = y[:, :, None, :]
        var = torch.exp(2.0 * params["log_stds"])
        log_component = -0.5 * (((target - params["means"]) ** 2) / var + 2.0 * params["log_stds"] + np.log(2.0 * np.pi)).sum(dim=-1)
        log_mix = F.log_softmax(params["logits"], dim=-1)
        log_p_y = torch.logsumexp(log_mix + log_component, dim=-1)
        return log_p_y - tanh_log_det

    def loss(self, context: torch.Tensor, action_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        unit = torch.clamp(action_unit.float(), -1.0 + float(self.config.action_squash_eps), 1.0 - float(self.config.action_squash_eps))
        previous = self._teacher_inputs(unit)
        params = self._params(context.float(), previous)
        y = self._pre_tanh(unit)
        log_prob_steps = self._step_log_prob(y, unit, params)
        nll = -torch.mean(torch.sum(log_prob_steps, dim=-1))
        entropy = torch.zeros((), device=context.device)
        if self.config.variant == "gmm":
            probs = torch.softmax(params["logits"].detach(), dim=-1)
            entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=-1).mean()
        return {"loss": nll, "nll": nll.detach(), "action_entropy": entropy}

    @torch.no_grad()
    def sample_unit(self, context, K: int, *, seed: int | None = None, temperature: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        temp = max(float(self.config.temperature if temperature is None else temperature), 1e-4)
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(context_np).float().to(self.device).repeat(int(K), 1)
        actions = torch.zeros((int(K), self.config.horizon, self.config.action_dim), device=self.device)
        log_prob = torch.zeros((int(K),), device=self.device)
        components = [] if self.config.variant == "gmm" else None
        for index in range(self.config.horizon):
            previous = self._teacher_inputs(actions)
            params = self._params(context_t, previous)
            if self.config.variant == "gaussian":
                mean = params["mean"][:, index, :]
                log_std = params["log_std"][:, index, :]
                y = mean + torch.randn(mean.shape, generator=generator, device=self.device) * torch.exp(log_std) * temp
                unit = torch.tanh(y)
                step_params = {"mean": mean[:, None, :], "log_std": log_std[:, None, :]}
            else:
                logits = params["logits"][:, index, :] / temp
                probs = torch.softmax(logits, dim=-1)
                component = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(-1)
                gather_index = component.view(-1, 1, 1).expand(-1, 1, self.config.action_dim)
                means = params["means"][:, index, :, :]
                log_stds = params["log_stds"][:, index, :, :]
                mean = torch.gather(means, 1, gather_index).squeeze(1)
                log_std = torch.gather(log_stds, 1, gather_index).squeeze(1)
                y = mean + torch.randn(mean.shape, generator=generator, device=self.device) * torch.exp(log_std) * temp
                unit = torch.tanh(y)
                step_params = {
                    "logits": params["logits"][:, index : index + 1, :],
                    "means": params["means"][:, index : index + 1, :, :],
                    "log_stds": params["log_stds"][:, index : index + 1, :, :],
                }
                assert components is not None
                components.append(component.cpu().numpy())
            actions[:, index, :] = unit
            log_prob = log_prob + self._step_log_prob(y[:, None, :], unit[:, None, :], step_params).squeeze(1)
        component_array = None
        if components is not None:
            component_array = np.stack(components, axis=1).astype(np.int64)
        return actions.cpu().numpy().astype(np.float32), log_prob.cpu().numpy().astype(np.float32), component_array

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
                f"Transformer configured for H={self.config.horizon}, action_dim={self.config.action_dim}; "
                f"got H={H}, action_dim={action_dim}."
            )
        started = time.perf_counter()
        unit, log_prob, components = self.sample_unit(context, K, seed=seed)
        actions = self._unit_to_action(unit, action_bounds=action_bounds)
        elapsed = time.perf_counter() - started
        diagnostics: dict[str, object] = {
            "variant": self.config.variant,
            "num_layers": int(self.config.num_layers),
            "num_heads": int(self.config.num_heads),
            "temperature": float(self.config.temperature),
            "log_prob_available": True,
            "conditioning": "context_plus_previous_actions",
        }
        if components is not None:
            diagnostics["component_usage"] = {str(i): int(np.sum(components == i)) for i in range(self.config.num_components)}
        return ProposalBatch(actions=actions, log_prob=log_prob, diagnostics=diagnostics, sample_time_sec=elapsed)

    def log_prob(self, context, actions) -> np.ndarray | None:
        action_unit = self._action_to_unit(np.asarray(actions, dtype=np.float32), action_bounds=None)
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(np.repeat(context_np, action_unit.shape[0], axis=0)).float().to(self.device)
        action_t = torch.from_numpy(action_unit).float().to(self.device)
        with torch.no_grad():
            unit = torch.clamp(action_t, -1.0 + float(self.config.action_squash_eps), 1.0 - float(self.config.action_squash_eps))
            params = self._params(context_t, self._teacher_inputs(unit))
            values = torch.sum(self._step_log_prob(self._pre_tanh(unit), unit, params), dim=-1)
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
    ) -> "TransformerActionGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "TransformerActionGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = TransformerConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
