"""Proposal-only DiffusionPolicy-AIF over action chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data.normalization import NormalizationStats
from generators.base import ProposalBatch, ProposalGenerator
from generators.utils import clip_actions, context_to_vector


@dataclass(frozen=True)
class DiffusionPolicyConfig:
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
    prediction_type: str = "epsilon"
    backbone: str = "mlp"


class _Denoiser(nn.Module):
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


class _TransformerDenoiser(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        time_dim: int,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("Diffusion transformer hidden_dim must be divisible by num_heads.")
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.context_embed = nn.Linear(context_dim, hidden_dim)
        self.time_embed = nn.Linear(time_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, int(horizon), int(hidden_dim)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=4 * int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t: torch.Tensor, context: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        tokens = self.action_embed(x_t.float()) + self.position[:, : x_t.shape[1], :]
        tokens = tokens + self.context_embed(context.float()).unsqueeze(1) + self.time_embed(time_embedding.float()).unsqueeze(1)
        return self.head(self.encoder(tokens))


class DiffusionPolicyGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: DiffusionPolicyConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if config.backbone not in {"mlp", "transformer"}:
            raise ValueError("DiffusionPolicyConfig.backbone must be 'mlp' or 'transformer'.")
        self.config = config
        self.normalizer = normalizer
        self.name = "diffusion_policy_aif"
        self.supports_log_prob = False
        self.supports_guidance = True
        self.is_learned = True
        self.is_stochastic = True
        self.default_action_low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
        self.default_action_high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
        flat = config.horizon * config.action_dim
        time_dim = 16
        self.time_dim = time_dim
        if config.backbone == "transformer":
            self.denoiser = _TransformerDenoiser(
                context_dim=config.context_dim,
                action_dim=config.action_dim,
                horizon=config.horizon,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                time_dim=time_dim,
            )
        else:
            self.denoiser = _Denoiser(flat + config.context_dim + time_dim, config.hidden_dim, flat, config.num_layers)
        betas = torch.linspace(float(config.beta_start), float(config.beta_end), int(config.diffusion_steps), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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

    def _unit_to_action(self, unit_actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
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
        return np.clip(2.0 * (np.asarray(actions, dtype=np.float32) - low) / span - 1.0, -1.0, 1.0)

    def _bounds(self, action_bounds: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
        if action_bounds is None:
            return self.default_action_low, self.default_action_high
        return np.asarray(action_bounds[0], dtype=np.float32), np.asarray(action_bounds[1], dtype=np.float32)

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _predict_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        emb = self._time_embedding(timesteps)
        if self.config.backbone == "transformer":
            return self.denoiser(x_t, context, emb).view_as(x_t)
        flat = x_t.reshape(x_t.shape[0], -1)
        return self.denoiser(torch.cat([flat, context, emb], dim=-1)).view_as(x_t)

    def loss(self, context: torch.Tensor, action_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = action_unit.shape[0]
        if self.training and self.config.context_dropout > 0.0:
            keep = (torch.rand((batch, 1), device=context.device) > float(self.config.context_dropout)).float()
            context = context * keep
        timesteps = torch.randint(0, self.config.diffusion_steps, (batch,), device=action_unit.device)
        noise = torch.randn_like(action_unit)
        a_bar = self.alpha_bar[timesteps].view(batch, 1, 1)
        x_t = torch.sqrt(a_bar) * action_unit + torch.sqrt(1.0 - a_bar) * noise
        pred = self._predict_noise(x_t, timesteps, context.float())
        loss = torch.mean((pred - noise) ** 2)
        return {"loss": loss, "denoising_loss": loss.detach()}

    @torch.no_grad()
    def sample_unit(self, context, K: int, *, sample_steps: int | None = None, seed: int | None = None) -> np.ndarray:
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        steps = int(self.config.sample_steps if sample_steps is None else sample_steps)
        steps = max(1, min(steps, int(self.config.diffusion_steps)))
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(context_np).float().to(self.device).repeat(int(K), 1)
        x = torch.randn((int(K), self.config.horizon, self.config.action_dim), generator=generator, device=self.device)
        schedule = torch.linspace(self.config.diffusion_steps - 1, 0, steps, device=self.device).round().long().unique(sorted=True)
        schedule = torch.flip(schedule, dims=[0])
        for t in schedule:
            timesteps = torch.full((int(K),), int(t.item()), device=self.device, dtype=torch.long)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            pred_noise = self._predict_noise(x, timesteps, context_t)
            mean = (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
            if int(t.item()) > 0:
                noise = torch.randn(x.shape, generator=generator, device=self.device)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return torch.clamp(x, -1.0, 1.0).cpu().numpy().astype(np.float32)

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
            raise ValueError(f"DiffusionPolicy configured for H={self.config.horizon}, action_dim={self.config.action_dim}; got H={H}, action_dim={action_dim}.")
        started = time.perf_counter()
        unit = self.sample_unit(context, K, seed=seed)
        actions = self._unit_to_action(unit, action_bounds)
        elapsed = time.perf_counter() - started
        return ProposalBatch(
            actions=actions,
            diagnostics={
                "denoising_steps": int(self.config.sample_steps),
                "diffusion_steps": int(self.config.diffusion_steps),
                "backbone": self.config.backbone,
            },
            sample_time_sec=elapsed,
        )

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
    ) -> "DiffusionPolicyGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "DiffusionPolicyGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = DiffusionPolicyConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
