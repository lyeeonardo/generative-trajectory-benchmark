"""Proposal-only conditional FlowMatching-AIF over action chunks."""

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
class FlowMatchingConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    hidden_dim: int = 128
    num_layers: int = 3
    ode_steps: int = 16
    solver: str = "euler"
    path_type: str = "linear"
    context_dropout: float = 0.0
    action_squash: str = "clip"


class _VectorField(nn.Module):
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


class FlowMatchingGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: FlowMatchingConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if config.path_type != "linear":
            raise ValueError("Only linear flow-matching paths are implemented for Generator smoke tests.")
        if config.solver not in {"euler", "heun"}:
            raise ValueError("FlowMatching solver must be 'euler' or 'heun'.")
        if config.action_squash not in {"clip", "tanh"}:
            raise ValueError("FlowMatching action_squash must be 'clip' or 'tanh'.")
        self.config = config
        self.normalizer = normalizer
        self.name = "flow_matching_aif"
        self.supports_log_prob = False
        self.supports_guidance = False
        self.is_learned = True
        self.is_stochastic = True
        self.default_action_low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
        self.default_action_high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
        flat = config.horizon * config.action_dim
        time_dim = 16
        self.time_dim = time_dim
        self.vector_field = _VectorField(flat + config.context_dim + time_dim, config.hidden_dim, flat, config.num_layers)
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

    def _time_embedding(self, r: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(torch.arange(half, device=r.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        args = r.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _predict_velocity(self, x_r: torch.Tensor, r: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        flat = x_r.reshape(x_r.shape[0], -1)
        emb = self._time_embedding(r)
        velocity = self.vector_field(torch.cat([flat, context.float(), emb], dim=-1))
        return velocity.view_as(x_r)

    def _unit_to_action(self, unit_actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
        if self.config.action_squash == "tanh":
            unit_actions = np.tanh(np.asarray(unit_actions, dtype=np.float32))
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

    def loss(self, context: torch.Tensor, action_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = action_unit.shape[0]
        if self.training and self.config.context_dropout > 0.0:
            keep = (torch.rand((batch, 1), device=context.device) > float(self.config.context_dropout)).float()
            context = context * keep
        epsilon = torch.randn_like(action_unit)
        r = torch.rand((batch,), device=action_unit.device)
        r_view = r.view(batch, 1, 1)
        x_r = r_view * action_unit + (1.0 - r_view) * epsilon
        target = action_unit - epsilon
        pred = self._predict_velocity(x_r, r, context.float())
        loss = torch.mean((pred - target) ** 2)
        return {"loss": loss, "flow_matching_loss": loss.detach()}

    @torch.no_grad()
    def sample_unit(self, context, K: int, *, ode_steps: int | None = None, seed: int | None = None) -> np.ndarray:
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        steps = max(1, int(self.config.ode_steps if ode_steps is None else ode_steps))
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(context_np).float().to(self.device).repeat(int(K), 1)
        x = torch.randn((int(K), self.config.horizon, self.config.action_dim), generator=generator, device=self.device)
        dt = 1.0 / float(steps)
        for index in range(steps):
            r = torch.full((int(K),), float(index) * dt, device=self.device, dtype=torch.float32)
            velocity = self._predict_velocity(x, r, context_t)
            if self.config.solver == "heun":
                x_euler = x + dt * velocity
                r_next = torch.full((int(K),), min(1.0, float(index + 1) * dt), device=self.device, dtype=torch.float32)
                velocity_next = self._predict_velocity(x_euler, r_next, context_t)
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
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
            raise ValueError(
                f"FlowMatching configured for H={self.config.horizon}, action_dim={self.config.action_dim}; "
                f"got H={H}, action_dim={action_dim}."
            )
        started = time.perf_counter()
        unit = self.sample_unit(context, K, seed=seed)
        actions = self._unit_to_action(unit, action_bounds)
        elapsed = time.perf_counter() - started
        diversity = float(np.std(actions.reshape(int(K), -1), axis=0).mean()) if int(K) > 1 else 0.0
        return ProposalBatch(
            actions=actions,
            diagnostics={
                "ode_steps": int(self.config.ode_steps),
                "solver": self.config.solver,
                "path_type": self.config.path_type,
                "action_squash": self.config.action_squash,
                "sample_diversity": diversity,
                "log_prob_available": False,
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
    ) -> "FlowMatchingGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "FlowMatchingGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = FlowMatchingConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
