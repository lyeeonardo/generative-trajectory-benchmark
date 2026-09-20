"""Conditional VAE over future action chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.archive_dataset import build_context_vector
from data.normalization import NormalizationStats
from generators.base import ProposalBatch, ProposalGenerator
from generators.utils import context_to_vector


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class CVAEConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    hidden_dim: int = 128
    latent_dim: int = 16
    beta_kl: float = 0.01
    recon_weight: float = 1.0
    sample_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)


class CVAEActionGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: CVAEConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.sample_noise_std = np.asarray(config.sample_noise_std, dtype=np.float32)
        self.name = "cvae_aif"
        self.supports_log_prob = False
        self.supports_guidance = False
        self.is_learned = True
        self.is_stochastic = True
        flat_action = config.horizon * config.action_dim
        self.encoder = _MLP(config.context_dim + flat_action, config.hidden_dim, 2 * config.latent_dim)
        self.decoder = _MLP(config.context_dim + config.latent_dim, config.hidden_dim, flat_action)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _context_vector_for_model(self, context) -> np.ndarray:
        if self.config.context_dim == 25 and hasattr(context, "obs_t") and hasattr(context, "action_history"):
            prev_action = np.asarray(context.action_history, dtype=np.float32).reshape(-1, self.config.action_dim)[-1]
            return build_context_vector(context.obs_t, prev_action=prev_action, progress=float(context.time_fraction))
        vector = context_to_vector(context)
        if vector.shape[0] > self.config.context_dim:
            return vector[: self.config.context_dim].astype(np.float32)
        if vector.shape[0] < self.config.context_dim:
            padded = np.zeros((self.config.context_dim,), dtype=np.float32)
            padded[: vector.shape[0]] = vector
            return padded
        return vector.astype(np.float32)

    def _normalize_context_np(self, context: np.ndarray) -> np.ndarray:
        vector = self._context_vector_for_model(context)
        if self.normalizer is None:
            return vector.astype(np.float32)
        if self.normalizer.context_mean.shape[0] != vector.shape[0]:
            return vector.astype(np.float32)
        return self.normalizer.normalize_context(vector)

    def _actions_to_unit_np(self, actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if self.normalizer is not None:
            return self.normalizer.action_to_unit(actions)
        if action_bounds is None:
            return np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        low, high = action_bounds
        span = np.maximum(high - low, 1e-6)
        return np.clip(2.0 * (np.asarray(actions, dtype=np.float32) - low) / span - 1.0, -1.0, 1.0)

    def _unit_to_actions_np(
        self,
        unit_actions: np.ndarray,
        action_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.normalizer is not None:
            actions = self.normalizer.unit_to_action(unit_actions)
        else:
            unit = np.clip(np.asarray(unit_actions, dtype=np.float32), -1.0, 1.0)
            if action_bounds is None:
                actions = unit
            else:
                low, high = action_bounds
                actions = low + 0.5 * (unit + 1.0) * (high - low)
        if action_bounds is not None:
            low, high = action_bounds
            actions = np.clip(actions, low.reshape(*(1 for _ in range(actions.ndim - 1)), -1), high.reshape(*(1 for _ in range(actions.ndim - 1)), -1))
        return np.asarray(actions, dtype=np.float32)

    def encode(self, context: torch.Tensor, action_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_action = action_unit.reshape(action_unit.shape[0], -1)
        params = self.encoder(torch.cat([context, flat_action], dim=-1))
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def decode_unit(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(torch.cat([context, z], dim=-1))
        return torch.tanh(flat).view(context.shape[0], self.config.horizon, self.config.action_dim)

    def forward(self, context: torch.Tensor, action_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(context, action_unit)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode_unit(context, z)
        return recon, mu, logvar

    def loss(
        self,
        context: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        beta_kl: float | None = None,
    ) -> dict[str, torch.Tensor]:
        recon, mu, logvar = self.forward(context, action_seq)
        recon_loss = torch.mean((recon - action_seq) ** 2)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        beta = self.config.beta_kl if beta_kl is None else float(beta_kl)
        total = self.config.recon_weight * recon_loss + beta * kl
        return {"loss": total, "recon": recon_loss.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def sample(
        self,
        context: np.ndarray,
        K: int,
        *,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> np.ndarray:
        self.eval()
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        else:
            generator = None
        context_np = self._normalize_context_np(context).reshape(1, -1).astype(np.float32)
        context_t = torch.from_numpy(context_np).to(self.device)
        context_t = context_t.repeat(int(K), 1)
        z = torch.randn((int(K), self.config.latent_dim), generator=generator, device=self.device)
        unit = self.decode_unit(context_t, z).cpu().numpy()
        actions = self._unit_to_actions_np(unit, action_bounds=action_bounds)
        noise_std = np.asarray(self.sample_noise_std, dtype=np.float32).reshape(1, 1, -1)
        if np.any(noise_std > 0.0):
            rng = np.random.default_rng(seed)
            actions = actions + rng.normal(0.0, noise_std, size=actions.shape).astype(np.float32)
            low, high = action_bounds
            actions = np.clip(actions, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
        return actions.astype(np.float32)

    @torch.no_grad()
    def reconstruct(
        self,
        context: np.ndarray,
        action_seq: np.ndarray,
        *,
        action_bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        self.eval()
        context_np = self._normalize_context_np(context).reshape(1, -1).astype(np.float32)
        unit_np = self._actions_to_unit_np(action_seq, action_bounds=action_bounds).reshape(
            1,
            self.config.horizon,
            self.config.action_dim,
        )
        context_t = torch.from_numpy(context_np).to(self.device)
        unit_t = torch.from_numpy(unit_np).to(self.device)
        mu, _ = self.encode(context_t, unit_t)
        recon_unit = self.decode_unit(context_t, mu).cpu().numpy()[0]
        return self._unit_to_actions_np(recon_unit, action_bounds=action_bounds)

    def propose(
        self,
        context: np.ndarray,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        if int(H) != self.config.horizon or int(action_dim) != self.config.action_dim:
            raise ValueError(
                f"CVAE configured for H={self.config.horizon}, action_dim={self.config.action_dim}; "
                f"got H={H}, action_dim={action_dim}."
            )
        started = time.perf_counter()
        actions = self.sample(context, K, action_bounds=action_bounds, seed=seed)
        elapsed = time.perf_counter() - started
        return ProposalBatch(actions=actions, diagnostics={"proposal_time": elapsed}, sample_time_sec=elapsed)

    def load(
        self,
        checkpoint_path: str | Path,
        normalization: object | None = None,
        config: dict[str, object] | None = None,
    ) -> "CVAEActionGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

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

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "CVAEActionGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = CVAEConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
