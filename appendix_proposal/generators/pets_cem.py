"""PETS-style learned dynamics plus CEM proposal generator for Generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import time

import numpy as np
import torch
import torch.nn as nn

from aif.scoring import AIFScorer
from data.normalization import NormalizationStats
from generators.base import ProposalBatch, ProposalGenerator
from generators.utils import clip_actions, context_to_vector


@dataclass(frozen=True)
class PETSCEMConfig:
    context_dim: int
    horizon: int
    action_dim: int = 3
    obs_dim: int = 14
    ensemble_size: int = 3
    hidden_dim: int = 128
    num_layers: int = 2
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    cem_iterations: int = 2
    cem_population: int = 64
    cem_elite_frac: float = 0.25
    init_std: tuple[float, float, float] = (0.25, 0.25, 1.25)
    min_std: tuple[float, float, float] = (0.02, 0.02, 0.10)
    smoothing_alpha: float = 1.0


class _DynamicsMember(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(max(int(num_layers), 1)):
            layers.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        layers.append(nn.Linear(current, 2 * output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(x).chunk(2, dim=-1)
        return mean, log_std


class PETSCEMGenerator(nn.Module, ProposalGenerator):
    def __init__(
        self,
        config: PETSCEMConfig,
        *,
        normalizer: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.name = "pets_cem_aif"
        self.supports_log_prob = False
        self.supports_guidance = True
        self.is_learned = True
        self.is_stochastic = True
        self.default_action_low = np.asarray([-0.8, -0.8, -4.0], dtype=np.float32)
        self.default_action_high = np.asarray([0.8, 0.8, 4.0], dtype=np.float32)
        input_dim = config.obs_dim + config.action_dim + config.context_dim
        self.members = nn.ModuleList(
            [_DynamicsMember(input_dim, config.hidden_dim, config.obs_dim, config.num_layers) for _ in range(config.ensemble_size)]
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

    def _obs_from_context(self, context) -> np.ndarray:
        if hasattr(context, "obs_t"):
            obs = np.asarray(context.obs_t, dtype=np.float32).reshape(-1)
        else:
            obs = context_to_vector(context)[: self.config.obs_dim]
        if obs.shape[0] >= self.config.obs_dim:
            return obs[: self.config.obs_dim].astype(np.float32)
        padded = np.zeros((self.config.obs_dim,), dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded

    def _model_input(self, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs.float(), action.float(), context.float()], dim=-1)

    def _member_prediction(self, member: _DynamicsMember, obs: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_delta, log_std = member(self._model_input(obs, action, context))
        return mean_delta, log_std.clamp(self.config.min_log_std, self.config.max_log_std)

    def loss(self, context: torch.Tensor, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        B, H, _ = action_seq.shape
        obs = obs_seq[:, :-1, :].reshape(B * H, self.config.obs_dim)
        next_obs = obs_seq[:, 1:, :].reshape(B * H, self.config.obs_dim)
        actions = action_seq.reshape(B * H, self.config.action_dim)
        context_rep = context[:, None, :].expand(B, H, context.shape[-1]).reshape(B * H, context.shape[-1])
        target_delta = next_obs - obs
        losses = []
        mse_values = []
        for member in self.members:
            mean_delta, log_std = self._member_prediction(member, obs, actions, context_rep)
            var = torch.exp(2.0 * log_std)
            nll = 0.5 * (((target_delta - mean_delta) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi))
            losses.append(nll.sum(dim=-1).mean())
            mse_values.append(torch.mean((target_delta - mean_delta) ** 2).detach())
        loss = torch.stack(losses).mean()
        return {"loss": loss, "dynamics_nll": loss.detach(), "prediction_mse": torch.stack(mse_values).mean()}

    @torch.no_grad()
    def rollout_model(self, context, actions: np.ndarray) -> tuple[np.ndarray, float]:
        self.eval()
        action_array = np.asarray(actions, dtype=np.float32)
        K, H, _ = action_array.shape
        context_np = self._context_np(context).reshape(1, -1)
        context_t = torch.from_numpy(np.repeat(context_np, int(K), axis=0)).float().to(self.device)
        current = torch.from_numpy(np.repeat(self._obs_from_context(context).reshape(1, -1), int(K), axis=0)).float().to(self.device)
        action_t = torch.from_numpy(action_array).float().to(self.device)
        observations = [current.cpu().numpy().astype(np.float32)]
        uncertainty_values: list[float] = []
        for index in range(int(H)):
            member_next = []
            member_var = []
            action = action_t[:, index, :]
            for member in self.members:
                mean_delta, log_std = self._member_prediction(member, current, action, context_t)
                member_next.append(current + mean_delta)
                member_var.append(torch.exp(2.0 * log_std))
            stacked_next = torch.stack(member_next, dim=0)
            stacked_var = torch.stack(member_var, dim=0)
            current = torch.mean(stacked_next, dim=0)
            uncertainty = torch.mean(torch.var(stacked_next, dim=0, unbiased=False) + torch.mean(stacked_var, dim=0)).item()
            uncertainty_values.append(float(uncertainty))
            observations.append(current.cpu().numpy().astype(np.float32))
        return np.stack(observations, axis=1).astype(np.float32), float(np.mean(uncertainty_values) if uncertainty_values else 0.0)

    def propose(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        seed: int | None = None,
    ) -> ProposalBatch:
        return self._cem_optimize(context, K, H, action_dim, action_bounds, scorer=None, belief=None, seed=seed)

    def propose_with_aif(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        *,
        scorer,
        belief,
        env_adapter,
        seed: int | None = None,
    ) -> ProposalBatch:
        del env_adapter
        return self._cem_optimize(context, K, H, action_dim, action_bounds, scorer=scorer, belief=belief, seed=seed)

    def _cem_optimize(
        self,
        context,
        K: int,
        H: int,
        action_dim: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        *,
        scorer,
        belief,
        seed: int | None,
    ) -> ProposalBatch:
        if int(H) != self.config.horizon or int(action_dim) != self.config.action_dim:
            raise ValueError(f"PETS-CEM configured for H={self.config.horizon}, action_dim={self.config.action_dim}; got H={H}, action_dim={action_dim}.")
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        low, high = action_bounds
        population = max(int(self.config.cem_population), int(K))
        elites = max(1, min(population, int(np.ceil(population * float(self.config.cem_elite_frac)))))
        mean = np.zeros((int(H), int(action_dim)), dtype=np.float32)
        std = np.broadcast_to(np.asarray(self.config.init_std, dtype=np.float32).reshape(1, -1), mean.shape).copy()
        min_std = np.broadcast_to(np.asarray(self.config.min_std, dtype=np.float32).reshape(1, -1), mean.shape).copy()
        global_actions: np.ndarray | None = None
        global_scores: np.ndarray | None = None
        initial_best_score: float | None = None
        final_uncertainty = 0.0
        for iteration in range(max(int(self.config.cem_iterations), 1)):
            samples = rng.normal(mean.reshape(1, int(H), int(action_dim)), std.reshape(1, int(H), int(action_dim)), size=(population, int(H), int(action_dim))).astype(np.float32)
            samples = clip_actions(samples, (low, high))
            predicted_obs, uncertainty = self.rollout_model(context, samples)
            scores = self._predicted_scores(predicted_obs, samples, scorer=scorer, belief=belief)
            final_uncertainty = float(uncertainty)
            if iteration == 0:
                initial_best_score = float(np.min(scores))
            elite_indices = np.argsort(scores)[:elites]
            elite_stack = samples[elite_indices]
            elite_mean = elite_stack.mean(axis=0)
            elite_std = np.maximum(elite_stack.std(axis=0), min_std)
            alpha = float(np.clip(self.config.smoothing_alpha, 0.0, 1.0))
            mean = ((1.0 - alpha) * mean + alpha * elite_mean).astype(np.float32)
            std = np.maximum(((1.0 - alpha) * std + alpha * elite_std).astype(np.float32), min_std)
            if global_actions is None:
                combined_actions = samples
                combined_scores = scores
            else:
                combined_actions = np.concatenate([global_actions, samples], axis=0)
                combined_scores = np.concatenate([global_scores, scores], axis=0)
            selected = np.argsort(combined_scores)[: int(K)]
            global_actions = combined_actions[selected]
            global_scores = combined_scores[selected]
        assert global_actions is not None and global_scores is not None
        elapsed = time.perf_counter() - started
        final_best_score = float(np.min(global_scores))
        diagnostics = {
            "proposal_time": elapsed,
            "cem_iterations": max(int(self.config.cem_iterations), 1),
            "cem_population": population,
            "cem_elites": elites,
            "initial_best_score": initial_best_score,
            "final_best_score": final_best_score,
            "ensemble_uncertainty": final_uncertainty,
            "learned_model_rollout_used": True,
            "final_simulator_scoring_required": True,
            "shared_scorer_used_for_predicted_rollouts": scorer is not None,
        }
        return ProposalBatch(actions=global_actions.astype(np.float32), model_score=global_scores.astype(np.float32), diagnostics=diagnostics, sample_time_sec=elapsed)

    def _predicted_scores(self, observations: np.ndarray, actions: np.ndarray, *, scorer, belief) -> np.ndarray:
        if scorer is None or belief is None:
            goal = observations[:, 0, 7:9]
            start = observations[:, 0, :2]
            heuristic_final = start + 0.04 * np.sum(actions[:, :, :2], axis=1)
            predicted_final = observations[:, -1, :2]
            final = 0.5 * predicted_final + 0.5 * heuristic_final
            smooth = np.mean(np.sum(np.diff(actions, axis=1) ** 2, axis=-1), axis=1) if actions.shape[1] > 1 else np.zeros((actions.shape[0],), dtype=np.float32)
            return (np.linalg.norm(final - goal, axis=-1) + 0.01 * smooth).astype(np.float32)
        active_scorer = scorer if scorer is not None else AIFScorer()
        values = []
        for obs_seq, action_seq in zip(observations, actions):
            values.append(active_scorer.score_rollout(self._rollout_object(obs_seq, action_seq), belief).G_total)
        return np.asarray(values, dtype=np.float32)

    def _rollout_object(self, obs_seq: np.ndarray, action_seq: np.ndarray):
        obs = np.asarray(obs_seq, dtype=np.float32)
        actions = np.asarray(action_seq, dtype=np.float32)
        ball = obs[:, :2]
        center = obs[0, 9:11]
        radius = float(obs[0, 11])
        clearance = np.linalg.norm(ball - center.reshape(1, 2), axis=1) - (radius + 0.03)
        path_length = float(np.sum(np.linalg.norm(np.diff(ball, axis=0), axis=1))) if ball.shape[0] > 1 else 0.0
        smoothness = float(np.mean(np.sum(np.diff(actions, axis=0) ** 2, axis=1))) if actions.shape[0] > 1 else 0.0
        return SimpleNamespace(
            observations=obs,
            actions=actions,
            collision=bool(np.min(clearance) <= 0.0),
            success=bool(np.linalg.norm(ball[-1] - obs[-1, 7:9]) <= 0.06),
            timeout=False,
            minimum_obstacle_clearance=float(np.min(clearance)),
            path_length=path_length,
            action_smoothness=smoothness,
            route_label=self._route_label(obs),
        )

    def _route_label(self, obs_seq: np.ndarray) -> str:
        xy = np.asarray(obs_seq, dtype=np.float32)[:, :2]
        center_y = float(obs_seq[0, 10])
        band = np.abs(xy[:, 1] - center_y) <= 0.30
        near = xy[band] if np.any(band) else xy
        if np.min(near[:, 0]) < -0.12:
            return "left"
        if np.max(near[:, 0]) > 0.12:
            return "right"
        return "center"

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
    ) -> "PETSCEMGenerator":
        del normalization
        device = "cpu" if config is None else str(config.get("device", "cpu"))
        return self.load_checkpoint(checkpoint_path, device=device)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "PETSCEMGenerator":
        payload = torch.load(Path(path), map_location=device)
        config = PETSCEMConfig(**payload["config"])
        normalizer = None if payload.get("normalizer") is None else NormalizationStats.from_dict(payload["normalizer"])
        model = cls(config, normalizer=normalizer, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
