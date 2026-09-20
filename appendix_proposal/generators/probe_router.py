"""Budget-neutral pre-action probing and learned proposal-source routing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from aif.tilt_context import snapshot_with_tilt
from generators.base import ProposalBatch, ProposalGenerator
from generators.composite import _call_generator, _score_pool


ROUTING_FEATURE_NAMES = (
    "tilt_entropy_unit",
    "primary_score_advantage",
    "primary_invalidity",
    "fallback_invalidity",
    "primary_diversity",
    "fallback_diversity",
    "lagged_prediction_error",
    "lagged_belief_kl",
    "lagged_primary_regret",
)


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    if result.ndim == 0:
        return float(result)
    return result


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LinearRoutingModel:
    """Small auditable logistic router with frozen feature normalization."""

    weights: tuple[float, ...]
    intercept: float
    feature_mean: tuple[float, ...] = (0.0,) * len(ROUTING_FEATURE_NAMES)
    feature_scale: tuple[float, ...] = (1.0,) * len(ROUTING_FEATURE_NAMES)
    rho_min: float = 0.0
    rho_max: float = 1.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        width = len(ROUTING_FEATURE_NAMES)
        if len(self.weights) != width or len(self.feature_mean) != width or len(self.feature_scale) != width:
            raise ValueError(f"Routing model arrays must all have width {width}")
        if any(not np.isfinite(value) for value in (*self.weights, self.intercept, *self.feature_mean, *self.feature_scale)):
            raise ValueError("Routing model contains non-finite values")
        if any(value <= 0.0 for value in self.feature_scale):
            raise ValueError("Routing feature scales must be positive")
        if not 0.0 <= self.rho_min <= self.rho_max <= 1.0:
            raise ValueError("Routing rho bounds must satisfy 0 <= rho_min <= rho_max <= 1")

    @classmethod
    def default(cls, *, rho_min: float = 0.05, rho_max: float = 0.95) -> "LinearRoutingModel":
        # Conservative initialization for development data collection.  The
        # score advantage dominates, while uncertainty and lagged mismatch
        # move budget away from the learned primary proposal source.
        return cls(
            weights=(-0.75, 2.75, -1.25, 1.25, 0.20, -0.20, -0.75, -0.50, -1.00),
            intercept=0.35,
            rho_min=float(rho_min),
            rho_max=float(rho_max),
        )

    def predict(self, features: dict[str, float] | Sequence[float]) -> float:
        if isinstance(features, dict):
            vector = np.asarray([features[name] for name in ROUTING_FEATURE_NAMES], dtype=np.float64)
        else:
            vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if vector.shape != (len(ROUTING_FEATURE_NAMES),):
            raise ValueError(f"Expected {len(ROUTING_FEATURE_NAMES)} routing features, got {vector.shape}")
        normalized = (vector - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        probability = float(_sigmoid(float(np.dot(normalized, np.asarray(self.weights)) + self.intercept)))
        return float(np.clip(probability, self.rho_min, self.rho_max))

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "model_type": "linear_logistic_probe_router",
            "feature_names": list(ROUTING_FEATURE_NAMES),
            "weights": list(self.weights),
            "intercept": float(self.intercept),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "rho_min": float(self.rho_min),
            "rho_max": float(self.rho_max),
        }
        if include_hash:
            payload["model_hash"] = _canonical_hash(payload)
        return payload

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        payload = self.payload()
        if metadata:
            payload["metadata"] = dict(metadata)
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError(f"Refusing to overwrite a different routing model: {target}")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LinearRoutingModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload.get("feature_names", ())) != ROUTING_FEATURE_NAMES:
            raise ValueError("Routing model feature contract does not match this code")
        expected_hash = str(payload.get("model_hash", ""))
        hash_payload = {key: value for key, value in payload.items() if key not in {"model_hash", "metadata"}}
        if expected_hash != _canonical_hash(hash_payload):
            raise ValueError("Routing model hash verification failed")
        return cls(
            weights=tuple(float(value) for value in payload["weights"]),
            intercept=float(payload["intercept"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            rho_min=float(payload["rho_min"]),
            rho_max=float(payload["rho_max"]),
            schema_version=int(payload["schema_version"]),
        )


def fit_linear_routing_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
    l2: float = 0.05,
    learning_rate: float = 0.05,
    iterations: int = 4000,
    batch_size: int = 8192,
    random_seed: int = 20_260_811,
) -> tuple[LinearRoutingModel, dict[str, float]]:
    """Fit the frozen development-only logistic router with NumPy."""

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] != len(ROUTING_FEATURE_NAMES) or X.shape[0] != y.shape[0]:
        raise ValueError("Routing training arrays have incompatible shapes")
    if X.shape[0] < 2 or not set(np.unique(y)).issubset({0.0, 1.0}) or np.unique(y).size < 2:
        raise ValueError("Routing labels must contain both binary classes")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Routing training data contain non-finite values")
    base_weight = np.ones_like(y) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if base_weight.shape != y.shape or not np.all(np.isfinite(base_weight)) or np.any(base_weight <= 0.0):
        raise ValueError("Routing sample weights must be finite, positive, and aligned with labels")
    base_weight = base_weight / np.mean(base_weight)
    mean = np.average(X, axis=0, weights=base_weight)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=base_weight))
    scale = np.where(scale < 1e-8, 1.0, scale)
    Z = (X - mean) / scale
    weights = np.zeros((Z.shape[1],), dtype=np.float64)
    prevalence = float(np.clip(np.average(y, weights=base_weight), 1e-6, 1.0 - 1e-6))
    intercept = math.log(prevalence / (1.0 - prevalence))
    positive_weight = 0.5 / prevalence
    negative_weight = 0.5 / (1.0 - prevalence)
    sample_weight = base_weight * np.where(y > 0.5, positive_weight, negative_weight)
    sample_weight = sample_weight / np.mean(sample_weight)
    optimization_batch_size = min(int(batch_size), X.shape[0])
    if optimization_batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(int(random_seed))
    for _ in range(int(iterations)):
        if optimization_batch_size == X.shape[0]:
            indices = slice(None)
        else:
            indices = rng.integers(0, X.shape[0], size=optimization_batch_size)
        batch_Z = Z[indices]
        batch_y = y[indices]
        batch_weight = sample_weight[indices]
        probability = np.asarray(_sigmoid(batch_Z @ weights + intercept), dtype=np.float64)
        residual = (probability - batch_y) * batch_weight
        grad_w = batch_Z.T @ residual / np.sum(batch_weight) + float(l2) * weights
        grad_b = float(np.sum(residual) / np.sum(batch_weight))
        weights -= float(learning_rate) * grad_w
        intercept -= float(learning_rate) * grad_b
    model = LinearRoutingModel(
        weights=tuple(float(value) for value in weights),
        intercept=float(intercept),
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
    )
    probability = np.asarray([model.predict(row) for row in X], dtype=np.float64)
    prediction = probability >= 0.5
    metrics = {
        "n": float(X.shape[0]),
        "positive_rate": prevalence,
        "accuracy": float(np.average(prediction == (y > 0.5), weights=base_weight)),
        "balanced_accuracy": float(
            0.5 * np.mean(prediction[y > 0.5]) + 0.5 * np.mean(~prediction[y <= 0.5])
        ),
        "brier_score": float(np.average((probability - y) ** 2, weights=base_weight)),
        "log_loss": float(-np.average(y * np.log(np.clip(probability, 1e-9, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-9, 1.0)), weights=base_weight)),
        "optimization_iterations": float(iterations),
        "optimization_batch_size": float(optimization_batch_size),
        "optimization_random_seed": float(random_seed),
    }
    return model, metrics


def _batch_parts(
    records: Iterable[tuple[str, ProposalBatch]],
    *,
    H: int,
    action_dim: int,
    elapsed: float,
    diagnostics: dict[str, Any],
) -> ProposalBatch:
    batches = [(str(label), batch) for label, batch in records if np.asarray(batch.actions).shape[0] > 0]
    actions = np.concatenate([np.asarray(batch.actions, dtype=np.float32) for _, batch in batches], axis=0)
    labels = [label for label, batch in batches for _ in range(np.asarray(batch.actions).shape[0])]
    observation_shape = None
    for _, batch in batches:
        if batch.observations is not None:
            shape = tuple(np.asarray(batch.observations).shape[1:])
            if observation_shape is None:
                observation_shape = shape
            elif observation_shape != shape:
                raise ValueError(f"Probe router observation shapes disagree: {observation_shape} vs {shape}")
    observations = None
    observation_mask = None
    if observation_shape is not None:
        observation_parts = []
        mask_parts = []
        for _, batch in batches:
            count = np.asarray(batch.actions).shape[0]
            if batch.observations is None:
                observation_parts.append(np.zeros((count, *observation_shape), dtype=np.float32))
                mask_parts.append(np.zeros((count,), dtype=bool))
            else:
                observation_parts.append(np.asarray(batch.observations, dtype=np.float32))
                mask_parts.append(np.ones((count,), dtype=bool))
        observations = np.concatenate(observation_parts, axis=0)
        observation_mask = np.concatenate(mask_parts, axis=0)
    log_prob = None
    if batches and all(batch.log_prob is not None for _, batch in batches):
        log_prob = np.concatenate([np.asarray(batch.log_prob, dtype=np.float32).reshape(-1) for _, batch in batches])
    merged = {
        **diagnostics,
        "candidate_sources": labels,
        "proposal_mixture_entropy": _source_entropy(labels),
        "proposal_time": float(elapsed),
    }
    if observation_mask is not None:
        merged["generated_observation_mask"] = observation_mask.astype(bool).tolist()
    return ProposalBatch(
        actions=actions.reshape(-1, int(H), int(action_dim)),
        observations=observations,
        log_prob=log_prob,
        diagnostics=merged,
        sample_time_sec=float(elapsed),
    )


def _source_entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    array = np.asarray(labels, dtype=object)
    return float(-sum(float(np.mean(array == label)) * math.log(max(float(np.mean(array == label)), 1e-12)) for label in set(labels)))


def _diversity(actions: np.ndarray, action_bounds: tuple[np.ndarray, np.ndarray]) -> float:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape[0] <= 1:
        return 0.0
    low, high = action_bounds
    scale = np.maximum(np.asarray(high) - np.asarray(low), 1e-6).reshape(1, 1, -1)
    return float(np.clip(np.mean(np.std(array / scale, axis=0)), 0.0, 1.0))


def _diagnostic_snapshot(env_adapter, scorer, belief):
    snapshot = env_adapter.clone_state()
    config = getattr(scorer, "config", {}) or {}
    tilt_belief = config.get("tilt_belief") if bool(config.get("hidden_tilt_enabled", False)) else None
    if tilt_belief is None:
        return snapshot
    probabilities = np.asarray(tilt_belief.probabilities, dtype=np.float64).reshape(-1)
    mean_tilt = np.sum(np.asarray(tilt_belief.tilt_grid, dtype=np.float64) * probabilities[:, None], axis=0)
    return snapshot_with_tilt(snapshot, mean_tilt.astype(np.float32))


def _invalidity(actions: np.ndarray, env_adapter, snapshot) -> float:
    invalid = []
    for candidate in np.asarray(actions, dtype=np.float32):
        rollout = env_adapter.rollout_from_state(snapshot, candidate)
        observations = np.asarray(rollout.observations)
        invalid.append(bool(rollout.collision) or bool(getattr(rollout, "fall_out", False)) or not np.all(np.isfinite(observations)))
    return float(np.mean(invalid)) if invalid else 0.0


def _oracle_key(actions: np.ndarray, env_adapter) -> tuple[float, ...]:
    snapshot = env_adapter.clone_state()
    best: tuple[float, ...] | None = None
    for candidate in np.asarray(actions, dtype=np.float32):
        rollout = env_adapter.rollout_from_state(snapshot, candidate)
        key = (
            0.0 if rollout.success else 1.0,
            1.0 if getattr(rollout, "fall_out", False) else 0.0,
            1.0 if rollout.collision else 0.0,
            float(rollout.final_distance_to_goal),
            float(rollout.path_length),
        )
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError("Oracle diagnostic requires at least one candidate")
    return best


class ProbeAndRouteGenerator(ProposalGenerator):
    """Probe both sources, then allocate the remaining fixed K pre-action."""

    name = "probe_router_aif"
    is_stochastic = True

    def __init__(
        self,
        primary: ProposalGenerator,
        fallback: ProposalGenerator,
        *,
        routing_model: LinearRoutingModel | None = None,
        probe_per_source: int = 8,
        risk_mode: str = "cvar",
        cvar_alpha: float = 0.25,
        risk_beta: float = 1.0,
        score_scale: float = 50.0,
        prediction_error_scale: float = 1.0,
        belief_kl_scale: float = 1.0,
        regret_scale: float = 50.0,
        collect_oracle_labels: bool = False,
        oracle_candidate_count: int | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.routing_model = routing_model or LinearRoutingModel.default()
        self.probe_per_source = int(probe_per_source)
        self.risk_mode = str(risk_mode)
        self.cvar_alpha = float(cvar_alpha)
        self.risk_beta = float(risk_beta)
        self.score_scale = float(score_scale)
        self.prediction_error_scale = float(prediction_error_scale)
        self.belief_kl_scale = float(belief_kl_scale)
        self.regret_scale = float(regret_scale)
        self.collect_oracle_labels = bool(collect_oracle_labels)
        self.oracle_candidate_count = None if oracle_candidate_count is None else int(oracle_candidate_count)
        self.is_learned = bool(primary.is_learned or fallback.is_learned)
        self._lagged = {"prediction_error": 0.0, "belief_kl": 0.0, "primary_regret": 0.0}
        if self.probe_per_source < 1:
            raise ValueError("probe_per_source must be positive")

    def update_lagged_signals(
        self,
        *,
        prediction_error: float | None = None,
        belief_kl: float | None = None,
        primary_regret: float | None = None,
    ) -> None:
        for key, value in (
            ("prediction_error", prediction_error),
            ("belief_kl", belief_kl),
            ("primary_regret", primary_regret),
        ):
            if value is not None and np.isfinite(float(value)):
                self._lagged[key] = max(0.0, float(value))

    def _features(self, primary: ProposalBatch, fallback: ProposalBatch, *, scorer, belief, env_adapter, seed, action_bounds):
        primary_actions = np.asarray(primary.actions, dtype=np.float32)
        fallback_actions = np.asarray(fallback.actions, dtype=np.float32)
        primary_scores = _score_pool(
            primary_actions,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
            seed=seed,
            risk_mode=self.risk_mode,
            cvar_alpha=self.cvar_alpha,
            risk_beta=self.risk_beta,
        )
        fallback_scores = _score_pool(
            fallback_actions,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
            seed=None if seed is None else int(seed) + 4049,
            risk_mode=self.risk_mode,
            cvar_alpha=self.cvar_alpha,
            risk_beta=self.risk_beta,
        )
        config = getattr(scorer, "config", {}) or {}
        tilt_belief = config.get("tilt_belief") if bool(config.get("hidden_tilt_enabled", False)) else None
        entropy_unit = 0.0
        if tilt_belief is not None:
            probabilities = np.asarray(tilt_belief.probabilities, dtype=np.float64).reshape(-1)
            entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
            entropy_unit = entropy / max(math.log(max(probabilities.size, 2)), 1e-12)
        diagnostic_snapshot = _diagnostic_snapshot(env_adapter, scorer, belief)
        score_scale = max(self.score_scale, 1e-8)
        features = {
            "tilt_entropy_unit": float(np.clip(entropy_unit, 0.0, 1.0)),
            "primary_score_advantage": float(np.tanh((float(np.min(fallback_scores)) - float(np.min(primary_scores))) / score_scale)),
            "primary_invalidity": _invalidity(primary_actions, env_adapter, diagnostic_snapshot),
            "fallback_invalidity": _invalidity(fallback_actions, env_adapter, diagnostic_snapshot),
            "primary_diversity": _diversity(primary_actions, action_bounds),
            "fallback_diversity": _diversity(fallback_actions, action_bounds),
            "lagged_prediction_error": float(np.clip(self._lagged["prediction_error"] / max(self.prediction_error_scale, 1e-8), 0.0, 5.0)),
            "lagged_belief_kl": float(np.clip(self._lagged["belief_kl"] / max(self.belief_kl_scale, 1e-8), 0.0, 5.0)),
            "lagged_primary_regret": float(np.clip(self._lagged["primary_regret"] / max(self.regret_scale, 1e-8), 0.0, 5.0)),
        }
        diagnostics = {
            "primary_probe_best_G": float(np.min(primary_scores)),
            "fallback_probe_best_G": float(np.min(fallback_scores)),
            "primary_probe_mean_G": float(np.mean(primary_scores)),
            "fallback_probe_mean_G": float(np.mean(fallback_scores)),
        }
        return features, diagnostics

    def propose(self, context, K, H, action_dim, action_bounds, seed=None) -> ProposalBatch:
        raise RuntimeError("probe_router_aif requires scorer, belief, and environment access")

    def propose_with_aif(self, context, K, H, action_dim, action_bounds, *, scorer, belief, env_adapter, seed=None) -> ProposalBatch:
        started = time.perf_counter()
        total = int(K)
        probe = min(self.probe_per_source, total // 2)
        if probe < 1:
            raise ValueError("Probe router requires K >= 2")
        primary_probe = _call_generator(
            self.primary,
            context=context,
            count=probe,
            H=H,
            action_dim=action_dim,
            action_bounds=action_bounds,
            seed=seed,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
        )
        fallback_probe = _call_generator(
            self.fallback,
            context=context,
            count=probe,
            H=H,
            action_dim=action_dim,
            action_bounds=action_bounds,
            seed=None if seed is None else int(seed) + 1_000_003,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
        )
        assert primary_probe is not None and fallback_probe is not None
        features, probe_diagnostics = self._features(
            primary_probe,
            fallback_probe,
            scorer=scorer,
            belief=belief,
            env_adapter=env_adapter,
            seed=seed,
            action_bounds=action_bounds,
        )
        requested_rho = self.routing_model.predict(features)
        remaining = total - 2 * probe
        primary_extra_count = int(np.floor(requested_rho * remaining + 0.5))
        primary_extra_count = int(np.clip(primary_extra_count, 0, remaining))
        fallback_extra_count = remaining - primary_extra_count
        primary_extra = None
        if primary_extra_count > 0:
            primary_extra = _call_generator(
                self.primary,
                context=context,
                count=primary_extra_count,
                H=H,
                action_dim=action_dim,
                action_bounds=action_bounds,
                seed=None if seed is None else int(seed) + 2_000_003,
                scorer=scorer,
                belief=belief,
                env_adapter=env_adapter,
            )
        fallback_extra = None
        if fallback_extra_count > 0:
            fallback_extra = _call_generator(
                self.fallback,
                context=context,
                count=fallback_extra_count,
                H=H,
                action_dim=action_dim,
                action_bounds=action_bounds,
                seed=None if seed is None else int(seed) + 3_000_007,
                scorer=scorer,
                belief=belief,
                env_adapter=env_adapter,
            )
        oracle_diagnostics: dict[str, Any] = {}
        if self.collect_oracle_labels:
            oracle_count = int(self.oracle_candidate_count or total)
            oracle_primary = _call_generator(
                self.primary,
                context=context,
                count=oracle_count,
                H=H,
                action_dim=action_dim,
                action_bounds=action_bounds,
                seed=None if seed is None else int(seed) + 4_000_037,
                scorer=scorer,
                belief=belief,
                env_adapter=env_adapter,
            )
            oracle_fallback = _call_generator(
                self.fallback,
                context=context,
                count=oracle_count,
                H=H,
                action_dim=action_dim,
                action_bounds=action_bounds,
                seed=None if seed is None else int(seed) + 5_000_081,
                scorer=scorer,
                belief=belief,
                env_adapter=env_adapter,
            )
            assert oracle_primary is not None and oracle_fallback is not None
            primary_key = _oracle_key(np.asarray(oracle_primary.actions), env_adapter)
            fallback_key = _oracle_key(np.asarray(oracle_fallback.actions), env_adapter)
            oracle_diagnostics = {
                "oracle_primary_wins": bool(primary_key <= fallback_key),
                "oracle_source_winner": "generator" if primary_key <= fallback_key else "fallback",
                "oracle_primary_key": list(primary_key),
                "oracle_fallback_key": list(fallback_key),
                "oracle_candidate_count_per_source": oracle_count,
                "oracle_true_state_used_for_label_only": True,
            }
        records: list[tuple[str, ProposalBatch]] = [("generator", primary_probe), ("fallback", fallback_probe)]
        if primary_extra is not None:
            records.append(("generator", primary_extra))
        if fallback_extra is not None:
            records.append(("fallback", fallback_extra))
        actual_primary = probe + primary_extra_count
        diagnostics = {
            "router_type": "linear_logistic_probe_router",
            "router_model_hash": self.routing_model.payload()["model_hash"],
            "probe_per_source": probe,
            "requested_rho_t": float(requested_rho),
            "rho_t": float(actual_primary / total),
            "K_generator": int(actual_primary),
            "K_fallback": int(total - actual_primary),
            **{f"routing_feature_{key}": float(value) for key, value in features.items()},
            **probe_diagnostics,
            **oracle_diagnostics,
        }
        batch = _batch_parts(records, H=H, action_dim=action_dim, elapsed=time.perf_counter() - started, diagnostics=diagnostics)
        if np.asarray(batch.actions).shape != (total, int(H), int(action_dim)):
            raise ValueError(f"Probe router expected {(total, H, action_dim)}, got {np.asarray(batch.actions).shape}")
        return batch

    def diagnostics(self) -> dict[str, object]:
        return {
            **super().diagnostics(),
            "primary": self.primary.diagnostics(),
            "fallback": self.fallback.diagnostics(),
            "probe_per_source": self.probe_per_source,
            "routing_model_hash": self.routing_model.payload()["model_hash"],
            "collect_oracle_labels": self.collect_oracle_labels,
        }
