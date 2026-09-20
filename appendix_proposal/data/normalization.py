"""Normalization stats shared by CVAE training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _safe_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.clip(np.std(values, axis=axis), 1e-6, None).astype(np.float32)


@dataclass(frozen=True)
class NormalizationStats:
    context_mean: np.ndarray
    context_std: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray

    @classmethod
    def fit(cls, dataset, *, action_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> "NormalizationStats":
        contexts = []
        obs = []
        actions = []
        for index in range(len(dataset)):
            item = dataset[index]
            contexts.append(item["context"])
            obs.append(item["obs_seq"])
            mask = item.get("mask")
            if mask is None:
                actions.append(item["action_seq"].reshape(-1, item["action_seq"].shape[-1]))
            else:
                valid = item["action_seq"][np.asarray(mask, dtype=bool)]
                if valid.size:
                    actions.append(valid.reshape(-1, item["action_seq"].shape[-1]))
        context_array = np.asarray(contexts, dtype=np.float32)
        obs_array = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1]) for x in obs], axis=0)
        action_array = np.concatenate(actions, axis=0).astype(np.float32)
        if action_bounds is None:
            action_min = np.min(action_array, axis=0).astype(np.float32)
            action_max = np.max(action_array, axis=0).astype(np.float32)
        else:
            action_min = np.asarray(action_bounds[0], dtype=np.float32)
            action_max = np.asarray(action_bounds[1], dtype=np.float32)
        return cls(
            context_mean=np.mean(context_array, axis=0).astype(np.float32),
            context_std=_safe_std(context_array),
            obs_mean=np.mean(obs_array, axis=0).astype(np.float32),
            obs_std=_safe_std(obs_array),
            action_mean=np.mean(action_array, axis=0).astype(np.float32),
            action_std=_safe_std(action_array),
            action_min=action_min,
            action_max=action_max,
        )

    def normalize_context(self, context: np.ndarray) -> np.ndarray:
        return ((np.asarray(context, dtype=np.float32) - self.context_mean) / self.context_std).astype(np.float32)

    def denormalize_context(self, context: np.ndarray) -> np.ndarray:
        return (np.asarray(context, dtype=np.float32) * self.context_std + self.context_mean).astype(np.float32)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return ((np.asarray(obs, dtype=np.float32) - self.obs_mean) / self.obs_std).astype(np.float32)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float32) * self.obs_std + self.obs_mean).astype(np.float32)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return ((np.asarray(action, dtype=np.float32) - self.action_mean) / self.action_std).astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action, dtype=np.float32) * self.action_std + self.action_mean).astype(np.float32)

    def action_to_unit(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        span = np.maximum(self.action_max - self.action_min, 1e-6)
        return np.clip(2.0 * (action - self.action_min) / span - 1.0, -1.0, 1.0).astype(np.float32)

    def unit_to_action(self, unit_action: np.ndarray) -> np.ndarray:
        unit = np.clip(np.asarray(unit_action, dtype=np.float32), -1.0, 1.0)
        return (self.action_min + 0.5 * (unit + 1.0) * (self.action_max - self.action_min)).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NormalizationStats":
        return cls(
            context_mean=np.asarray(payload["context_mean"], dtype=np.float32),
            context_std=np.asarray(payload["context_std"], dtype=np.float32),
            obs_mean=np.asarray(payload["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(payload["obs_std"], dtype=np.float32),
            action_mean=np.asarray(payload["action_mean"], dtype=np.float32),
            action_std=np.asarray(payload["action_std"], dtype=np.float32),
            action_min=np.asarray(payload["action_min"], dtype=np.float32),
            action_max=np.asarray(payload["action_max"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GeneratorContextNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, vectors: np.ndarray | list[np.ndarray]) -> "GeneratorContextNormalizer":
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return cls(mean=np.mean(array, axis=0).astype(np.float32), std=_safe_std(array))

    def normalize_context(self, context: np.ndarray) -> np.ndarray:
        return self.transform(context)

    def transform(self, context: np.ndarray) -> np.ndarray:
        return ((np.asarray(context, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, context: np.ndarray) -> np.ndarray:
        return (np.asarray(context, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GeneratorContextNormalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeneratorContextNormalizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
