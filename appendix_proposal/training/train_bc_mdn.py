"""Train the Generator BC/MDN proposal baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from data.archive_dataset import GeneratorActionWindowDataset
from data.normalization import NormalizationStats
from generators.bc_mdn import BCMDNConfig, BCMDNGenerator


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "context": torch.from_numpy(np.stack([row["context"] for row in batch], axis=0)),
        "action_seq": torch.from_numpy(np.stack([row["action_seq"] for row in batch], axis=0)),
    }


def _device(config: dict[str, Any]) -> str:
    requested = str(config.get("device", "cpu"))
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _split_archive_root(config: dict[str, Any], split: str) -> Path | None:
    split_key = {"validation": "val", "nominal_test": "test", "heldout_test": "test"}.get(str(split), str(split))
    explicit = config.get(f"{split_key}_archive_root") or config.get(f"{split}_archive_root")
    if explicit:
        return Path(explicit)
    dataset_root = config.get("dataset_root")
    if dataset_root:
        return Path(dataset_root) / split_key
    return None


def _random_subset(dataset, size: int, *, seed: int):
    count = min(int(size), len(dataset))
    if count >= len(dataset):
        return dataset
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False))
    return Subset(dataset, indices.tolist())


def _epoch(model: BCMDNGenerator, loader: DataLoader, normalizer: NormalizationStats, optimizer=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "nll": 0.0, "mixture_entropy": 0.0}
    batches = 0
    for batch in loader:
        context = torch.from_numpy(normalizer.normalize_context(batch["context"].numpy())).float().to(model.device)
        action_unit = torch.from_numpy(normalizer.action_to_unit(batch["action_seq"].numpy())).float().to(model.device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        losses = model.loss(context, action_unit)
        if train:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_bc_mdn(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 13))
    np.random.seed(seed)
    torch.manual_seed(seed)
    preset = str(config.get("preset", "smoke"))
    horizon = int(config.get("H", config.get("horizon", 8)))
    output_dir = Path(config.get("output_dir", "outputs/generators/bc_mdn_aif"))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_archive_root = _split_archive_root(config, "train")
    val_archive_root = _split_archive_root(config, "val")
    train_dataset = GeneratorActionWindowDataset(archive_root=train_archive_root, preset=preset, split="train", horizon=horizon, history_len=int(config.get("history_len", 4)))
    val_dataset = GeneratorActionWindowDataset(archive_root=val_archive_root, preset=preset, split="val", horizon=horizon, history_len=int(config.get("history_len", 4)))
    if config.get("max_train_windows") is not None:
        train_dataset = _random_subset(train_dataset, int(config["max_train_windows"]), seed=seed)
    if config.get("max_val_windows") is not None:
        val_dataset = _random_subset(val_dataset, int(config["max_val_windows"]), seed=seed + 1)
    action_low = np.asarray(config.get("action_low", [-0.8, -0.8, -4.0]), dtype=np.float32)
    action_high = np.asarray(config.get("action_high", [0.8, 0.8, 4.0]), dtype=np.float32)
    normalizer = NormalizationStats.fit(train_dataset, action_bounds=(action_low, action_high))
    normalizer_path = output_dir / "normalization.json"
    normalizer.save(normalizer_path)
    batch_size = min(int(config.get("batch_size", 32)), len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False, collate_fn=_collate)
    device = _device(config)
    model = BCMDNGenerator(
        BCMDNConfig(
            context_dim=train_dataset.context_dim,
            horizon=horizon,
            variant=str(config.get("variant", "mdn")),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_components=int(config.get("num_components", 5)),
            dropout=float(config.get("dropout", 0.0)),
            layer_norm=bool(config.get("layer_norm", False)),
        ),
        normalizer=normalizer,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 5e-4)))
    epochs = int(config.get("epochs", 3))
    history = {"train": [], "val": []}
    best_val = float("inf")
    checkpoint_path = output_dir / "checkpoints" / "bc_mdn_aif.pt"
    for _ in range(epochs):
        train_metrics = _epoch(model, train_loader, normalizer, optimizer)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, normalizer, None)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save_checkpoint(checkpoint_path)
    history_path = output_dir / "metrics" / "bc_mdn_training.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_path),
        "normalization_path": str(normalizer_path),
        "history_path": str(history_path),
        "dataset": {
            "train_archive_root": None if train_archive_root is None else str(train_archive_root),
            "val_archive_root": None if val_archive_root is None else str(val_archive_root),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
        },
        "history": history,
    }
