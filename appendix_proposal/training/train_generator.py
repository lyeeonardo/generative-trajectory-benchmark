"""Common Generator trainer for learned proposal generators."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.archive_dataset import FailureConditionedGeneratorActionWindowDataset, GeneratorActionWindowDataset
from data.normalization import NormalizationStats
from generators.registry import get_generator
from generators.utils import stable_config_hash, validate_proposal_batch


LEARNED_GENERATOR_MODELS = {
    "bc_mdn_aif",
    "cvae_aif",
    "diffusion_policy_aif",
    "diffuser_aif",
    "flow_matching_aif",
    "normalizing_flow_aif",
    "transformer_aif",
    "pets_cem_aif",
}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(requested: str | None) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    payload = {
        "context": torch.from_numpy(np.stack([row["context"] for row in batch], axis=0)),
        "action_seq": torch.from_numpy(np.stack([row["action_seq"] for row in batch], axis=0)),
        "obs_seq": torch.from_numpy(np.stack([row["obs_seq"] for row in batch], axis=0)),
        "mask": torch.from_numpy(np.stack([row["mask"] for row in batch], axis=0)),
    }
    if all("state_action_seq" in row for row in batch):
        payload["state_action_seq"] = torch.from_numpy(np.stack([row["state_action_seq"] for row in batch], axis=0))
    return payload


def _subset(dataset, size: int):
    count = min(int(size), len(dataset))
    return Subset(dataset, list(range(count)))


def _random_subset(dataset, size: int, *, seed: int):
    count = min(int(size), len(dataset))
    if count >= len(dataset):
        return dataset
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False))
    return Subset(dataset, indices.tolist())


def _first_item(dataset):
    return dataset[0]


def _make_loader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    workers = int(num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": min(int(batch_size), len(dataset)),
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": _collate,
        "pin_memory": bool(pin_memory),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return DataLoader(dataset, **kwargs)


def _shutdown_loader(loader: DataLoader) -> None:
    """Release persistent worker processes before the next campaign fit.

    PyTorch keeps a persistent loader's iterator (and its file descriptors) on
    the ``DataLoader`` instance.  Campaign training calls this module many
    times in one process, so relying on eventual garbage collection can exhaust
    a normal per-process descriptor limit.  The iterator hook is private but is
    the same cleanup path used by PyTorch's iterator destructor.
    """

    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    loader._iterator = None


def _split_archive_root(cfg: dict[str, Any], split: str) -> Path | None:
    split_key = {"validation": "val", "nominal_test": "test", "heldout_test": "test"}.get(str(split), str(split))
    explicit = cfg.get(f"{split_key}_archive_root") or cfg.get(f"{split}_archive_root")
    if explicit:
        return Path(explicit)
    dataset_root = cfg.get("dataset_root")
    if dataset_root:
        return Path(dataset_root) / split_key
    return None


def _as_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value)
        parts = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parts = [str(item) for item in value]
    paths: list[Path] = []
    for part in parts:
        if any(char in part for char in "*?[]"):
            paths.extend(sorted(Path().glob(part)))
        else:
            paths.append(Path(part))
    return paths


def _batch_loss(model_name: str, model, batch: dict[str, torch.Tensor], normalizer: NormalizationStats) -> dict[str, torch.Tensor]:
    context_np = batch["context"].numpy()
    context = torch.from_numpy(normalizer.normalize_context(context_np)).float().to(model.device, non_blocking=True)
    if model_name == "pets_cem_aif":
        obs_seq = batch["obs_seq"].float().to(model.device, non_blocking=True)
        action_seq = batch["action_seq"].float().to(model.device, non_blocking=True)
        return model.loss(context, obs_seq, action_seq)
    if model_name == "diffuser_aif" and getattr(getattr(model, "config", None), "mode", "action_only") == "state_action":
        if "state_action_seq" not in batch:
            raise ValueError("diffuser_aif state_action mode requires dataset target_mode='state_action'.")
        state_action_np = batch["state_action_seq"].numpy()
        state_action_unit = torch.from_numpy(model.state_action_to_unit(state_action_np)).float().to(model.device, non_blocking=True)
        return model.loss(context, state_action_unit)
    action_np = batch["action_seq"].numpy()
    action_unit = torch.from_numpy(normalizer.action_to_unit(action_np)).float().to(model.device, non_blocking=True)
    return model.loss(context, action_unit)


def _epoch(model_name: str, model, loader: DataLoader, normalizer: NormalizationStats, optimizer=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        losses = _batch_loss(model_name, model, batch, normalizer)
        if train:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key, value in losses.items():
            if torch.is_tensor(value):
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True, default=str), flush=True)


def _sample_diagnostics(model_name: str, model, dataset, action_bounds: tuple[np.ndarray, np.ndarray], path: Path) -> dict[str, Any]:
    item = _first_item(dataset)
    batch = model.propose(item["context"], 4, model.config.horizon, model.config.action_dim, action_bounds, seed=123)
    validate_proposal_batch(batch, K=4, H=model.config.horizon, action_dim=model.config.action_dim, action_bounds=action_bounds)
    actions = np.asarray(batch.actions, dtype=np.float32)
    payload = {
        "model": model_name,
        "shape": list(actions.shape),
        "action_mean": actions.mean(axis=(0, 1)).tolist(),
        "action_std": actions.std(axis=(0, 1)).tolist(),
        "sample_time_sec": float(batch.sample_time_sec),
        "diagnostics": dict(batch.diagnostics),
    }
    if batch.observations is not None:
        observations = np.asarray(batch.observations, dtype=np.float32)
        payload["observation_shape"] = list(observations.shape)
        payload["observation_mean"] = observations.mean(axis=(0, 1)).tolist()
        payload["observation_std"] = observations.std(axis=(0, 1)).tolist()
    _write_json(path, payload)
    return payload


def train_generator_model(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    model_name = str(cfg.get("model", cfg.get("model_name", "")))
    if model_name not in LEARNED_GENERATOR_MODELS:
        raise ValueError(f"Common trainer supports learned Generator models only, got {model_name!r}.")
    seed = int(cfg.get("seed", 13))
    seed_everything(seed)
    preset = str(cfg.get("preset", "smoke"))
    horizon = int(cfg.get("H", cfg.get("horizon", 8)))
    history_len = int(cfg.get("history_len", 4))
    debug_overfit = bool(cfg.get("debug_overfit", False))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0))
    output_dir = Path(cfg.get("output_dir", "outputs/generators")) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_archive_root = _split_archive_root(cfg, "train")
    val_archive_root = _split_archive_root(cfg, "val")
    dataset_kwargs = {
        "horizon": horizon,
        "history_len": history_len,
        "successful_only": bool(cfg.get("successful_only", True)),
        "include_failures": bool(cfg.get("include_failures", False)),
        "target_mode": str(cfg.get("target_mode", "action")),
    }
    dataset_cls = FailureConditionedGeneratorActionWindowDataset if bool(cfg.get("failure_conditioned", False)) else GeneratorActionWindowDataset
    train_extra: dict[str, Any] = {}
    if dataset_cls is FailureConditionedGeneratorActionWindowDataset:
        train_extra = {
            "failure_sidecar_paths": _as_paths(cfg.get("failure_sidecar_paths", cfg.get("train_failure_sidecar_paths"))),
            "max_failure_windows": cfg.get("max_failure_windows"),
            "append_outcome_to_context": bool(cfg.get("append_outcome_to_context", True)),
        }
    train_dataset = dataset_cls(
        archive_root=train_archive_root,
        preset=preset,
        split="train",
        route_balance=bool(cfg.get("route_balance", False)),
        **dataset_kwargs,
        **train_extra,
    )
    if dataset_cls is FailureConditionedGeneratorActionWindowDataset:
        val_dataset = dataset_cls(
            archive_root=val_archive_root,
            preset=preset,
            split="val",
            route_balance=False,
            failure_sidecar_paths=_as_paths(cfg.get("val_failure_sidecar_paths")),
            max_failure_windows=cfg.get("max_val_failure_windows"),
            append_outcome_to_context=bool(cfg.get("append_outcome_to_context", True)),
            **dataset_kwargs,
        )
    else:
        val_dataset = GeneratorActionWindowDataset(
            archive_root=val_archive_root,
            preset=preset,
            split="val",
            route_balance=False,
            **dataset_kwargs,
        )
    if debug_overfit:
        train_dataset = _subset(train_dataset, int(cfg.get("debug_windows", 32)))
        val_dataset = train_dataset
        batch_size = min(batch_size, len(train_dataset))
    else:
        if cfg.get("max_train_windows") is not None:
            train_dataset = _random_subset(train_dataset, int(cfg["max_train_windows"]), seed=seed)
        if cfg.get("max_val_windows") is not None:
            val_dataset = _random_subset(val_dataset, int(cfg["max_val_windows"]), seed=seed + 1)
    action_low = np.asarray(cfg.get("action_low", [-0.8, -0.8, -4.0]), dtype=np.float32)
    action_high = np.asarray(cfg.get("action_high", [0.8, 0.8, 4.0]), dtype=np.float32)
    action_bounds = (action_low, action_high)
    _log_event(
        "dataset_ready",
        model=model_name,
        train_archive_root=None if train_archive_root is None else str(train_archive_root),
        val_archive_root=None if val_archive_root is None else str(val_archive_root),
        train_windows=len(train_dataset),
        val_windows=len(val_dataset),
        horizon=horizon,
        failure_conditioned=bool(cfg.get("failure_conditioned", False)),
    )
    norm_started = time.perf_counter()
    normalizer = NormalizationStats.fit(train_dataset, action_bounds=action_bounds)
    _log_event("normalization_ready", model=model_name, elapsed_sec=round(time.perf_counter() - norm_started, 3))
    normalization_path = output_dir / "normalization.json"
    normalizer.save(normalization_path)
    normalization_hash = _file_sha256(normalization_path)[:16]

    first = _first_item(train_dataset)
    cfg["context_dim"] = int(first["context"].shape[0])
    cfg["H"] = horizon
    cfg["horizon"] = horizon
    cfg["action_dim"] = int(first["action_seq"].shape[-1])
    cfg["device"] = _device(str(cfg.get("device", "cpu")))
    resume_path = cfg.get("resume")
    if resume_path:
        cfg["checkpoint_path"] = str(resume_path)
    else:
        cfg.pop("checkpoint_path", None)
    model = get_generator(model_name, cfg)
    if hasattr(model, "normalizer"):
        model.normalizer = normalizer

    pin_memory = str(model.device).startswith("cuda")
    prefetch_factor = int(cfg.get("prefetch_factor", 4))
    train_loader = _make_loader(
        train_dataset,
        batch_size,
        shuffle=not debug_overfit,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
    epochs = int(cfg.get("epochs", 3))
    if debug_overfit:
        epochs = max(1, epochs)
    _log_event(
        "training_start",
        model=model_name,
        device=str(model.device),
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        output_dir=str(output_dir),
    )
    history = {"train": [], "val": []}
    best_val = float("inf")
    checkpoints_dir = output_dir / "checkpoints"
    best_checkpoint_path = checkpoints_dir / "best.pt"
    final_checkpoint_path = checkpoints_dir / "final.pt"
    try:
        for epoch_index in range(epochs):
            epoch_started = time.perf_counter()
            train_metrics = _epoch(model_name, model, train_loader, normalizer, optimizer)
            with torch.no_grad():
                val_metrics = _epoch(model_name, model, val_loader, normalizer, None)
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            val_loss = float(val_metrics.get("loss", train_metrics.get("loss", float("inf"))))
            if val_loss < best_val:
                best_val = val_loss
                model.save_checkpoint(best_checkpoint_path)
            _log_event(
                "epoch_complete",
                model=model_name,
                epoch=epoch_index + 1,
                epochs=epochs,
                train=train_metrics,
                val=val_metrics,
                best_val=best_val,
                elapsed_sec=round(time.perf_counter() - epoch_started, 3),
            )
    finally:
        _shutdown_loader(train_loader)
        _shutdown_loader(val_loader)
    model.save_checkpoint(final_checkpoint_path)

    metrics_path = output_dir / "metrics" / "training_curves.json"
    _write_json(metrics_path, history)
    hash_payload = {
        "model": model_name,
        "config": cfg,
        "dataset": {
            "train_archive_root": None if train_archive_root is None else str(train_archive_root),
            "val_archive_root": None if val_archive_root is None else str(val_archive_root),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "failure_conditioned": bool(cfg.get("failure_conditioned", False)),
            "failure_sidecar_paths": [str(path) for path in _as_paths(cfg.get("failure_sidecar_paths", cfg.get("train_failure_sidecar_paths")))],
        },
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "normalization": normalizer.to_dict(),
        "normalization_hash": normalization_hash,
    }
    config_hash = stable_config_hash(hash_payload)
    config_hash_path = output_dir / "metrics" / "config_hash.json"
    _write_json(config_hash_path, {"config_hash": config_hash, **hash_payload})
    sample_diagnostics_path = output_dir / "plots" / "sample_diagnostics.json"
    sample_payload = _sample_diagnostics(model_name, model, train_dataset, action_bounds, sample_diagnostics_path)
    result = {
        "model": model_name,
        "checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "normalization_path": str(normalization_path),
        "normalization_hash": normalization_hash,
        "history_path": str(metrics_path),
        "config_hash_path": str(config_hash_path),
        "sample_diagnostics_path": str(sample_diagnostics_path),
        "config_hash": config_hash,
        "dataset": hash_payload["dataset"],
        "history": history,
        "sample_diagnostics": sample_payload,
    }
    result_path = output_dir / "metrics" / "training_result.json"
    _write_json(result_path, {k: v for k, v in result.items() if k not in {"history", "sample_diagnostics"}})
    result["result_path"] = str(result_path)
    return result
