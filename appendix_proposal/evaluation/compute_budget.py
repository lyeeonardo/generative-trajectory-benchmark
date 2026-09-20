"""Compute-budget and fairness utilities for Generator benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from generators.base import ProposalBatch, ProposalGenerator
from generators.utils import stable_config_hash

from mujoco_task.config import get_preset

SCORER_KEYS = (
    "w_goal_terminal",
    "w_goal_path",
    "w_progress",
    "w_collision",
    "w_fall_out",
    "w_clearance",
    "w_boundary",
    "w_smooth",
    "w_control",
    "w_timeout",
    "w_dyn",
    "w_epistemic",
)

BENCHMARK_INVARIANT_HASH_KEYS = (
    "scorer_config_hash",
    "environment_config_hash",
    "split_file_hash",
)


@dataclass(frozen=True)
class ComputeBudget:
    mode: str
    K: int
    proposal_time_budget_sec: float | None = None

    @classmethod
    def equal_k(cls, K: int) -> "ComputeBudget":
        return cls(mode="equal_k", K=int(K), proposal_time_budget_sec=None)

    @classmethod
    def equal_time(cls, proposal_time_budget_sec: float, *, max_K: int = 1024) -> "ComputeBudget":
        return cls(mode="equal_time", K=int(max_K), proposal_time_budget_sec=float(proposal_time_budget_sec))


def propose_under_budget(
    generator: ProposalGenerator,
    context,
    *,
    budget: ComputeBudget,
    H: int,
    action_dim: int,
    action_bounds: tuple[np.ndarray, np.ndarray],
    seed: int | None = None,
) -> ProposalBatch:
    if budget.mode == "equal_k":
        return generator.propose(context, budget.K, H, action_dim, action_bounds, seed=seed)
    if budget.mode != "equal_time":
        raise ValueError(f"Unknown compute budget mode {budget.mode!r}.")
    # Conservative equal-time smoke implementation: request one candidate at a time
    # until the proposal budget is exhausted or max_K is reached.
    import time

    started = time.perf_counter()
    actions = []
    log_probs = []
    candidate_index = 0
    while candidate_index < budget.K:
        if time.perf_counter() - started >= float(budget.proposal_time_budget_sec or 0.0) and candidate_index > 0:
            break
        batch = generator.propose(context, 1, H, action_dim, action_bounds, seed=None if seed is None else seed + candidate_index)
        actions.append(np.asarray(batch.actions, dtype=np.float32)[0])
        if batch.log_prob is not None:
            log_probs.append(float(np.asarray(batch.log_prob).reshape(-1)[0]))
        candidate_index += 1
    if not actions:
        batch = generator.propose(context, 1, H, action_dim, action_bounds, seed=seed)
        actions.append(np.asarray(batch.actions, dtype=np.float32)[0])
        if batch.log_prob is not None:
            log_probs.append(float(np.asarray(batch.log_prob).reshape(-1)[0]))
    elapsed = time.perf_counter() - started
    log_prob = np.asarray(log_probs, dtype=np.float32) if len(log_probs) == len(actions) else None
    return ProposalBatch(
        actions=np.asarray(actions, dtype=np.float32),
        log_prob=log_prob,
        diagnostics={"budget_mode": "equal_time", "requested_max_K": budget.K, "actual_K": len(actions)},
        sample_time_sec=elapsed,
    )


def scorer_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in SCORER_KEYS if key in config}


def benchmark_invariant_payload(config: dict[str, Any]) -> dict[str, Any]:
    preset = get_preset(str(config.get("preset", "smoke")))
    return {
        "preset": preset.name,
        "split": str(config.get("split", "test")),
        "H": int(config.get("H", config.get("horizon", preset.dataset.horizon))),
        "W": int(config.get("W", config.get("execution_window", 1))),
        "action_low": list(config.get("action_low", [-preset.sim.action_max_speed, -preset.sim.action_max_speed, -preset.sim.action_max_omega])),
        "action_high": list(config.get("action_high", [preset.sim.action_max_speed, preset.sim.action_max_speed, preset.sim.action_max_omega])),
        "sim": asdict(preset.sim),
        "scene": asdict(preset.scene),
        "max_steps": int(preset.sim.max_steps),
        "physics_backend": str(config.get("physics_backend", "mujoco_rigid")),
        "physics_params": dict(config.get("physics_params") or {}),
        "sim_params": dict(config.get("sim_params") or {}),
        "campaign_manifest_hash": config.get("campaign_manifest_hash"),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_archive_root(config: dict[str, Any], preset: str, split: str) -> Path:
    split_key = {"validation": "val", "nominal_test": "test", "heldout_test": "test"}.get(str(split), str(split))
    explicit = config.get(f"{split_key}_archive_root") or config.get(f"{split}_archive_root")
    if explicit:
        return Path(explicit)
    dataset_root = config.get("dataset_root")
    if dataset_root:
        return Path(dataset_root) / split_key
    resolved = get_preset(preset)
    return resolved.paths.data_dir / resolved.name / split_key


def split_file_hash(preset: str, split: str, root: str | Path | None = None) -> str:
    archive_root = Path(root) if root is not None else _split_archive_root({}, preset, split)
    digest = hashlib.sha256()
    for filename in ("episodes.npz", "steps.npz", "meta.json"):
        path = archive_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(filename.encode("utf-8"))
        digest.update(file_sha256(path).encode("utf-8"))
    scenes_path = archive_root / "scenes.npz"
    if scenes_path.exists():
        digest.update(b"scenes.npz")
        digest.update(file_sha256(scenes_path).encode("utf-8"))
    return digest.hexdigest()[:16]


def optional_file_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return file_sha256(candidate)[:16]


def _normalization_path(config: dict[str, Any]) -> str | Path | None:
    explicit = config.get("normalization_path")
    if explicit is not None:
        return explicit
    checkpoint = config.get("checkpoint_path")
    if checkpoint is None:
        return None
    checkpoint_path = Path(checkpoint)
    model_root = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    candidate = model_root / "normalization.json"
    return candidate if candidate.exists() else None


def compute_benchmark_hashes(config: dict[str, Any]) -> dict[str, str | None]:
    preset = str(config.get("preset", "smoke"))
    split = str(config.get("split", "test"))
    archive_root = _split_archive_root(config, preset, split)
    benchmark_manifest_path = config.get("campaign_manifest_path") or config.get("primary_manifest_path")
    if benchmark_manifest_path:
        split_hash = optional_file_hash(benchmark_manifest_path)
    else:
        try:
            split_hash = split_file_hash(preset, split, archive_root)
        except FileNotFoundError:
            split_hash = None
    return {
        "scorer_config_hash": stable_config_hash(scorer_config_payload(config)),
        "environment_config_hash": stable_config_hash(benchmark_invariant_payload(config)),
        "split_file_hash": split_hash,
        "checkpoint_hash": optional_file_hash(config.get("checkpoint_path")),
        "normalization_hash": optional_file_hash(_normalization_path(config)),
        "campaign_manifest_hash": config.get("campaign_manifest_hash"),
    }


def assert_matching_benchmark_hashes(records: dict[str, dict[str, str | None]]) -> None:
    if not records:
        raise ValueError("No benchmark hash records supplied.")
    reference_name, reference = next(iter(records.items()))
    for model_name, hashes in records.items():
        for key in BENCHMARK_INVARIANT_HASH_KEYS:
            expected = reference.get(key)
            actual = hashes.get(key)
            if actual != expected:
                raise ValueError(
                    f"Benchmark fairness mismatch for {model_name}: {key}={actual} differs from "
                    f"{reference_name} {key}={expected}."
                )


def save_benchmark_hashes(records: dict[str, dict[str, str | None]], path: str | Path) -> None:
    assert_matching_benchmark_hashes(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
