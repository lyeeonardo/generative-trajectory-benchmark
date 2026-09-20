#!/usr/bin/env python
"""Verify and select Step 4 paper generator checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.archive_dataset import OUTCOME_LABELS, outcome_one_hot
from evaluation.compute_budget import file_sha256
from evaluation.generator_benchmark import run_generator_benchmark
from generators.registry import get_generator
from generators.utils import validate_proposal_batch
from mujoco_task.config import get_preset
from stage1_config import load_config


PAPER_MODELS = (
    "bc_mdn_aif",
    "cvae_aif",
    "transformer_aif",
    "diffusion_policy_aif",
    "flow_matching_aif",
)

MIN_HIDDEN_DIM = {
    "bc_mdn_aif": 256,
    "cvae_aif": 256,
    "transformer_aif": 512,
    "diffusion_policy_aif": 512,
    "flow_matching_aif": 512,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _load_benchmark_config(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    common_path = cfg.get("common_config")
    if not common_path:
        return cfg
    common_file = Path(common_path)
    if not common_file.is_absolute():
        common_file = path.parent / common_file if (path.parent / common_file).exists() else ROOT / common_file
    merged = load_config(common_file)
    merged.update({key: value for key, value in cfg.items() if key != "common_config"})
    return merged


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _ensure(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _checkpoint_stats(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", {})
    tensor_count = 0
    parameter_count = 0
    for value in state.values():
        if torch.is_tensor(value):
            tensor_count += 1
            parameter_count += int(value.numel())
    return {
        "sha256_16": file_sha256(path)[:16],
        "size_bytes": int(path.stat().st_size),
        "state_tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "has_config": isinstance(payload.get("config"), dict),
        "has_normalizer": payload.get("normalizer") is not None,
        "checkpoint_config": payload.get("config") if isinstance(payload.get("config"), dict) else {},
    }


def _shared_action_bounds(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    preset = get_preset(str(config.get("preset", "run")))
    low = np.asarray(
        config.get("action_low", [-preset.sim.action_max_speed, -preset.sim.action_max_speed, -preset.sim.action_max_omega]),
        dtype=np.float32,
    )
    high = np.asarray(
        config.get("action_high", [preset.sim.action_max_speed, preset.sim.action_max_speed, preset.sim.action_max_omega]),
        dtype=np.float32,
    )
    return low, high


def _success_context(context_dim: int) -> np.ndarray:
    context = np.zeros((int(context_dim),), dtype=np.float32)
    outcome = outcome_one_hot("success")
    if context.shape[0] >= outcome.shape[0]:
        context[-outcome.shape[0] :] = outcome
    return context


def _proposal_validation(model: str, checkpoint_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    generator = get_generator(model, {"checkpoint_path": str(checkpoint_path), "device": "cpu"})
    model_config = getattr(generator, "config", None)
    context_dim = int(getattr(model_config, "context_dim", config.get("context_dim", 0)))
    horizon = int(getattr(model_config, "horizon", config.get("H", 8)))
    action_dim = int(getattr(model_config, "action_dim", 3))
    batch = generator.propose(_success_context(context_dim), 4, horizon, action_dim, _shared_action_bounds(config), seed=123)
    validate_proposal_batch(batch, K=4, H=horizon, action_dim=action_dim, action_bounds=_shared_action_bounds(config))
    actions = np.asarray(batch.actions, dtype=np.float32)
    return {
        "context_dim": context_dim,
        "queried_outcome": "success" if context_dim >= len(OUTCOME_LABELS) else "unconditioned",
        "shape": list(actions.shape),
        "action_min": actions.min(axis=(0, 1)).astype(float).tolist(),
        "action_max": actions.max(axis=(0, 1)).astype(float).tolist(),
        "sample_time_sec": float(batch.sample_time_sec),
        "diagnostics": dict(batch.diagnostics),
    }


def _artifact_audit(
    *,
    manifest_path: Path,
    paper_config_path: Path,
    models: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest = _read_json(manifest_path)
    paper_cfg = _load_benchmark_config(paper_config_path)
    checkpoint_paths = paper_cfg.get("checkpoint_paths", {})
    _ensure(manifest.get("status") == "complete", errors, f"{_rel(manifest_path)} status is not complete.")
    _ensure(bool(paper_cfg.get("outcome_conditioning", False)), errors, "Paper config must enable outcome_conditioning.")
    _ensure(str(paper_cfg.get("control_outcome_label", "")) == "success", errors, "Paper config must query generators with y=success.")

    records: dict[str, Any] = {}
    for model in models:
        result = dict((manifest.get("results") or {}).get(model) or {})
        _ensure(result.get("status") == "complete", errors, f"{model} training manifest entry is not complete.")
        config_path = ROOT / str(result.get("config_path", ""))
        cfg = load_config(config_path) if config_path.exists() else {}
        _ensure(bool(cfg.get("failure_conditioned", False)), errors, f"{model} config is not failure_conditioned.")
        _ensure(int(cfg.get("hidden_dim", 0)) >= MIN_HIDDEN_DIM[model], errors, f"{model} hidden_dim is below Step 4 policy.")
        sidecar_spec = cfg.get("failure_sidecar_paths")
        _ensure(bool(sidecar_spec), errors, f"{model} config has no failure_sidecar_paths.")

        best_path = ROOT / str(result.get("best_checkpoint_path", ""))
        final_path = ROOT / str(result.get("final_checkpoint_path", ""))
        normalization_path = ROOT / str(result.get("normalization_path", ""))
        training_result_path = ROOT / str(result.get("training_result_path", ""))
        config_hash_path = ROOT / str(result.get("config_hash_path", ""))
        sample_path = ROOT / str(result.get("sample_diagnostics_path", ""))
        for label, path in (
            ("best checkpoint", best_path),
            ("final checkpoint", final_path),
            ("normalization", normalization_path),
            ("training result", training_result_path),
            ("config hash", config_hash_path),
            ("sample diagnostics", sample_path),
        ):
            _ensure(path.exists() and path.stat().st_size > 0, errors, f"{model} missing {label}: {_rel(path)}.")

        training_result = _read_json(training_result_path) if training_result_path.exists() else {}
        config_hash = _read_json(config_hash_path) if config_hash_path.exists() else {}
        sample_diag = _read_json(sample_path) if sample_path.exists() else {}
        dataset = dict(training_result.get("dataset") or result.get("dataset") or {})
        _ensure(bool(dataset.get("failure_conditioned", False)), errors, f"{model} training_result does not record failure_conditioned=true.")
        sidecars = [ROOT / str(path) for path in dataset.get("failure_sidecar_paths", [])]
        _ensure(len(sidecars) >= 12, errors, f"{model} should record all 12 failure sidecars.")
        missing_sidecars = [_rel(path) for path in sidecars if not path.exists()]
        _ensure(not missing_sidecars, errors, f"{model} has missing failure sidecars: {missing_sidecars[:3]}.")
        _ensure(int(dataset.get("train_windows", 0)) > 0 and int(dataset.get("val_windows", 0)) > 0, errors, f"{model} has empty train/val windows.")
        if normalization_path.exists():
            _ensure(file_sha256(normalization_path)[:16] == result.get("normalization_hash"), errors, f"{model} normalization hash mismatch.")
        _ensure(config_hash.get("config_hash") == result.get("config_hash"), errors, f"{model} config hash mismatch.")
        _ensure(sample_diag.get("shape") == [4, int(cfg.get("H", cfg.get("horizon", 8))), 3], errors, f"{model} sample diagnostics shape mismatch.")

        best_stats = _checkpoint_stats(best_path) if best_path.exists() else {}
        final_stats = _checkpoint_stats(final_path) if final_path.exists() else {}
        _ensure(int(best_stats.get("parameter_count", 0)) > 0, errors, f"{model} best checkpoint has no parameters.")
        _ensure(int(final_stats.get("parameter_count", 0)) > 0, errors, f"{model} final checkpoint has no parameters.")
        if model == "flow_matching_aif":
            _ensure(int(best_stats.get("parameter_count", 0)) > 1000, errors, "flow_matching_aif checkpoint looks too small to be a real model.")

        config_checkpoint = ROOT / str(checkpoint_paths.get(model, ""))
        known_checkpoints = {best_path, final_path}
        configured_candidate = "best" if config_checkpoint == best_path else "final" if config_checkpoint == final_path else "unknown"
        _ensure(config_checkpoint in known_checkpoints, errors, f"{model} paper config does not point at a trained Step 4 checkpoint.")
        proposal = _proposal_validation(model, config_checkpoint, {**paper_cfg, **cfg}) if config_checkpoint.exists() else {}
        records[model] = {
            "config_path": _rel(config_path),
            "configured_checkpoint_path": _rel(config_checkpoint),
            "configured_candidate": configured_candidate,
            "best_checkpoint_path": _rel(best_path),
            "final_checkpoint_path": _rel(final_path),
            "normalization_path": _rel(normalization_path),
            "training_result_path": _rel(training_result_path),
            "config_hash_path": _rel(config_hash_path),
            "sample_diagnostics_path": _rel(sample_path),
            "config_hash": result.get("config_hash"),
            "normalization_hash": result.get("normalization_hash"),
            "dataset": dataset,
            "best_checkpoint": best_stats,
            "final_checkpoint": final_stats,
            "proposal_validation": proposal,
        }
    return records, errors


def _read_episode_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _candidate_metrics(output_dir: Path, models: tuple[str, ...]) -> dict[str, Any]:
    rows = _read_episode_rows(output_dir / "per_episode.csv")
    failures_doc = _read_json(output_dir / "failures.json") if (output_dir / "failures.json").exists() else {"failures": []}
    failures = failures_doc.get("failures", [])
    metrics: dict[str, Any] = {}
    for model in models:
        model_rows = [row for row in rows if str(row.get("model")) == model and str(row.get("status", "ok")) == "ok"]
        distances = [float(row.get("final_distance_to_goal", "nan")) for row in model_rows]
        metrics[model] = {
            "episodes": len(model_rows),
            "successes": sum(_bool(row.get("success", False)) for row in model_rows),
            "collisions": sum(_bool(row.get("collision", False)) for row in model_rows),
            "fall_out": sum(_bool(row.get("fall_out", False)) for row in model_rows),
            "timeouts": sum(_bool(row.get("timeout", False)) for row in model_rows),
            "mean_final_distance": float(np.nanmean(distances)) if distances else float("nan"),
            "failures": len([failure for failure in failures if str(failure.get("model")) == model]),
        }
    return metrics


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    episodes = max(int(metrics.get("episodes", 0)), 1)
    success_rate = float(metrics.get("successes", 0)) / episodes
    failure_rate = float(metrics.get("failures", 0)) / episodes
    collision_rate = float(metrics.get("collisions", 0)) / episodes
    fall_out_rate = float(metrics.get("fall_out", 0)) / episodes
    timeout_rate = float(metrics.get("timeouts", 0)) / episodes
    distance = float(metrics.get("mean_final_distance", float("inf")))
    if not np.isfinite(distance):
        distance = float("inf")
    return (success_rate, -failure_rate, -collision_rate, -fall_out_rate, -timeout_rate, -distance)


def _run_closed_loop_selection(
    *,
    paper_config_path: Path,
    output_root: Path,
    manifest_records: dict[str, Any],
    models: tuple[str, ...],
    scene_ids: list[int],
    seeds: list[int],
    candidate_budget: int,
    max_steps: int,
    device: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    base_cfg = _load_benchmark_config(paper_config_path)
    candidate_runs: dict[str, Any] = {}
    for candidate in ("best", "final"):
        checkpoint_paths = {
            model: manifest_records[model][f"{candidate}_checkpoint_path"]
            for model in models
        }
        model_configs = dict(base_cfg.get("model_configs") or {})
        for model in models:
            merged = dict(model_configs.get(model) or {})
            merged["device"] = device
            model_configs[model] = merged
        output_dir = output_root / candidate
        cfg = {
            **base_cfg,
            "experiment": "step4_checkpoint_selection",
            "variant": candidate,
            "models": list(models),
            "checkpoint_paths": checkpoint_paths,
            "model_configs": model_configs,
            "proposal_source": "generator",
            "K": int(candidate_budget),
            "scene_ids": [int(value) for value in scene_ids],
            "seeds": [int(value) for value in seeds],
            "max_steps": int(max_steps),
            "output_dir": str(output_dir),
            "save_visuals": False,
        }
        run_generator_benchmark(cfg)
        metrics = _candidate_metrics(output_dir, models)
        candidate_runs[candidate] = {
            "output_dir": _rel(output_dir),
            "checkpoint_paths": checkpoint_paths,
            "metrics": metrics,
        }
        for model, values in metrics.items():
            _ensure(int(values.get("episodes", 0)) == len(scene_ids) * len(seeds), errors, f"{candidate} {model} closed-loop smoke episode count mismatch.")
            _ensure(int(values.get("failures", 0)) == 0, errors, f"{candidate} {model} closed-loop smoke had benchmark failures.")

    selections: dict[str, Any] = {}
    for model in models:
        best_metrics = candidate_runs["best"]["metrics"][model]
        final_metrics = candidate_runs["final"]["metrics"][model]
        if _selection_key(final_metrics) > _selection_key(best_metrics):
            selected = "final"
        else:
            selected = "best"
        selections[model] = {
            "selected_candidate": selected,
            "selected_checkpoint_path": candidate_runs[selected]["checkpoint_paths"][model],
            "selection_policy": "closed_loop_success_then_failure_safety_distance_then_validation_best_tie_break",
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
        }
    return {"candidate_runs": candidate_runs, "selections": selections}, errors


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Step 4 Checkpoint Verification",
        "",
        f"Status: {report['status']}",
        "",
        "## Selected Checkpoints",
        "",
        "| model | selected | checkpoint | episodes | successes | failures |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    selections = (report.get("closed_loop_selection") or {}).get("selections", {})
    for model in report.get("models", list(PAPER_MODELS)):
        selected = selections.get(model, {})
        candidate = selected.get("selected_candidate", "not-run")
        metrics = selected.get(f"{candidate}_metrics", {}) if candidate in {"best", "final"} else {}
        lines.append(
            "| {model} | {candidate} | {path} | {episodes} | {successes} | {failures} |".format(
                model=model,
                candidate=candidate,
                path=selected.get("selected_checkpoint_path", report.get("artifacts", {}).get(model, {}).get("best_checkpoint_path", "")),
                episodes=metrics.get("episodes", ""),
                successes=metrics.get("successes", ""),
                failures=metrics.get("failures", ""),
            )
        )
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Step 4 paper generator checkpoints.")
    parser.add_argument("--manifest", default="outputs/generators/paper_step4_training_manifest.json")
    parser.add_argument("--paper_config", default="configs/paper/id_n_sweep.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated model subset. Defaults to original paper models.")
    parser.add_argument("--output", default="outputs/generators/paper_step4_verification.json")
    parser.add_argument("--report", default="outputs/generators/paper_step4_verification.md")
    parser.add_argument("--closed_loop_output_root", default="outputs/paper/step4_checkpoint_selection")
    parser.add_argument("--skip_closed_loop_selection", action="store_true")
    parser.add_argument("--scene_ids", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.models:
        requested_models = tuple(part.strip() for part in str(args.models).split(",") if part.strip())
        valid_models = set(PAPER_MODELS)
        unknown = sorted(set(requested_models) - valid_models)
        if unknown:
            raise SystemExit(f"Unknown Step 4 verification model(s): {unknown}")
    else:
        requested_models = PAPER_MODELS

    artifacts, errors = _artifact_audit(
        manifest_path=ROOT / args.manifest,
        paper_config_path=ROOT / args.paper_config,
        models=requested_models,
    )
    selection: dict[str, Any] | None = None
    if not args.skip_closed_loop_selection:
        selection, selection_errors = _run_closed_loop_selection(
            paper_config_path=ROOT / args.paper_config,
            output_root=ROOT / args.closed_loop_output_root,
            manifest_records=artifacts,
            models=requested_models,
            scene_ids=args.scene_ids,
            seeds=args.seeds,
            candidate_budget=args.K,
            max_steps=args.max_steps,
            device=str(args.device),
        )
        errors.extend(selection_errors)
        for model, record in selection["selections"].items():
            configured = artifacts[model].get("configured_checkpoint_path")
            _ensure(
                str(configured) == str(record["selected_checkpoint_path"]),
                errors,
                f"{model} configured checkpoint {configured} does not match closed-loop selected {record['selected_checkpoint_path']}.",
            )

    report = {
        "status": "passed" if not errors else "failed",
        "models": list(requested_models),
        "artifact_manifest": args.manifest,
        "paper_config": args.paper_config,
        "artifacts": artifacts,
        "closed_loop_selection": selection,
        "errors": errors,
    }
    _write_json(ROOT / args.output, report)
    _write_markdown(ROOT / args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
