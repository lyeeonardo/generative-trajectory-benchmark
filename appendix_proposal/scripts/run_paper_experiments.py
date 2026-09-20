#!/usr/bin/env python
"""Run the canonical Step 5 paper experiment suite."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.generator_benchmark import run_generator_benchmark
from evaluation.metrics import summarize_episode_logs
from scripts.build_experiment_summary import build_experiment_summary
from scripts.build_paper_figures import build_paper_figures
from scripts.build_paper_ready_report import build_paper_ready_report
from scripts.make_generator_report import make_generator_report
from stage1_config import load_config


PAPER_EXPERIMENTS = ("id_n_sweep", "heldout_start_goal")
LEARNED_MODELS = ("bc_mdn_aif", "cvae_aif", "transformer_aif", "diffusion_policy_aif", "flow_matching_aif")
FULL_CANDIDATE_BUDGETS = (8, 16, 32, 64, 128, 256)
SMOKE_CANDIDATE_BUDGETS = (4,)
FULL_ID_CEM_REFERENCE_BUDGETS = (512, 1024)
SMOKE_ID_CEM_REFERENCE_BUDGETS: tuple[int, ...] = ()
GENERATOR_ONLY_BUDGETS = (1,)


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    variant: str
    candidate_budget: int
    models: tuple[str, ...]
    config: dict[str, Any]
    output_dir: Path


def _as_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = load_config(config_path)
    common_path = config.get("common_config")
    if not common_path:
        return config
    common_file = Path(common_path)
    if not common_file.is_absolute():
        common_file = config_path.parent / common_file if (config_path.parent / common_file).exists() else ROOT / common_file
    merged = load_config(common_file)
    merged.update({key: value for key, value in config.items() if key != "common_config"})
    return merged


def _config_path(experiment: str) -> Path:
    return ROOT / "configs" / "paper" / f"{experiment}.yaml"


def _profile_scene_ids(experiment: str, profile: str, config: dict[str, Any], override: list[int] | None) -> list[int]:
    if override:
        return [int(item) for item in override]
    if profile == "smoke":
        if experiment == "heldout_start_goal":
            return [110]
        return [0]
    return [int(item) for item in config.get("scene_ids", [0])]


def _profile_seeds(experiment: str, profile: str, config: dict[str, Any], override: list[int] | None) -> list[int]:
    if override:
        return [int(item) for item in override]
    if profile == "smoke":
        return [0]
    configured = [int(item) for item in config.get("seeds", [0])]
    if experiment in {"id_n_sweep", "heldout_start_goal"} and len(set(configured)) < 3:
        return [0, 1, 2]
    return configured


def _profile_budgets(profile: str, override: list[int] | None) -> list[int]:
    if override:
        return [int(item) for item in override]
    return list(SMOKE_CANDIDATE_BUDGETS if profile == "smoke" else FULL_CANDIDATE_BUDGETS)


def _profile_cem_reference_budgets(profile: str, override: list[int] | None, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    if override is not None:
        return [int(item) for item in override]
    return list(SMOKE_ID_CEM_REFERENCE_BUDGETS if profile == "smoke" else FULL_ID_CEM_REFERENCE_BUDGETS)


def _profile_generator_only_budgets(override: list[int] | None, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    if override is not None:
        return [int(item) for item in override]
    return list(GENERATOR_ONLY_BUDGETS)


def _horizon_checkpoint_path(checkpoint_root: str | Path, model: str) -> Path:
    return Path(checkpoint_root) / model / "checkpoints" / "best.pt"


def _apply_horizon_overrides(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(base)
    horizon = getattr(args, "H", None)
    if horizon is not None:
        cfg["H"] = int(horizon)
        cfg["horizon"] = int(horizon)
    checkpoint_root = getattr(args, "checkpoint_root", None)
    if checkpoint_root:
        checkpoint_paths = dict(cfg.get("checkpoint_paths", {}))
        for model in LEARNED_MODELS:
            checkpoint_paths[model] = str(_horizon_checkpoint_path(checkpoint_root, model))
        cfg["checkpoint_paths"] = checkpoint_paths
    return cfg


def _validate_required_checkpoints(specs: list[RunSpec]) -> None:
    missing: list[str] = []
    for spec in specs:
        checkpoint_paths = spec.config.get("checkpoint_paths", {})
        if not isinstance(checkpoint_paths, dict):
            checkpoint_paths = {}
        for model in spec.models:
            if model not in LEARNED_MODELS:
                continue
            path = checkpoint_paths.get(model)
            if not path or not Path(path).exists():
                missing.append(f"{spec.experiment}/{spec.variant}/K{spec.candidate_budget}:{model}:{path or 'unset'}")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing required checkpoint(s): {preview}{suffix}")


def _with_common_run_fields(
    base: dict[str, Any],
    *,
    experiment: str,
    variant: str,
    output_dir: Path,
    candidate_budget: int,
    models: tuple[str, ...],
    scene_ids: list[int],
    seeds: list[int],
    max_steps: int | None,
) -> dict[str, Any]:
    cfg = dict(base)
    cfg.update(
        {
            "experiment": experiment,
            "variant": variant,
            "models": list(models),
            "K": int(candidate_budget),
            "output_dir": str(output_dir),
            "scene_ids": scene_ids,
            "seeds": seeds,
            "save_visuals": False,
            "use_generator_context": True,
        }
    )
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    return cfg


def _id_or_heldout_specs(
    experiment: str,
    base: dict[str, Any],
    *,
    budgets: list[int],
    learned_models: tuple[str, ...],
    scene_ids: list[int],
    seeds: list[int],
    max_steps: int | None,
    root: Path,
    generator_only_budgets: list[int] | None = None,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for budget in budgets:
        variants = (
            ("aif_cem", ("cem_aif",), {"proposal_source": "aif_cem"}),
            ("generator_aif_score", learned_models, {"proposal_source": "generator"}),
            (
                "gated_hybrid",
                learned_models,
                {
                    "proposal_source": "mixture",
                    "fallback_model": "cem_aif",
                    "gate_enabled": True,
                    "omega_init": 1.0,
                    "omega_min": 0.0,
                    "omega_max": 1.0,
                    "rho_min": 0.15,
                    "rho_max": 0.85,
                },
            ),
        )
        for variant, models, overrides in variants:
            out = root / experiment / "runs" / f"{variant}_K{int(budget)}"
            cfg = _with_common_run_fields(
                {**base, **overrides},
                experiment=experiment,
                variant=variant,
                output_dir=out,
                candidate_budget=budget,
                models=tuple(models),
                scene_ids=scene_ids,
                seeds=seeds,
                max_steps=max_steps,
            )
            specs.append(RunSpec(experiment, variant, int(budget), tuple(models), cfg, out))
    for budget in generator_only_budgets or []:
        out = root / experiment / "runs" / f"pure_generator_only_K{int(budget)}"
        cfg = _with_common_run_fields(
            {
                **base,
                "proposal_source": "generator",
                "selection_mode": "generator_only_first",
                "controller_mode": "generator_only_first",
            },
            experiment=experiment,
            variant="pure_generator_only",
            output_dir=out,
            candidate_budget=int(budget),
            models=learned_models,
            scene_ids=scene_ids,
            seeds=seeds,
            max_steps=max_steps,
        )
        specs.append(RunSpec(experiment, "pure_generator_only", int(budget), tuple(learned_models), cfg, out))
    return specs


def _id_cem_reference_specs(
    base: dict[str, Any],
    *,
    budgets: list[int],
    scene_ids: list[int],
    seeds: list[int],
    max_steps: int | None,
    root: Path,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for budget in budgets:
        out = root / "id_n_sweep" / "runs" / f"aif_cem_reference_K{int(budget)}"
        cfg = _with_common_run_fields(
            {
                **base,
                "proposal_source": "aif_cem",
                "cem_iterations": 2,
                "cem_elite_frac": 0.125,
            },
            experiment="id_n_sweep",
            variant="aif_cem_reference",
            output_dir=out,
            candidate_budget=int(budget),
            models=("cem_aif",),
            scene_ids=scene_ids,
            seeds=seeds,
            max_steps=max_steps,
        )
        specs.append(RunSpec("id_n_sweep", "aif_cem_reference", int(budget), ("cem_aif",), cfg, out))
    return specs


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    experiments = tuple(_as_csv(args.experiments) or PAPER_EXPERIMENTS)
    unknown = sorted(set(experiments) - set(PAPER_EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown paper experiment(s): {unknown}")
    requested_variants = set(_as_csv(getattr(args, "variants", None)))
    learned_models = tuple(_as_csv(args.models) or LEARNED_MODELS)
    budgets = _profile_budgets(args.profile, args.candidate_budgets)
    cem_reference_budgets = _profile_cem_reference_budgets(
        args.profile,
        args.cem_reference_budgets,
        enabled=bool(args.include_cem_reference),
    )
    generator_only_budgets = _profile_generator_only_budgets(
        args.generator_only_budgets,
        enabled=bool(args.include_generator_only),
    )
    root = Path(args.output_root)
    specs: list[RunSpec] = []
    for experiment in experiments:
        base = _apply_horizon_overrides(_load_benchmark_config(_config_path(experiment)), args)
        scene_ids = _profile_scene_ids(experiment, args.profile, base, args.scene_ids)
        seeds = _profile_seeds(experiment, args.profile, base, args.seeds)
        max_steps = args.max_steps if args.max_steps is not None else (2 if args.profile == "smoke" else int(base.get("max_steps", 100)))
        if experiment == "id_n_sweep":
            specs.extend(
                _id_or_heldout_specs(
                    experiment,
                    base,
                    budgets=budgets,
                    learned_models=learned_models,
                    scene_ids=scene_ids,
                    seeds=seeds,
                    max_steps=max_steps,
                    root=root,
                    generator_only_budgets=generator_only_budgets,
                )
            )
            specs.extend(
                _id_cem_reference_specs(
                    base,
                    budgets=cem_reference_budgets,
                    scene_ids=scene_ids,
                    seeds=seeds,
                    max_steps=max_steps,
                    root=root,
                )
            )
        elif experiment == "heldout_start_goal":
            specs.extend(
                _id_or_heldout_specs(
                    experiment,
                    base,
                    budgets=[int(base.get("K", budgets[-1] if budgets else 128)) if args.profile == "full" else budgets[0]],
                    learned_models=learned_models,
                    scene_ids=scene_ids,
                    seeds=seeds,
                    max_steps=max_steps,
                    root=root,
                    generator_only_budgets=generator_only_budgets,
                )
            )
    if requested_variants:
        known_variants = {spec.variant for spec in specs}
        unknown_variants = sorted(requested_variants - known_variants)
        if unknown_variants:
            raise ValueError(f"Unknown variant(s) for selected experiments: {unknown_variants}")
        specs = [spec for spec in specs if spec.variant in requested_variants]
    if bool(getattr(args, "require_checkpoints", False)):
        _validate_required_checkpoints(specs)
    return specs


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _numeric_episode(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = dict(row)
    for key in ("success", "collision", "fall_out", "timeout", "ood"):
        if key in payload:
            payload[key] = _bool_value(payload[key])
    for key, value in list(payload.items()):
        if key in {"model", "run_id", "experiment", "variant", "benchmark_mode", "physics_backend", "status", "terminal_reason", "error"}:
            continue
        try:
            text = str(value)
            if text.strip() == "":
                continue
            payload[key] = float(text)
        except (TypeError, ValueError):
            pass
    return payload


def _aggregate_rows(experiment: str, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        variant = str(row.get("variant", "unknown"))
        budget = str(row.get("K", row.get("candidate_budget", "unknown")))
        model = str(row.get("model", "unknown"))
        key = f"{variant}:K{budget}:{model}"
        groups.setdefault(key, []).append(_numeric_episode(row))
    models: dict[str, Any] = {}
    for key, episodes in groups.items():
        models[key] = {
            "episodes": len(episodes),
            "failures": len([failure for failure in failures if str(failure.get("model", "")) in key]),
            "summary": summarize_episode_logs(episodes),
        }
    return {
        "experiment": experiment,
        "benchmark_mode": "equal_k",
        "physics_backend": "mujoco_rigid",
        "failure_count": len(failures),
        "models": models,
    }


def _actual_rows_from_npz(path: Path, spec: RunSpec) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with np.load(path) as data:
        count = int(data["success"].shape[0])
        for index in range(count):
            tmask = np.asarray(data["trajectory_mask"][index], dtype=bool)
            amask = np.asarray(data["action_mask"][index], dtype=bool)
            success = bool(data["success"][index])
            collision = bool(data["collision"][index])
            fall_out = bool(data["fall_out"][index]) if "fall_out" in data.files else False
            timeout = bool(data["timeout"][index])
            terminal_reason = (
                str(data["terminal_reason"][index])
                if "terminal_reason" in data.files
                else "success"
                if success
                else "collision"
                if collision
                else "fall_out"
                if fall_out
                else "timeout"
                if timeout
                else "running"
            )
            rows.append(
                {
                    "trajectory_xy": np.asarray(data["trajectory_xy"][index][tmask], dtype=np.float32),
                    "actions": np.asarray(data["actions"][index][amask], dtype=np.float32),
                    "goal_xy": np.asarray(data["goal_xy"][index], dtype=np.float32),
                    "obstacle_xy": np.asarray(data["obstacle_xy"][index], dtype=np.float32),
                    "obstacle_radius": float(data["obstacle_radius"][index]),
                    "run_id": f"{spec.experiment}:{spec.variant}:K{spec.candidate_budget}:{str(data['run_id'][index])}",
                    "model": str(data["model"][index]),
                    "experiment": spec.experiment,
                    "variant": spec.variant,
                    "benchmark_mode": str(data["benchmark_mode"][index]),
                    "split": str(data["split"][index]),
                    "preset": str(data["preset"][index]),
                    "scene_id": int(data["scene_id"][index]),
                    "seed": int(data["seed"][index]),
                    "candidate_budget": int(spec.candidate_budget),
                    "success": success,
                    "collision": collision,
                    "fall_out": fall_out,
                    "timeout": timeout,
                    "terminal_reason": terminal_reason,
                    "final_distance_to_goal": float(data["final_distance_to_goal"][index]),
                    "episode_length": int(data["episode_length"][index]),
                    "physics_backend": str(data["physics_backend"][index]) if "physics_backend" in data.files else "mujoco_rigid",
                    "ood": bool(data["ood"][index]) if "ood" in data.files else bool(spec.experiment == "ood_gate"),
                    "true_tilt_lateral": float(data["true_tilt_lateral"][index]) if "true_tilt_lateral" in data.files else np.nan,
                    "true_tilt_longitudinal": float(data["true_tilt_longitudinal"][index]) if "true_tilt_longitudinal" in data.files else np.nan,
                }
            )
    return rows


def _trajectory_npz_payload(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "trajectory_xy": np.zeros((0, 0, 2), dtype=np.float32),
            "trajectory_mask": np.zeros((0, 0), dtype=bool),
            "actions": np.zeros((0, 0, 3), dtype=np.float32),
            "action_mask": np.zeros((0, 0), dtype=bool),
            "goal_xy": np.zeros((0, 2), dtype=np.float32),
            "obstacle_xy": np.zeros((0, 2), dtype=np.float32),
            "obstacle_radius": np.zeros((0,), dtype=np.float32),
        }
    max_t = max(np.asarray(row["trajectory_xy"]).shape[0] for row in rows)
    max_a = max(np.asarray(row["actions"]).shape[0] for row in rows)
    trajectory_xy = np.zeros((len(rows), max_t, 2), dtype=np.float32)
    trajectory_mask = np.zeros((len(rows), max_t), dtype=bool)
    actions = np.zeros((len(rows), max_a, 3), dtype=np.float32)
    action_mask = np.zeros((len(rows), max_a), dtype=bool)
    for index, row in enumerate(rows):
        path = np.asarray(row["trajectory_xy"], dtype=np.float32).reshape(-1, 2)
        action = np.asarray(row["actions"], dtype=np.float32).reshape(-1, 3)
        trajectory_xy[index, : path.shape[0]] = path
        trajectory_mask[index, : path.shape[0]] = True
        actions[index, : action.shape[0]] = action
        action_mask[index, : action.shape[0]] = True
    return {
        "trajectory_xy": trajectory_xy,
        "trajectory_mask": trajectory_mask,
        "actions": actions,
        "action_mask": action_mask,
        "goal_xy": np.asarray([row["goal_xy"] for row in rows], dtype=np.float32),
        "obstacle_xy": np.asarray([row["obstacle_xy"] for row in rows], dtype=np.float32),
        "obstacle_radius": np.asarray([row["obstacle_radius"] for row in rows], dtype=np.float32),
        "run_id": np.asarray([row["run_id"] for row in rows], dtype=str),
        "model": np.asarray([row["model"] for row in rows], dtype=str),
        "experiment": np.asarray([row["experiment"] for row in rows], dtype=str),
        "variant": np.asarray([row["variant"] for row in rows], dtype=str),
        "benchmark_mode": np.asarray([row["benchmark_mode"] for row in rows], dtype=str),
        "split": np.asarray([row["split"] for row in rows], dtype=str),
        "preset": np.asarray([row["preset"] for row in rows], dtype=str),
        "scene_id": np.asarray([row["scene_id"] for row in rows], dtype=np.int64),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "candidate_budget": np.asarray([row["candidate_budget"] for row in rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in rows], dtype=bool),
        "collision": np.asarray([row["collision"] for row in rows], dtype=bool),
        "fall_out": np.asarray([row["fall_out"] for row in rows], dtype=bool),
        "timeout": np.asarray([row["timeout"] for row in rows], dtype=bool),
        "terminal_reason": np.asarray([row["terminal_reason"] for row in rows], dtype=str),
        "final_distance_to_goal": np.asarray([row["final_distance_to_goal"] for row in rows], dtype=np.float32),
        "episode_length": np.asarray([row["episode_length"] for row in rows], dtype=np.int64),
        "physics_backend": np.asarray([row["physics_backend"] for row in rows], dtype=str),
        "ood": np.asarray([row["ood"] for row in rows], dtype=bool),
        "true_tilt_lateral": np.asarray([row["true_tilt_lateral"] for row in rows], dtype=np.float32),
        "true_tilt_longitudinal": np.asarray([row["true_tilt_longitudinal"] for row in rows], dtype=np.float32),
    }


def combine_experiment_outputs(experiment: str, specs: list[RunSpec], output_root: Path) -> dict[str, Any]:
    experiment_dir = output_root / experiment
    metrics_dir = experiment_dir / "metrics"
    per_episode: list[dict[str, Any]] = []
    per_step: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    config_hashes: dict[str, Any] = {}
    model_cards: dict[str, Any] = {}
    run_records: list[dict[str, Any]] = []
    for spec in specs:
        run_key = f"{spec.variant}:K{spec.candidate_budget}"
        run_records.append(
            {
                "variant": spec.variant,
                "candidate_budget": spec.candidate_budget,
                "models": list(spec.models),
                "output_dir": str(spec.output_dir),
            }
        )
        for row in _read_csv(spec.output_dir / "per_episode.csv"):
            row.update({"experiment": experiment, "variant": spec.variant, "candidate_budget": str(spec.candidate_budget), "source_run_dir": str(spec.output_dir)})
            row.setdefault("K", str(spec.candidate_budget))
            per_episode.append(row)
        for row in _read_csv(spec.output_dir / "per_step.csv"):
            row.update({"experiment": experiment, "variant": spec.variant, "candidate_budget": str(spec.candidate_budget), "source_run_dir": str(spec.output_dir)})
            row.setdefault("K", str(spec.candidate_budget))
            per_step.append(row)
        failure_doc = json.loads((spec.output_dir / "failures.json").read_text(encoding="utf-8")) if (spec.output_dir / "failures.json").exists() else {"failures": []}
        for failure in failure_doc.get("failures", []):
            failure.update({"experiment": experiment, "variant": spec.variant, "candidate_budget": spec.candidate_budget, "source_run_dir": str(spec.output_dir)})
            failures.append(failure)
        if (spec.output_dir / "config_hashes.json").exists():
            config_hashes[run_key] = json.loads((spec.output_dir / "config_hashes.json").read_text(encoding="utf-8"))
        if (spec.output_dir / "model_card.json").exists():
            model_cards[run_key] = json.loads((spec.output_dir / "model_card.json").read_text(encoding="utf-8"))
        trajectory_rows.extend(_actual_rows_from_npz(spec.output_dir / "episode_trajectories.npz", spec))

    _write_csv(metrics_dir / "per_episode.csv", per_episode)
    _write_csv(metrics_dir / "per_step.csv", per_step)
    aggregate = _aggregate_rows(experiment, per_episode, failures)
    _write_json(metrics_dir / "aggregate_metrics.json", aggregate)
    _write_json(metrics_dir / "failures.json", {"failures": failures})
    _write_json(metrics_dir / "config_hashes.json", config_hashes)
    _write_json(metrics_dir / "model_card.json", {"runs": run_records, "model_cards": model_cards})
    np.savez_compressed(metrics_dir / "episode_trajectories.npz", **_trajectory_npz_payload(trajectory_rows))
    (metrics_dir / "summary_report.md").write_text(
        f"# {experiment} Metrics\n\nRuns: {len(specs)}\nEpisodes: {len(per_episode)}\nFailures: {len(failures)}\n",
        encoding="utf-8",
    )
    return {
        "experiment": experiment,
        "metrics_dir": str(metrics_dir),
        "episodes": len(per_episode),
        "failures": len(failures),
        "runs": run_records,
    }


def _run_spec_worker(spec: RunSpec, skip_existing: bool) -> dict[str, Any]:
    if skip_existing and (spec.output_dir / "aggregate_metrics.json").exists():
        status = "skipped_existing"
    else:
        run_generator_benchmark(spec.config)
        status = "complete"
    return {
        "experiment": spec.experiment,
        "variant": spec.variant,
        "candidate_budget": spec.candidate_budget,
        "models": list(spec.models),
        "output_dir": str(spec.output_dir),
        "status": status,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    specs = build_run_specs(args)
    manifest_path = output_root / "paper_step5_manifest.json"
    manifest: dict[str, Any] = {
        "profile": args.profile,
        "output_root": str(output_root),
        "runs_requested": len(specs),
        "learned_models": list(_as_csv(args.models) or LEARNED_MODELS),
        "paper_ready_features": {
            "id_heldout_default_min_seeds": 3 if args.profile == "full" and args.seeds is None else None,
            "include_cem_reference": bool(args.include_cem_reference),
            "cem_reference_budgets": [
                spec.candidate_budget
                for spec in specs
                if spec.experiment == "id_n_sweep" and spec.variant == "aif_cem_reference"
            ],
            "include_generator_only": bool(args.include_generator_only),
            "generator_only_budgets": sorted(
                {
                    spec.candidate_budget
                    for spec in specs
                    if spec.variant == "pure_generator_only"
                }
            ),
        },
        "runs": [],
        "experiments": {},
    }
    if int(args.parallel_jobs) <= 1:
        for spec in specs:
            manifest["runs"].append(_run_spec_worker(spec, bool(args.skip_existing)))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.parallel_jobs))) as executor:
            futures = {executor.submit(_run_spec_worker, spec, bool(args.skip_existing)): spec for spec in specs}
            for future in as_completed(futures):
                manifest["runs"].append(future.result())

    for experiment in PAPER_EXPERIMENTS:
        experiment_specs = [spec for spec in specs if spec.experiment == experiment]
        if not experiment_specs:
            continue
        summary = combine_experiment_outputs(experiment, experiment_specs, output_root)
        metrics_dir = Path(summary["metrics_dir"])
        report_path = output_root / experiment / "report.md"
        make_generator_report(metrics_dir, report_path)
        summary.update({"report_path": str(report_path)})
        manifest["experiments"][experiment] = summary

    if args.build_summary:
        figures = build_paper_figures(output_root, "datasets/generator_training")
        manifest["figure_manifest_path"] = figures["manifest"]
        combined = build_experiment_summary(
            output_json=output_root / "paper_results.json",
            output_md=output_root / "paper_results.md",
            paper_root=output_root,
        )
        manifest["combined_summary_path"] = combined["summary_path"]
        manifest["combined_report_path"] = combined["report_path"]
        _write_json(manifest_path, manifest)
        paper_ready = build_paper_ready_report(output_root)
        manifest["paper_ready_report_path"] = paper_ready["report_path"]
        manifest["paper_ready_audit_path"] = paper_ready["json_path"]
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two final Appendix A proposal experiments.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--experiments", default=None, help="Comma-separated subset. Defaults to the two Appendix A experiments.")
    parser.add_argument("--variants", default=None, help="Comma-separated run variant subset after experiment expansion.")
    parser.add_argument("--models", default=None, help="Comma-separated learned model subset. Defaults to all five paper generators.")
    parser.add_argument("--candidate_budgets", nargs="*", type=int, default=None)
    parser.add_argument("--cem_reference_budgets", nargs="*", type=int, default=None, help="ID-only AIF-CEM reference budgets. Defaults to 512 and 1024 in full profile.")
    parser.add_argument("--no_cem_reference", dest="include_cem_reference", action="store_false", default=True)
    parser.add_argument("--generator_only_budgets", nargs="*", type=int, default=None, help="Pure generator-only ablation budgets. Defaults to K=1.")
    parser.add_argument("--no_generator_only", dest="include_generator_only", action="store_false", default=True)
    parser.add_argument("--scene_ids", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--H", "--horizon", dest="H", type=int, default=None, help="Override planning/generator horizon for all selected experiments.")
    parser.add_argument("--checkpoint_root", default=None, help="Root containing <model>/checkpoints/best.pt for horizon-specific learned checkpoints.")
    parser.add_argument("--require_checkpoints", action="store_true", help="Fail before running if any requested learned checkpoint path is missing.")
    parser.add_argument("--output_root", default="outputs/paper")
    parser.add_argument("--parallel_jobs", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_build_summary", dest="build_summary", action="store_false", default=True)
    args = parser.parse_args()
    print(json.dumps(run_suite(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
