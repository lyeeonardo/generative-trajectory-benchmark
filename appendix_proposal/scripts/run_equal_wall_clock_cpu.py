#!/usr/bin/env python
"""CPU-only calibrated equal wall-clock experiment runner.

This runner does not use the existing benchmark_mode="equal_time" placeholder.
It first calibrates a fixed K per model/time-budget on CPU, then evaluates the
standard benchmark at those locked K values while recording the intended budget
and measured wall-clock timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import statistics
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.generator_benchmark import run_generator_benchmark
from stage1_config import load_config


LEARNED_MODELS = ("bc_mdn_aif", "cvae_aif", "transformer_aif", "diffusion_policy_aif", "flow_matching_aif")
MODEL_CONFIG_KEYS = LEARNED_MODELS
ALL_MODELS = ("cem_aif",) + LEARNED_MODELS
DEFAULT_BUDGETS_MS = (50, 100, 250, 500, 1000)
DEFAULT_K_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _as_csv(value: str | None, *, cast=str) -> list:
    if value is None:
        return []
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def _config_path(experiment: str) -> Path:
    return ROOT / "configs" / "paper" / f"{experiment}.yaml"


def _load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = load_config(config_path)
    common_path = config.get("common_config")
    if not common_path:
        return config
    common_file = Path(common_path)
    if not common_file.is_absolute():
        local_candidate = config_path.parent / common_file
        common_file = local_candidate if local_candidate.exists() else ROOT / common_file
    merged = load_config(common_file)
    merged.update({key: value for key, value in config.items() if key != "common_config"})
    return merged


def _force_cpu_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    model_configs = dict(cfg.get("model_configs") or {})
    for model in MODEL_CONFIG_KEYS:
        per_model = dict(model_configs.get(model) or {})
        per_model["device"] = "cpu"
        model_configs[model] = per_model
    cfg["model_configs"] = model_configs
    return cfg


def _scene_ids_for(experiment: str, config: dict[str, Any]) -> list[int]:
    return [int(item) for item in config.get("scene_ids", [0])]


def _seeds_for(experiment: str, config: dict[str, Any]) -> list[int]:
    configured = [int(item) for item in config.get("seeds", [0])]
    if experiment in {"id_n_sweep", "heldout_start_goal"} and len(set(configured)) < 3:
        return [0, 1, 2]
    return configured


def _model_run_fields(model: str, *, calibration: bool = False) -> tuple[str, dict[str, Any]]:
    suffix = "_calibration" if calibration else "_equal_wall_clock"
    if model == "cem_aif":
        return f"aif_cem{suffix}", {"proposal_source": "aif_cem"}
    return f"generator_aif_score{suffix}", {"proposal_source": "generator"}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _bool_float(value: Any) -> float | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return _float(value)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _base_run_config(
    *,
    experiment: str,
    model: str,
    K: int,
    output_dir: Path,
    max_steps: int,
    scene_ids: list[int],
    seeds: list[int],
    calibration: bool,
) -> dict[str, Any]:
    base = _force_cpu_config(_load_benchmark_config(_config_path(experiment)))
    variant, overrides = _model_run_fields(model, calibration=calibration)
    cfg = dict(base)
    cfg.update(overrides)
    cfg.update(
        {
            "experiment": experiment,
            "variant": variant,
            "models": [model],
            "K": int(K),
            "output_dir": str(output_dir),
            "scene_ids": [int(item) for item in scene_ids],
            "seeds": [int(item) for item in seeds],
            "max_steps": int(max_steps),
            "benchmark_mode": "equal_k",
            "save_visuals": False,
            "use_generator_context": True,
            "equal_wall_clock_cpu": True,
        }
    )
    return cfg


def _run_or_read(config: dict[str, Any], *, force: bool) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    per_episode = output_dir / "per_episode.csv"
    per_step = output_dir / "per_step.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if not force and per_episode.exists() and per_step.exists():
        return {
            "per_episode_path": str(per_episode),
            "per_step_path": str(per_step),
            "failures_path": str(output_dir / "failures.json"),
            "failures": [],
            "reused": True,
        }
    result = run_generator_benchmark(config)
    result["reused"] = False
    return result


def _timing_summary(step_rows: list[dict[str, str]], budget_ms: int | None = None) -> dict[str, Any]:
    planning = [_float(row.get("planning_time")) for row in step_rows]
    proposal = [_float(row.get("proposal_time")) for row in step_rows]
    scoring = [_float(row.get("rollout_scoring_time")) for row in step_rows]
    wall = [_float(row.get("step_wall_clock_time")) for row in step_rows]
    planning_values = [value for value in planning if value is not None]
    proposal_values = [value for value in proposal if value is not None]
    scoring_values = [value for value in scoring if value is not None]
    wall_values = [value for value in wall if value is not None]
    budget_sec = None if budget_ms is None else float(budget_ms) / 1000.0
    return {
        "step_count": len(step_rows),
        "planning_p50_sec": _quantile(planning_values, 0.50),
        "planning_p90_sec": _quantile(planning_values, 0.90),
        "planning_mean_sec": _mean(planning_values),
        "proposal_p50_sec": _quantile(proposal_values, 0.50),
        "proposal_p90_sec": _quantile(proposal_values, 0.90),
        "scoring_p50_sec": _quantile(scoring_values, 0.50),
        "scoring_p90_sec": _quantile(scoring_values, 0.90),
        "wall_p50_sec": _quantile(wall_values, 0.50),
        "wall_p90_sec": _quantile(wall_values, 0.90),
        "budget_violation_rate": (
            _mean([1.0 if value > budget_sec else 0.0 for value in planning_values])
            if budget_sec is not None and planning_values
            else float("nan")
        ),
    }


def calibrate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiments = _as_csv(args.experiments, cast=str) or ["id_n_sweep", "heldout_start_goal"]
    models = _as_csv(args.models, cast=str) or list(ALL_MODELS)
    k_grid = _as_csv(args.k_grid, cast=int) or list(DEFAULT_K_GRID)
    budgets = _as_csv(args.budgets_ms, cast=int) or list(DEFAULT_BUDGETS_MS)
    root = Path(args.output_root)
    measurement_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        base = _force_cpu_config(_load_benchmark_config(_config_path(experiment)))
        scenes = _scene_ids_for(experiment, base)[: max(1, int(args.calibration_scene_count))]
        seeds = _as_csv(args.calibration_seeds, cast=int) or [_seeds_for(experiment, base)[0]]
        for model in models:
            for K in k_grid:
                output_dir = root / "calibration_runs" / experiment / model / f"K{int(K)}"
                cfg = _base_run_config(
                    experiment=experiment,
                    model=model,
                    K=int(K),
                    output_dir=output_dir,
                    max_steps=int(args.calibration_max_steps),
                    scene_ids=scenes,
                    seeds=seeds,
                    calibration=True,
                )
                print(f"[calibrate] {experiment} {model} K={K}", flush=True)
                result = _run_or_read(cfg, force=bool(args.force))
                step_rows = _read_csv(Path(result["per_step_path"]))
                timing = _timing_summary(step_rows)
                failures = _read_failures(output_dir / "failures.json")
                measurement_rows.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "K": int(K),
                        "calibration_scene_ids": ",".join(str(item) for item in scenes),
                        "calibration_seeds": ",".join(str(item) for item in seeds),
                        "calibration_max_steps": int(args.calibration_max_steps),
                        "run_dir": str(output_dir),
                        "failure_count": len(failures),
                        "reused": bool(result.get("reused", False)),
                        **timing,
                    }
                )

    selected_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in measurement_rows:
        grouped[(str(row["experiment"]), str(row["model"]))].append(row)
    for (experiment, model), rows in sorted(grouped.items()):
        valid_rows = [row for row in rows if int(row.get("failure_count", 0)) == 0 and math.isfinite(float(row.get("planning_p90_sec", float("nan"))))]
        valid_rows.sort(key=lambda row: int(row["K"]))
        for budget_ms in budgets:
            budget_sec = float(budget_ms) / 1000.0
            fitting = [row for row in valid_rows if float(row["planning_p90_sec"]) <= budget_sec]
            selected = fitting[-1] if fitting else (valid_rows[0] if valid_rows else sorted(rows, key=lambda row: int(row["K"]))[0])
            selected_rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "budget_ms": int(budget_ms),
                    "calibrated_K": int(selected["K"]),
                    "calibration_planning_p50_sec": selected["planning_p50_sec"],
                    "calibration_planning_p90_sec": selected["planning_p90_sec"],
                    "calibration_planning_mean_sec": selected["planning_mean_sec"],
                    "calibration_step_count": selected["step_count"],
                    "calibration_run_dir": selected["run_dir"],
                    "calibration_over_budget": bool(float(selected["planning_p90_sec"]) > budget_sec),
                }
            )

    _write_csv(root / "calibration_measurements.csv", measurement_rows)
    _write_csv(root / "calibration.csv", selected_rows)
    return measurement_rows, selected_rows


def _read_failures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    failures = payload.get("failures", [])
    return failures if isinstance(failures, list) else []


def _evaluate_one_task(payload: dict[str, Any]) -> dict[str, Any]:
    configure_torch_threads(int(payload["torch_threads"]))
    root = Path(payload["output_root"])
    row = dict(payload["calibration_row"])
    experiment = str(row["experiment"])
    model = str(row["model"])
    budget_ms = int(float(row["budget_ms"]))
    K = int(float(row["calibrated_K"]))
    base = _force_cpu_config(_load_benchmark_config(_config_path(experiment)))
    scenes = _scene_ids_for(experiment, base)
    seeds = _seeds_for(experiment, base)
    output_dir = root / "evaluation_runs" / experiment / f"budget_{budget_ms}ms" / f"{model}_K{K}"
    cfg = _base_run_config(
        experiment=experiment,
        model=model,
        K=K,
        output_dir=output_dir,
        max_steps=int(payload["evaluation_max_steps"]),
        scene_ids=scenes,
        seeds=seeds,
        calibration=False,
    )
    cfg["equal_wall_clock_budget_ms"] = budget_ms
    cfg["calibration_run_dir"] = row.get("calibration_run_dir", "")
    result = _run_or_read(cfg, force=bool(payload["force"]))
    failures = _read_failures(output_dir / "failures.json")
    metadata = {
        "equal_wall_clock_budget_ms": budget_ms,
        "calibrated_K": K,
        "calibration_planning_p50_sec": row.get("calibration_planning_p50_sec", ""),
        "calibration_planning_p90_sec": row.get("calibration_planning_p90_sec", ""),
        "calibration_planning_mean_sec": row.get("calibration_planning_mean_sec", ""),
        "calibration_over_budget": row.get("calibration_over_budget", ""),
        "equal_wall_clock_run_dir": str(output_dir),
        "equal_wall_clock_device": "cpu",
    }
    episode_rows = _read_csv(Path(result["per_episode_path"]))
    step_rows = _read_csv(Path(result["per_step_path"]))
    annotated_episode_rows: list[dict[str, Any]] = []
    annotated_step_rows: list[dict[str, Any]] = []
    for episode in episode_rows:
        payload_row = dict(episode)
        payload_row.update(metadata)
        annotated_episode_rows.append(payload_row)
    for step in step_rows:
        payload_row = dict(step)
        payload_row.update(metadata)
        annotated_step_rows.append(payload_row)
    return {
        "sort_key": (experiment, model, budget_ms),
        "run_row": {
            "experiment": experiment,
            "model": model,
            "budget_ms": budget_ms,
            "calibrated_K": K,
            "episode_rows": len(episode_rows),
            "step_rows": len(step_rows),
            "failure_count": len(failures),
            "run_dir": str(output_dir),
            "reused": bool(result.get("reused", False)),
        },
        "episode_rows": annotated_episode_rows,
        "step_rows": annotated_step_rows,
    }


def evaluate(args: argparse.Namespace, calibration_rows: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(args.output_root)
    if calibration_rows is None:
        calibration_rows = [dict(row) for row in _read_csv(root / "calibration.csv")]
    if not calibration_rows:
        raise FileNotFoundError(f"No calibration rows found at {root / 'calibration.csv'}")

    payloads = [
        {
            "calibration_row": dict(row),
            "output_root": str(root),
            "evaluation_max_steps": int(args.evaluation_max_steps),
            "force": bool(args.force),
            "torch_threads": int(args.torch_threads),
        }
        for row in calibration_rows
    ]
    results: list[dict[str, Any]] = []
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for payload in payloads:
            row = payload["calibration_row"]
            print(
                f"[evaluate] {row['experiment']} budget={int(float(row['budget_ms']))}ms "
                f"{row['model']} K={int(float(row['calibrated_K']))}",
                flush=True,
            )
            results.append(_evaluate_one_task(payload))
    else:
        print(f"[evaluate] running {len(payloads)} locked evaluations with jobs={jobs}", flush=True)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
            futures = {executor.submit(_evaluate_one_task, payload): payload for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                run = result["run_row"]
                status = "reused" if run["reused"] else "done"
                print(
                    f"[evaluate:{status}] {run['experiment']} budget={run['budget_ms']}ms "
                    f"{run['model']} K={run['calibrated_K']} failures={run['failure_count']}",
                    flush=True,
                )
                results.append(result)

    results.sort(key=lambda item: item["sort_key"])
    all_episode_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for result in results:
        run_rows.append(result["run_row"])
        all_episode_rows.extend(result["episode_rows"])
        all_step_rows.extend(result["step_rows"])

    _write_csv(root / "per_episode.csv", all_episode_rows)
    _write_csv(root / "per_step.csv", all_step_rows)
    _write_csv(root / "evaluation_runs.csv", run_rows)
    summary_rows = summarize(all_episode_rows, all_step_rows)
    _write_csv(root / "per_budget_summary.csv", summary_rows)
    write_report(root, calibration_rows, run_rows, summary_rows)
    return all_episode_rows, all_step_rows, summary_rows


def summarize(episode_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_episodes: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_steps: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        key = (
            str(row.get("experiment", "")),
            str(row.get("variant", "")),
            str(row.get("model", "")),
            str(row.get("equal_wall_clock_budget_ms", "")),
            str(row.get("calibrated_K", row.get("K", ""))),
        )
        grouped_episodes[key].append(row)
    for row in step_rows:
        key = (
            str(row.get("experiment", "")),
            str(row.get("variant", "")),
            str(row.get("model", "")),
            str(row.get("equal_wall_clock_budget_ms", "")),
            str(row.get("calibrated_K", row.get("K", ""))),
        )
        grouped_steps[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, episodes in sorted(grouped_episodes.items()):
        experiment, variant, model, budget_ms, K = key
        steps = grouped_steps.get(key, [])
        timing = _timing_summary(steps, int(float(budget_ms)) if budget_ms else None)
        success = [_bool_float(row.get("success")) for row in episodes]
        collision = [_bool_float(row.get("collision")) for row in episodes]
        fall_out = [_bool_float(row.get("fall_out")) for row in episodes]
        timeout = [_bool_float(row.get("timeout")) for row in episodes]
        final_distance = [_float(row.get("final_distance_to_goal")) for row in episodes]
        episode_length = [_float(row.get("episode_length")) for row in episodes]
        summary_rows.append(
            {
                "experiment": experiment,
                "variant": variant,
                "model": model,
                "budget_ms": int(float(budget_ms)) if budget_ms else "",
                "calibrated_K": int(float(K)) if K else "",
                "episodes": len(episodes),
                "steps": len(steps),
                "success_rate": _mean([value for value in success if value is not None]),
                "collision_rate": _mean([value for value in collision if value is not None]),
                "fall_out_rate": _mean([value for value in fall_out if value is not None]),
                "timeout_rate": _mean([value for value in timeout if value is not None]),
                "final_distance_mean": _mean([value for value in final_distance if value is not None]),
                "episode_length_mean": _mean([value for value in episode_length if value is not None]),
                **timing,
            }
        )
    return summary_rows


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    numeric = _float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.4f}"


def write_report(root: Path, calibration_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CPU Equal Wall-Clock Experiment Record",
        "",
        "This experiment uses CPU-only fixed-K evaluations selected by a calibration pass. It does not use the existing `benchmark_mode=equal_time` placeholder.",
        "",
        "## Artifacts",
        "",
        _md_table(
            ["file", "contents"],
            [
                ["calibration.csv", "selected K per experiment/model/budget"],
                ["calibration_measurements.csv", "all measured calibration K values"],
                ["per_episode.csv", "combined evaluation per-episode rows"],
                ["per_step.csv", "combined evaluation per-step timing rows"],
                ["per_budget_summary.csv", "grouped descriptive summary"],
                ["evaluation_runs.csv", "one row per locked-K evaluation run"],
            ],
        ),
        "",
        "## Scope",
        "",
        "- Experiments: `id_n_sweep`, `heldout_start_goal`.",
        "- Device override: all learned model configs use `device=cpu`.",
        "- Evaluation uses locked calibrated K values; K is not adapted online.",
        "- Figures are not generated by this runner.",
        "",
        "## Calibration Selection",
        "",
        _md_table(
            ["experiment", "model", "budget_ms", "K", "cal_p50_s", "cal_p90_s", "over_budget"],
            [
                [
                    row.get("experiment", ""),
                    row.get("model", ""),
                    row.get("budget_ms", ""),
                    row.get("calibrated_K", ""),
                    _fmt(row.get("calibration_planning_p50_sec", "")),
                    _fmt(row.get("calibration_planning_p90_sec", "")),
                    row.get("calibration_over_budget", ""),
                ]
                for row in calibration_rows
            ],
        ),
        "",
        "## Evaluation Run Status",
        "",
        _md_table(
            ["experiment", "model", "budget_ms", "K", "episodes", "steps", "failures", "reused"],
            [
                [
                    row.get("experiment", ""),
                    row.get("model", ""),
                    row.get("budget_ms", ""),
                    row.get("calibrated_K", ""),
                    row.get("episode_rows", ""),
                    row.get("step_rows", ""),
                    row.get("failure_count", ""),
                    row.get("reused", ""),
                ]
                for row in run_rows
            ],
        ),
        "",
        "## Descriptive Summary",
        "",
        _md_table(
            ["experiment", "variant", "model", "budget_ms", "K", "episodes", "success", "final_dist", "p50_s", "p90_s", "viol_rate"],
            [
                [
                    row.get("experiment", ""),
                    row.get("variant", ""),
                    row.get("model", ""),
                    row.get("budget_ms", ""),
                    row.get("calibrated_K", ""),
                    row.get("episodes", ""),
                    _fmt(row.get("success_rate", "")),
                    _fmt(row.get("final_distance_mean", "")),
                    _fmt(row.get("planning_p50_sec", "")),
                    _fmt(row.get("planning_p90_sec", "")),
                    _fmt(row.get("budget_violation_rate", "")),
                ]
                for row in summary_rows
            ],
        ),
        "",
    ]
    (root / "experiment_record.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "output_root": str(root),
        "calibration_rows": len(calibration_rows),
        "evaluation_runs": len(run_rows),
        "summary_rows": len(summary_rows),
        "artifacts": {
            "calibration": str(root / "calibration.csv"),
            "calibration_measurements": str(root / "calibration_measurements.csv"),
            "per_episode": str(root / "per_episode.csv"),
            "per_step": str(root / "per_step.csv"),
            "per_budget_summary": str(root / "per_budget_summary.csv"),
            "evaluation_runs": str(root / "evaluation_runs.csv"),
            "experiment_record": str(root / "experiment_record.md"),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def configure_torch_threads(threads: int) -> None:
    try:
        import torch

        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(max(1, min(int(threads), 4)))
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "calibrate", "evaluate", "report"), default="all")
    parser.add_argument("--output-root", default="outputs/equal_wall_clock_cpu")
    parser.add_argument("--experiments", default="id_n_sweep,heldout_start_goal")
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--budgets-ms", default=",".join(str(item) for item in DEFAULT_BUDGETS_MS))
    parser.add_argument("--k-grid", default=",".join(str(item) for item in DEFAULT_K_GRID))
    parser.add_argument("--calibration-scene-count", type=int, default=2)
    parser.add_argument("--calibration-seeds", default="0")
    parser.add_argument("--calibration-max-steps", type=int, default=5)
    parser.add_argument("--evaluation-max-steps", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel evaluation workers; each worker uses --torch-threads CPU threads.")
    parser.add_argument("--force", action="store_true", help="Rerun benchmark directories even if per_episode/per_step already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_threads(int(args.torch_threads))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    calibration_rows: list[dict[str, Any]] | None = None
    if args.mode in {"all", "calibrate"}:
        _, calibration_rows = calibrate(args)
    if args.mode in {"all", "evaluate"}:
        evaluate(args, calibration_rows=calibration_rows)
    elif args.mode == "report":
        calibration = [dict(row) for row in _read_csv(root / "calibration.csv")]
        runs = [dict(row) for row in _read_csv(root / "evaluation_runs.csv")]
        summary = [dict(row) for row in _read_csv(root / "per_budget_summary.csv")]
        write_report(root, calibration, runs, summary)


if __name__ == "__main__":
    main()
