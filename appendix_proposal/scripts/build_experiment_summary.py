#!/usr/bin/env python
"""Build a combined summary for paper-facing benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_EXPERIMENTS = ("id_n_sweep", "heldout_start_goal")
DEFAULT_PAPER_ROOT = Path("outputs/paper")
DEFAULT_OUTPUT_JSON = DEFAULT_PAPER_ROOT / "paper_results.json"
DEFAULT_OUTPUT_MD = DEFAULT_PAPER_ROOT / "paper_results.md"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _empty_result(name: str, path: Path) -> dict[str, Any]:
    return {
        "result": name,
        "status": "missing",
        "source": str(path),
        "episodes": 0,
        "success_rate": None,
        "collision_rate": None,
        "fall_out_rate": None,
        "timeout_rate": None,
        "mean_final_distance_to_goal": None,
        "mean_episode_length": None,
    }


def _weighted_metric(models: dict[str, Any], key: str) -> float | None:
    total_episodes = sum(int(model.get("episodes", 0)) for model in models.values())
    if total_episodes <= 0:
        return None
    weighted = 0.0
    seen = False
    for model in models.values():
        summary = model.get("summary", {})
        if key not in summary or summary[key] is None:
            continue
        weighted += float(summary[key]) * int(model.get("episodes", 0))
        seen = True
    return float(weighted / total_episodes) if seen else None


def _aggregate_summary(name: str, path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return _empty_result(name, path)
    models = payload.get("models", {})
    episodes = sum(int(model.get("episodes", 0)) for model in models.values())
    mean_final_distance = _weighted_metric(models, "final_distance_to_goal")
    if mean_final_distance is None:
        mean_final_distance = _weighted_metric(models, "final_distance")
    return {
        "result": name,
        "status": "available",
        "source": str(path),
        "physics_backend": payload.get("physics_backend", "mujoco_rigid"),
        "experiment": payload.get("experiment", name),
        "variant": payload.get("variant"),
        "benchmark_mode": payload.get("benchmark_mode"),
        "models": len(models),
        "episodes": int(episodes),
        "success_rate": _weighted_metric(models, "success_rate"),
        "collision_rate": _weighted_metric(models, "collision_rate"),
        "fall_out_rate": _weighted_metric(models, "fall_out_rate"),
        "timeout_rate": _weighted_metric(models, "timeout_rate"),
        "mean_final_distance_to_goal": mean_final_distance,
        "mean_episode_length": _weighted_metric(models, "episode_length"),
        "episode_failure_count": int(payload.get("episode_failure_count", payload.get("failure_count", 0))),
    }


def _rate_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _number_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Combined Results",
        "",
        "This summary lists the result roots recorded in the Sources section.",
        "",
        "## Canonical Roots",
        "",
        f"- outputs: `{payload['canonical_roots']['outputs']}`",
        f"- paper: `{payload['canonical_roots']['paper']}`",
        f"- datasets: `{payload['canonical_roots']['datasets']}`",
        "",
        "## Result Metrics",
        "",
        "| result | status | episodes | success | collision | fall_out | timeout | mean final distance | mean episode length |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload["results"]:
        lines.append(
            "| {result} | {status} | {episodes} | {success} | {collision} | {fall_out} | {timeout} | {distance} | {length} |".format(
                result=result["result"],
                status=result["status"],
                episodes=int(result.get("episodes", 0)),
                success=_rate_text(result.get("success_rate")),
                collision=_rate_text(result.get("collision_rate")),
                fall_out=_rate_text(result.get("fall_out_rate")),
                timeout=_rate_text(result.get("timeout_rate")),
                distance=_number_text(result.get("mean_final_distance_to_goal")),
                length=_number_text(result.get("mean_episode_length")),
            )
        )
    lines.extend(["", "## Sources", ""])
    for result in payload["results"]:
        lines.append(f"- {result['result']}: `{result['source']}`")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paper_result_path(paper_root: Path, name: str) -> Path:
    direct = paper_root / name / "aggregate_metrics.json"
    if direct.exists():
        return direct
    return paper_root / name / "metrics" / "aggregate_metrics.json"


def build_experiment_summary(
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    output_md: str | Path = DEFAULT_OUTPUT_MD,
    *,
    paper_root: str | Path = DEFAULT_PAPER_ROOT,
) -> dict[str, Any]:
    paper_root = Path(paper_root)
    results = [_aggregate_summary(name, _paper_result_path(paper_root, name)) for name in PAPER_EXPERIMENTS]
    payload = {
        "canonical_roots": {
            "outputs": "outputs",
            "paper": str(paper_root),
            "datasets": "datasets",
        },
        "results": results,
    }
    output_json = Path(output_json)
    output_md = Path(output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(payload, output_md)
    payload["summary_path"] = str(output_json)
    payload["report_path"] = str(output_md)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined paper experiment results.")
    parser.add_argument("--paper_root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    print(
        json.dumps(
            build_experiment_summary(
                args.output_json,
                args.output_md,
                paper_root=args.paper_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
