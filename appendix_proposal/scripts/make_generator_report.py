#!/usr/bin/env python
"""Build a compact markdown report from Generator benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "episodes": 0, "successes": 0, "failures": 0}
    with np.load(path) as data:
        required = ("trajectory_xy", "trajectory_mask", "success", "collision", "timeout", "final_distance_to_goal")
        missing = [name for name in required if name not in data.files]
        if missing:
            return {"present": True, "invalid": True, "missing": missing, "episodes": 0, "successes": 0, "failures": 0}
        success = np.asarray(data["success"], dtype=bool)
        collision = np.asarray(data["collision"], dtype=bool)
        fall_out = np.asarray(data["fall_out"], dtype=bool) if "fall_out" in data.files else np.zeros_like(success, dtype=bool)
        timeout = np.asarray(data["timeout"], dtype=bool)
        return {
            "present": True,
            "episodes": int(success.shape[0]),
            "successes": int(np.sum(success)),
            "failures": int(np.sum((~success) & (collision | fall_out | timeout))),
            "collisions": int(np.sum(collision)),
            "fall_outs": int(np.sum(fall_out)),
            "timeouts": int(np.sum(timeout)),
            "mean_final_distance_to_goal": float(np.mean(np.asarray(data["final_distance_to_goal"], dtype=np.float32))) if success.shape[0] else None,
        }


def _video_lines(manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return ["- video manifest: missing"]
    lines: list[str] = []
    for case in manifest.get("cases", []):
        status = case.get("status", "unknown")
        if status == "generated":
            lines.append(f"- `{case.get('case')}`: generated from `{case.get('run_id')}`")
        else:
            lines.append(f"- `{case.get('case')}`: skipped ({case.get('reason', 'no reason recorded')})")
    return lines or ["- no video cases recorded"]


def make_generator_report(input_dir: str | Path, output_path: str | Path) -> dict[str, str]:
    root = Path(input_dir)
    aggregate = _load_json(root / "aggregate_metrics.json", {"models": {}})
    failures = _load_json(root / "failures.json", {"failures": []}).get("failures", [])
    manifest = _load_json(root / "videos" / "video_manifest.json", None)
    if manifest is None:
        manifest = _load_json(root.parent / "videos" / "video_manifest.json", None)
    if manifest is None:
        manifest = _load_json(root.parent.parent / "videos" / "video_manifest.json", None)
    trajectory = _trajectory_summary(root / "episode_trajectories.npz")

    lines = ["# Generator Benchmark Report", "", f"Input: `{root}`", "", "## Models", ""]
    for model, payload in aggregate.get("models", {}).items():
        summary = payload.get("summary", {})
        lines.append(
            f"- `{model}`: episodes={payload.get('episodes', 0)}, failures={payload.get('failures', 0)}, "
            f"success_rate={summary.get('success_rate', 'n/a')}, final_distance={summary.get('final_distance_to_goal', 'n/a')}"
        )

    lines.extend(["", "## Executed Episodes", ""])
    if trajectory.get("invalid"):
        lines.append(f"- `episode_trajectories.npz`: invalid, missing {trajectory.get('missing')}")
    elif trajectory.get("present"):
        lines.append(
            f"- `episode_trajectories.npz`: episodes={trajectory['episodes']}, successes={trajectory['successes']}, "
            f"failures={trajectory['failures']}, collisions={trajectory.get('collisions', 0)}, "
            f"fall_outs={trajectory.get('fall_outs', 0)}, timeouts={trajectory.get('timeouts', 0)}, "
            f"mean_final_distance={trajectory.get('mean_final_distance_to_goal')}"
        )
    else:
        lines.append("- `episode_trajectories.npz`: missing")

    lines.extend(["", "## Videos", ""])
    lines.extend(_video_lines(manifest))

    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(f"- `{failure.get('model', 'unknown')}` {failure.get('stage', '')}: {failure.get('error', '')}")
    else:
        lines.append("No benchmark failures logged.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `per_step.csv`",
            "- `per_episode.csv`",
            "- `episode_trajectories.npz`",
            "- `aggregate_metrics.json`",
            "- `config_hashes.json`",
            "- `model_card.json`",
            "- `plots/`",
            "- `video_manifest.json` when videos were rendered",
        ]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/evals/generator_specialist")
    parser.add_argument("--output", default="outputs/evals/generator_specialist/generator_report.md")
    args = parser.parse_args()
    print(json.dumps(make_generator_report(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
