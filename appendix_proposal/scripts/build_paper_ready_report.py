#!/usr/bin/env python
"""Build a paper-writing report and acceptance audit for paper-ready runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENTS = ("id_n_sweep", "heldout_start_goal")
REQUIRED_EPISODE_COLUMNS = (
    "success",
    "collision",
    "fall_out",
    "timeout",
    "final_distance_to_goal",
    "planning_time_per_decision",
    "proposal_time_per_decision",
    "rollout_scoring_time_per_decision",
    "wall_clock_time_per_decision",
    "total_wall_clock_time_per_episode",
    "mean_feasible_proposal_rate",
    "mean_both_side_routes_covered",
)
REQUIRED_STEP_COLUMNS = (
    "metric_best_of_K_G_total",
    "metric_feasible_proposal_rate",
    "metric_left_route_fraction",
    "metric_right_route_fraction",
    "metric_invalid_route_fraction",
    "metric_both_side_routes_covered",
    "step_wall_clock_time",
    "selection_mode",
    "aif_score_used_for_selection",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _rate(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([_bool(row.get(key, False)) for row in rows]))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [_float(row.get(key)) for row in rows]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else float("nan")


def _fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _experiment_audit(root: Path, experiment: str) -> dict[str, Any]:
    metrics = root / experiment / "metrics"
    runs_dir = root / experiment / "runs"
    episodes = _read_csv(metrics / "per_episode.csv")
    steps = _read_csv(metrics / "per_step.csv")
    failures = _read_json(metrics / "failures.json").get("failures", [])
    episode_cols = set(episodes[0]) if episodes else set()
    step_cols = set(steps[0]) if steps else set()
    return {
        "experiment": experiment,
        "episodes": len(episodes),
        "steps": len(steps),
        "failures": len(failures),
        "completed_run_dirs": len(list(runs_dir.glob("*/summary_report.md"))) if runs_dir.exists() else 0,
        "seeds": sorted({int(float(row["seed"])) for row in episodes if row.get("seed", "") != ""}),
        "variants": sorted({row.get("variant", "") for row in episodes}),
        "budgets": sorted({int(float(row["K"])) for row in episodes if row.get("K", "") != ""}),
        "models": sorted({row.get("model", "") for row in episodes}),
        "missing_episode_columns": sorted(set(REQUIRED_EPISODE_COLUMNS) - episode_cols),
        "missing_step_columns": sorted(set(REQUIRED_STEP_COLUMNS) - step_cols),
        "success_rate": _rate(episodes, "success"),
        "collision_rate": _rate(episodes, "collision"),
        "fall_out_rate": _rate(episodes, "fall_out"),
        "timeout_rate": _rate(episodes, "timeout"),
        "mean_final_distance": _mean(episodes, "final_distance_to_goal"),
        "mean_wall_clock_time_per_decision": _mean(episodes, "wall_clock_time_per_decision"),
    }


def _acceptance(audits: dict[str, dict[str, Any]]) -> list[tuple[str, bool, str]]:
    id_audit = audits.get("id_n_sweep", {})
    heldout = audits.get("heldout_start_goal", {})
    all_failures = sum(int(audit.get("failures", 0)) for audit in audits.values())
    id_budgets = set(id_audit.get("budgets", []))
    id_variants = set(id_audit.get("variants", []))
    heldout_variants = set(heldout.get("variants", []))
    checks = [
        ("No failed benchmark cells", all_failures == 0, f"failure_count={all_failures}"),
        ("ID has at least 3 seeds", len(set(id_audit.get("seeds", []))) >= 3, f"seeds={id_audit.get('seeds', [])}"),
        ("Heldout has at least 3 seeds", len(set(heldout.get("seeds", []))) >= 3, f"seeds={heldout.get('seeds', [])}"),
        ("ID includes large-K CEM reference", {512, 1024}.issubset(id_budgets) and "aif_cem_reference" in id_variants, f"budgets={sorted(id_budgets)} variants={sorted(id_variants)}"),
        ("Pure generator-only included", "pure_generator_only" in id_variants and "pure_generator_only" in heldout_variants, f"id={sorted(id_variants)} heldout={sorted(heldout_variants)}"),
        (
            "Latency and support columns present",
            all(not audit.get("missing_episode_columns") and not audit.get("missing_step_columns") for audit in audits.values() if audit.get("episodes", 0)),
            "missing columns reported per experiment below",
        ),
    ]
    return checks


def build_paper_ready_report(paper_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(paper_root)
    output = Path(output_path) if output_path is not None else root / "paper_ready_report.md"
    manifest = _read_json(root / "paper_step5_manifest.json")
    figure_manifest = _read_json(root / "figures" / "figure_manifest.json")
    audits = {experiment: _experiment_audit(root, experiment) for experiment in EXPERIMENTS if (root / experiment / "metrics").exists()}
    checks = _acceptance(audits)

    lines = [
        "# Paper-Ready Experiment Report",
        "",
        "## Claim Scope",
        "",
        "Use these artifacts for a workshop-style benchmark claim: learned generative proposals improve finite-candidate AIF planning efficiency under a shared MuJoCo rollout scorer.",
        "",
        "Do not claim definitive scaling proof unless the large-K CEM and multi-seed results support it.",
        "",
        "## Acceptance Audit",
        "",
        "| requirement | status | evidence |",
        "| --- | --- | --- |",
    ]
    for label, ok, evidence in checks:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} | `{evidence}` |")

    lines.extend(["", "## Experiment Summary", "", "| experiment | episodes | seeds | variants | success | fall_out | wall-clock/decision | failures |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for experiment in EXPERIMENTS:
        audit = audits.get(experiment)
        if not audit:
            lines.append(f"| {experiment} | 0 | [] | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| {experiment} | {episodes} | `{seeds}` | {variants} | {success} | {fall_out} | {wall} | {failures} |".format(
                experiment=experiment,
                episodes=int(audit["episodes"]),
                seeds=audit["seeds"],
                variants=len(audit["variants"]),
                success=_fmt(float(audit["success_rate"])),
                fall_out=_fmt(float(audit["fall_out_rate"])),
                wall=_fmt(float(audit["mean_wall_clock_time_per_decision"])),
                failures=int(audit["failures"]),
            )
        )

    lines.extend(["", "## Column Audit", ""])
    for experiment, audit in audits.items():
        lines.append(f"- `{experiment}` missing episode columns: `{audit['missing_episode_columns']}`")
        lines.append(f"- `{experiment}` missing step columns: `{audit['missing_step_columns']}`")

    lines.extend(["", "## Figures", ""])
    for path in figure_manifest.get("figures", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Tables", ""])
    for path in figure_manifest.get("tables", []):
        lines.append(f"- `{path}`")

    total_episodes = sum(int(audit.get("episodes", 0)) for audit in audits.values())
    total_steps = sum(int(audit.get("steps", 0)) for audit in audits.values())
    total_run_dirs = sum(int(audit.get("completed_run_dirs", 0)) for audit in audits.values())
    lines.extend(
        [
            "",
            "## Output Inventory",
            "",
            f"- completed run directories: `{total_run_dirs}`",
            f"- audited episodes: `{total_episodes}`",
            f"- audited planner steps: `{total_steps}`",
            f"- latest runner manifest: `{root / 'paper_step5_manifest.json'}`",
            f"- latest invocation runs requested: `{manifest.get('runs_requested', 'n/a')}`",
            "",
            "The acceptance audit above scans the complete output root; the latest runner manifest may describe only the most recent staged invocation.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"report_path": str(output), "audits": audits, "acceptance": [{"requirement": label, "passed": ok, "evidence": evidence} for label, ok, evidence in checks]}
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["json_path"] = str(json_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-ready report and acceptance audit.")
    parser.add_argument("--paper_root", default="outputs/paper_ready/main")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(json.dumps(build_paper_ready_report(args.paper_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
