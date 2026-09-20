#!/usr/bin/env python
"""Build paper-facing summary tables and figures from canonical outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_task.dataset.schema import EpisodeArchive
from evaluation.paper_results import PaperResults, load_paper_results
from evaluation.uncertainty import default_uncertainty_metadata


PAPER_ROOT = Path("outputs/paper")


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_value(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _group(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, str]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _episode_summary_from_rows(rows: list[dict[str, str]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(_group(rows, group_keys).items()):
        payload = {name: value for name, value in zip(group_keys, key)}
        payload["episodes"] = len(group_rows)
        for metric in ("success", "collision", "fall_out", "timeout"):
            payload[f"{metric}_rate"] = float(np.mean([_bool_value(row.get(metric, False)) for row in group_rows])) if group_rows else 0.0
        payload["mean_final_distance_to_goal"] = float(np.mean([_float_value(row.get("final_distance_to_goal")) for row in group_rows]))
        payload["mean_episode_length"] = float(np.mean([_float_value(row.get("episode_length")) for row in group_rows]))
        payload["mean_planning_time_per_decision"] = float(np.nanmean([_float_value(row.get("planning_time_per_decision")) for row in group_rows]))
        payload["mean_proposal_time_per_decision"] = float(np.nanmean([_float_value(row.get("proposal_time_per_decision")) for row in group_rows]))
        payload["mean_rollout_scoring_time_per_decision"] = float(np.nanmean([_float_value(row.get("rollout_scoring_time_per_decision")) for row in group_rows]))
        payload["mean_wall_clock_time_per_decision"] = float(np.nanmean([_float_value(row.get("wall_clock_time_per_decision")) for row in group_rows]))
        payload["mean_total_wall_clock_time_per_episode"] = float(np.nanmean([_float_value(row.get("total_wall_clock_time_per_episode")) for row in group_rows]))
        payload["mean_total_planning_time_per_episode"] = float(np.nanmean([_float_value(row.get("total_planning_time_per_episode")) for row in group_rows]))
        for metric in (
            "mean_feasible_proposal_rate",
            "mean_invalid_action_fraction",
            "mean_invalid_route_fraction",
            "mean_left_route_fraction",
            "mean_right_route_fraction",
            "mean_center_route_fraction",
            "mean_both_side_routes_covered",
            "mean_feasible_side_route_coverage",
            "mean_route_entropy",
            "mean_best_of_K_G_total",
        ):
            values = [_float_value(row.get(metric)) for row in group_rows]
            values = [value for value in values if np.isfinite(value)]
            if values:
                payload[metric] = float(np.mean(values))
        out.append(payload)
    return out


def _episode_summary_rows(input_path: Path, group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Compatibility wrapper for non-paper smoke and staged run roots."""

    return _episode_summary_from_rows(_read_csv(input_path), group_keys)


def _step_rows_from_canonical_sources(results: PaperResults, experiment_group: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_paths = sorted(
        {
            Path(row["paper_source_path"]).with_name("per_step.csv")
            for row in results.experiment_rows(experiment_group)
        }
    )
    for source_path in source_paths:
        rows.extend(_read_csv(source_path))
    return rows


def build_dataset_summary(dataset_root: Path, table_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        root = dataset_root / split
        if not (root / "episodes.npz").exists():
            continue
        archive = EpisodeArchive.load(root, split)
        tilts = sorted(set(zip(np.round(np.degrees(archive.lateral_tilt), 6), np.round(np.degrees(archive.longitudinal_tilt), 6))))
        for lateral_deg, longitudinal_deg in tilts:
            mask = (np.isclose(np.degrees(archive.lateral_tilt), lateral_deg)) & (np.isclose(np.degrees(archive.longitudinal_tilt), longitudinal_deg))
            rows.append(
                {
                    "split": split,
                    "lateral_tilt_deg": float(lateral_deg),
                    "longitudinal_tilt_deg": float(longitudinal_deg),
                    "episodes": int(np.sum(mask)),
                    "successes": int(np.sum(np.asarray(archive.success, dtype=bool)[mask])),
                    "steps": int(np.sum(np.asarray(archive.length, dtype=np.int64)[mask])),
                }
            )
    path = table_dir / "dataset_summary_by_tilt.csv"
    _write_csv(path, rows)
    return path


def _plot_line_groups(path: Path, rows: list[dict[str, Any]], *, x_key: str, y_key: str, label_key: str, title: str, ylabel: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = sorted(set(str(row[label_key]) for row in rows))
    for label in labels:
        selected = [row for row in rows if str(row[label_key]) == label]
        xs = sorted(set(float(row[x_key]) for row in selected))
        ys = []
        for x in xs:
            vals = [float(row[y_key]) for row in selected if float(row[x_key]) == x and np.isfinite(float(row[y_key]))]
            ys.append(float(np.mean(vals)) if vals else float("nan"))
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_scatter_groups(path: Path, rows: list[dict[str, Any]], *, x_key: str, y_key: str, label_key: str, title: str, xlabel: str, ylabel: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = sorted(set(str(row[label_key]) for row in rows))
    for label in labels:
        selected = [row for row in rows if str(row[label_key]) == label]
        xs = [_float_value(row.get(x_key)) for row in selected]
        ys = [_float_value(row.get(y_key)) for row in selected]
        keep = [index for index, (x, y) in enumerate(zip(xs, ys)) if np.isfinite(x) and np.isfinite(y)]
        if keep:
            ax.scatter([xs[index] for index in keep], [ys[index] for index in keep], s=36, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_multi_metric_line_groups(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_keys: tuple[str, ...],
    label_key: str,
    title: str,
    ylabel: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    labels = sorted(set(str(row[label_key]) for row in rows))
    for label in labels:
        selected = [row for row in rows if str(row[label_key]) == label]
        xs = sorted(set(_float_value(row.get(x_key)) for row in selected if np.isfinite(_float_value(row.get(x_key)))))
        for y_key in y_keys:
            ys = []
            for x in xs:
                vals = [_float_value(row.get(y_key)) for row in selected if _float_value(row.get(x_key)) == x]
                vals = [value for value in vals if np.isfinite(value)]
                ys.append(float(np.mean(vals)) if vals else float("nan"))
            if any(np.isfinite(ys)):
                ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"{label}:{y_key.replace('mean_', '').replace('_time_per_decision', '')}")
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_time_trace(path: Path, rows: list[dict[str, str]], *, y_keys: tuple[str, ...], title: str, ylabel: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    variants = sorted(set(row.get("variant", "") for row in rows))
    for variant in variants:
        selected = [row for row in rows if row.get("variant", "") == variant]
        steps = sorted(set(int(float(row.get("step", 0))) for row in selected))
        for y_key in y_keys:
            values = []
            for step in steps:
                vals = [_float_value(row.get(y_key)) for row in selected if int(float(row.get("step", 0))) == step]
                vals = [value for value in vals if np.isfinite(value)]
                values.append(float(np.mean(vals)) if vals else float("nan"))
            if any(np.isfinite(values)):
                label = variant if len(y_keys) == 1 else f"{variant}:{y_key}"
                ax.plot(steps, values, marker="o", linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_paper_figures(
    paper_root: str | Path = PAPER_ROOT,
    dataset_root: str | Path = "datasets/generator_training",
    raw_results: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(paper_root)
    figure_dir = root / "figures"
    table_dir = figure_dir / "tables"
    outputs: dict[str, Any] = {"figures": [], "tables": []}

    raw_path = Path(raw_results) if raw_results is not None else root / "paper_raw_results.csv"
    canonical_results = load_paper_results(raw_path, require_existing_source_paths=True) if raw_path.is_file() else None
    if raw_results is not None and canonical_results is None:
        raise FileNotFoundError(f"Canonical paper results not found: {raw_path}")
    if canonical_results is not None:
        outputs["canonical_results"] = {
            "path": str(canonical_results.path),
            "sha256": canonical_results.sha256,
            "source_counts": canonical_results.source_counts,
        }
        outputs["statistical_contract"] = "docs/statistical_analysis_contract.md"
        outputs["uncertainty"] = default_uncertainty_metadata()

    dataset_table = build_dataset_summary(Path(dataset_root), table_dir)
    outputs["tables"].append(str(dataset_table))

    experiment_group_keys = {
        "id_n_sweep": ("variant", "K", "model"),
        "heldout_start_goal": ("variant", "K", "model"),
    }
    for experiment, keys in experiment_group_keys.items():
        table = table_dir / f"{experiment}_summary.csv"
        if canonical_results is not None:
            episode_rows = canonical_results.experiment_rows(experiment)
            rows = _episode_summary_from_rows(episode_rows, keys)
        else:
            rows = _episode_summary_rows(root / experiment / "metrics" / "per_episode.csv", keys)
        _write_csv(table, rows)
        outputs["tables"].append(str(table))
        if experiment == "id_n_sweep" and rows:
            outputs["figures"].append(
                str(
                    _plot_line_groups(
                        figure_dir / "success_vs_candidate_budget.png",
                        rows,
                        x_key="K",
                        y_key="success_rate",
                        label_key="variant",
                        title="ID success vs candidate budget",
                        ylabel="success rate",
                    )
                )
            )
            outputs["figures"].append(
                str(
                    _plot_line_groups(
                        figure_dir / "final_distance_vs_candidate_budget.png",
                        rows,
                        x_key="K",
                        y_key="mean_final_distance_to_goal",
                        label_key="variant",
                        title="ID final distance vs candidate budget",
                        ylabel="mean final distance",
                    )
                )
            )
            if any(np.isfinite(_float_value(row.get("mean_wall_clock_time_per_decision"))) for row in rows):
                outputs["figures"].append(
                    str(
                        _plot_scatter_groups(
                            figure_dir / "success_vs_wall_clock_time.png",
                            rows,
                            x_key="mean_wall_clock_time_per_decision",
                            y_key="success_rate",
                            label_key="variant",
                            title="ID success vs wall-clock decision time",
                            xlabel="mean wall-clock time per decision (s)",
                            ylabel="success rate",
                        )
                    )
                )
            if any("mean_feasible_proposal_rate" in row for row in rows):
                outputs["figures"].append(
                    str(
                        _plot_line_groups(
                            figure_dir / "feasible_support_vs_candidate_budget.png",
                            rows,
                            x_key="K",
                            y_key="mean_feasible_proposal_rate",
                            label_key="variant",
                            title="ID feasible proposal support vs candidate budget",
                            ylabel="mean feasible proposal rate",
                        )
                    )
                )
            if any("mean_both_side_routes_covered" in row for row in rows):
                outputs["figures"].append(
                    str(
                        _plot_line_groups(
                            figure_dir / "route_coverage_vs_candidate_budget.png",
                            rows,
                            x_key="K",
                            y_key="mean_both_side_routes_covered",
                            label_key="variant",
                            title="ID route coverage vs candidate budget",
                            ylabel="mean both-side route coverage",
                        )
                    )
                )
            outputs["figures"].append(
                str(
                    _plot_multi_metric_line_groups(
                        figure_dir / "latency_breakdown_vs_candidate_budget.png",
                        rows,
                        x_key="K",
                        y_keys=(
                            "mean_proposal_time_per_decision",
                            "mean_rollout_scoring_time_per_decision",
                            "mean_wall_clock_time_per_decision",
                        ),
                        label_key="variant",
                        title="ID latency breakdown vs candidate budget",
                        ylabel="seconds per decision",
                    )
                )
            )

    id_steps = (
        _step_rows_from_canonical_sources(canonical_results, "id_n_sweep")
        if canonical_results is not None
        else _read_csv(root / "id_n_sweep" / "metrics" / "per_step.csv")
    )
    energy_rows = []
    for key, group_rows in _group(id_steps, ("variant", "K")).items():
        vals = [_float_value(row.get("metric_best_of_K_G_total")) for row in group_rows]
        vals = [value for value in vals if np.isfinite(value)]
        if vals:
            energy_rows.append({"variant": key[0], "K": key[1], "mean_best_of_K_G_total": float(np.mean(vals))})
    if energy_rows:
        table = table_dir / "id_n_sweep_energy_summary.csv"
        _write_csv(table, energy_rows)
        outputs["tables"].append(str(table))
        outputs["figures"].append(
            str(
                _plot_line_groups(
                    figure_dir / "best_of_n_energy_vs_candidate_budget.png",
                    energy_rows,
                    x_key="K",
                    y_key="mean_best_of_K_G_total",
                    label_key="variant",
                    title="Best-of-N energy vs candidate budget",
                    ylabel="mean best-of-K G_total",
                )
            )
        )

    manifest = figure_dir / "figure_manifest.json"
    import json

    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    outputs["manifest"] = str(manifest)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper summary tables and figures.")
    parser.add_argument("--paper_root", default=str(PAPER_ROOT))
    parser.add_argument("--dataset_root", default="datasets/generator_training")
    parser.add_argument("--raw_results", default=None)
    args = parser.parse_args()
    import json

    print(json.dumps(build_paper_figures(args.paper_root, args.dataset_root, args.raw_results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
