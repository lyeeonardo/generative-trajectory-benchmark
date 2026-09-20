#!/usr/bin/env python3
"""Build final Appendix Figure A1 from retained archived summary CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_figure_style import apply_style

SOURCE = ROOT / "appendix_proposal" / "results"
OUT = ROOT / "results" / "paper" / "figures"
DERIVED = ROOT / "results" / "paper" / "source" / "appendix_figure_a1"

LEARNED = (
    "bc_mdn_aif", "cvae_aif", "diffusion_policy_aif",
    "flow_matching_aif", "transformer_aif",
)
COLORS = {
    "CEM": "#3f4349",
    "Proposal model": "#138a80",
    "Gated proposal model": "#d18419",
    "Large-K CEM": "#777777",
    "CVAE": "#3d65a5",
}
MARKERS = {
    "CEM": "o", "Proposal model": "s", "Gated proposal model": "^",
    "Large-K CEM": "D", "CVAE": "P",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def budget_curves(rows):
    curves = {}
    definitions = (
        ("CEM", "aif_cem", ("cem_aif",)),
        ("Proposal model", "generator_aif_score", LEARNED),
        ("Gated proposal model", "gated_hybrid", LEARNED),
        ("Large-K CEM", "aif_cem_reference", ("cem_aif",)),
    )
    for label, variant, models in definitions:
        points = []
        budgets = sorted({int(row["K"]) for row in rows if row["variant"] == variant})
        for budget in budgets:
            selected = [row for row in rows if row["variant"] == variant
                        and int(row["K"]) == budget and row["model"] in models]
            if not selected:
                continue
            points.append(dict(series=label, K=budget,
                               success_percent=100 * mean(selected, "success_rate"),
                               planning_seconds=mean(selected, "mean_planning_time_per_decision")))
        curves[label] = points
    return curves


def cpu_curves(rows, experiment):
    curves = {}
    for label, models in (
        ("CEM", ("cem_aif",)),
        ("Proposal model", LEARNED),
        ("CVAE", ("cvae_aif",)),
    ):
        points = []
        for budget in sorted({int(row["budget_ms"]) for row in rows}):
            selected = [row for row in rows if row["experiment"] == experiment
                        and int(row["budget_ms"]) == budget and row["model"] in models]
            points.append(dict(series=label, budget_ms=budget,
                               success_percent=100 * mean(selected, "success_rate")))
        curves[label] = points
    return curves


def plot_curve(ax, points, x_key, label):
    ax.plot([point[x_key] for point in points],
            [point["success_percent"] for point in points],
            color=COLORS[label], marker=MARKERS[label], linewidth=1.7,
            markersize=4.5, label=label)


def main() -> None:
    equal_k = read_csv(SOURCE / "id_n_sweep_summary.csv")
    cpu = read_csv(SOURCE / "per_budget_summary.csv")
    budget = budget_curves(equal_k)
    nominal = cpu_curves(cpu, "id_n_sweep")
    transfer = cpu_curves(cpu, "heldout_start_goal")

    apply_style()
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 105 / 25.4))
    fig.subplots_adjust(left=.095, right=.985, bottom=.20, top=.93, wspace=.32, hspace=.55)

    for label in ("CEM", "Proposal model", "Gated proposal model", "Large-K CEM"):
        plot_curve(axes[0, 0], budget[label], "K", label)
        plot_curve(axes[0, 1], budget[label], "planning_seconds", label)
    for label in ("CEM", "Proposal model", "CVAE"):
        plot_curve(axes[1, 0], nominal[label], "budget_ms", label)
        plot_curve(axes[1, 1], transfer[label], "budget_ms", label)

    titles = (
        "A  Nominal success vs K", "B  Success vs planning time",
        "C  CPU targets: nominal", "D  CPU targets: transfer",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks([8, 32, 128, 512, 1024], labels=["8", "32", "128", "512", "1024"])
    axes[0, 0].set_xlabel("Candidates K (log scale)")
    axes[0, 0].set_ylabel("Success (%)")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Time/decision (s, log scale)")
    axes[1, 0].set_xlabel("CPU target (ms)")
    axes[1, 0].set_ylabel("Success (%)")
    axes[1, 1].set_xlabel("CPU target (ms)")
    for ax in axes[1]:
        ax.set_xticks([50, 250, 500, 1000])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    cvae_handle, _ = axes[1, 0].get_legend_handles_labels()
    handles.append(cvae_handle[-1])
    labels.append("CVAE")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .035), fontsize=7.3, handlelength=2.0)

    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_a1_proposal_planning.{extension}",
                    dpi=450, facecolor="white")
    plt.close(fig)

    DERIVED.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel, curves, x_key in (
        ("A", budget, "K"), ("B", budget, "planning_seconds"),
        ("C", nominal, "budget_ms"), ("D", transfer, "budget_ms"),
    ):
        for label, points in curves.items():
            for point in points:
                rows.append(dict(panel=panel, series=label, x=point[x_key],
                                 success_percent=point["success_percent"]))
    with (DERIVED / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("panel", "series", "x", "success_percent"))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "figure_a1_proposal_planning.pdf")


if __name__ == "__main__":
    main()
