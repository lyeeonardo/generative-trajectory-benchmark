#!/usr/bin/env python3
"""Run or replot the final E2 replayed 0° to +15° belief-tracking experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_figure_style import apply_style, PALETTE, MODEL_COLORS

CONFIG = ROOT / "configs/experiment2_switching_plus15.json"
DEFAULT_OUT = ROOT / "outputs/e2_switching_plus15"
OUT = DEFAULT_OUT
MODEL_LABELS = {"diffusion": "Diffusion/DiT", "autoregressive": "Autoregressive Transformer",
                "cvae": "Joint CVAE", "flow_matching": "Flow Matching"}
STYLES = {
    "diffusion": ("o", "-"),
    "autoregressive": ("s", "--"),
    "cvae": ("^", "-."),
    "flow_matching": ("D", ":"),
}


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("observation", "available_blocks", "mean_p_plus15", "ci95_low", "ci95_high"):
            if key in row:
                row[key] = float(row[key])
    return rows


def render_figure(curves):
    config = json.loads(CONFIG.read_text())
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.linewidth": .7})
    fig, ax = plt.subplots(figsize=(170 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=.115, right=.985, bottom=.18, top=.975)
    for model in config["models"]:
        rows = [row for row in curves if row["model"] == model]
        x = np.asarray([row["observation"] for row in rows])
        color = MODEL_COLORS[model]
        marker, style = STYLES[model]
        ax.fill_between(
            x,
            [row["ci95_low"] for row in rows],
            [row["ci95_high"] for row in rows],
            color=color,
            alpha=.14,
            linewidth=0,
        )
        ax.plot(
            x,
            [row["mean_p_plus15"] for row in rows],
            color=color,
            marker=marker,
            linestyle=style,
            markevery=2,
            markersize=4,
            linewidth=1.7,
            label=MODEL_LABELS[model],
        )
    ax.axvline(10, color=PALETTE["ink"], linestyle="--", linewidth=1)
    ax.axhline(1 / 3, color="#8f8a7d", linestyle=(0, (2, 2)), linewidth=.9)
    ax.text(8, .965, "0°", transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.text(20, .965, "+15°", transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set(
        xlabel="Number of observations",
        ylabel="Posterior probability of +15° tilt",
        xlim=(0, 30),
        ylim=(0, 1.02),
    )
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linewidth=.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.3,
              handlelength=2.5, borderaxespad=.7, labelspacing=.65)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_belief_tracking.{extension}",
                    dpi=450, facecolor="white")
    if OUT.resolve() == (ROOT / "results/paper/experiment2").resolve():
        paper_out = ROOT / "results/paper/figures"
        paper_out.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(paper_out / f"figure3_belief_tracking.{extension}",
                        dpi=450, facecolor="white")
    plt.close(fig)


def plot_and_summarize(manifest, reports):
    from aif.switching_belief import filter_log_likelihoods, symmetric_transition
    from evaluation.common import atomic_npz
    config = json.loads(CONFIG.read_text())
    transition = symmetric_transition(config["fixed_total_hazard"] / 2)
    rows = manifest["rows"]
    switched = [i for i, row in enumerate(rows) if row["switch_step"] is not None]
    controls = [i for i, row in enumerate(rows) if row["switch_step"] is None]
    assert len({rows[i]["source_episode_id"] for i in switched}) == len(switched) == 12
    rng = np.random.default_rng(260917)
    draws = rng.integers(0, 12, (20000, 12))
    curves, metrics, trial_rows = [], [], []
    for model in config["models"]:
        with np.load(OUT / f"inference/test/{model}/likelihoods.npz") as saved:
            likelihood, valid = saved["log_likelihood"], saved["valid"]
        posterior = np.full((len(rows), 31, 3), np.nan)
        for i, row in enumerate(rows):
            length = row["transitions"]
            assert valid[i, :length].all() and not valid[i, length:].any()
            with np.load(ROOT / row["trajectory"]) as trial:
                truth = trial["true_tilt_index"]
                np.testing.assert_array_equal(truth[:11], 1)
                if i in switched:
                    np.testing.assert_array_equal(truth[11:], 2)
            _, belief = filter_log_likelihoods(likelihood[i, :length], transition)
            posterior[i, :length + 1] = belief
            for step, values in enumerate(belief):
                trial_rows.append(dict(
                    model=model,
                    trial_id=row["trial_id"],
                    source_episode_id=row["source_episode_id"],
                    switched=i in switched,
                    observation=step,
                    p_minus15=values[0],
                    p_flat=values[1],
                    p_plus15=values[2],
                ))
        atomic_npz(OUT / f"{model}_beliefs.npz", posterior=posterior)
        values = posterior[switched, :, 2]
        for step in range(31):
            column = values[:, step]
            n = int(np.isfinite(column).sum())
            sampled = column[draws]
            counts = np.isfinite(sampled).sum(axis=1)
            bootstrap = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
            low, high = np.quantile(bootstrap, [.025, .975])
            curves.append(dict(
                model=model,
                observation=step,
                available_blocks=n,
                mean_p_plus15=float(np.nanmean(column)),
                ci95_low=float(low),
                ci95_high=float(high),
            ))
        delays = []
        for trial in values:
            crossing = np.flatnonzero(trial[11:] >= .9)
            if len(crossing):
                delays.append(int(crossing[0]) + 1)
        metric = dict(
            model=model,
            switched_trials=12,
            no_switch_trials=12,
            delay_median=statistics.median(delays),
            delay_min=min(delays),
            delay_max=max(delays),
            identified_by6=sum(delay <= 6 for delay in delays),
            correct_observation30=int(np.sum(posterior[switched, 30].argmax(axis=1) == 2)),
            maintained_from16=int(np.sum(np.all(values[:, 16:31] >= .9, axis=1))),
            no_switch_any_false_alarm_fraction=float(np.mean([
                np.any(posterior[i, 10:rows[i]["transitions"] + 1].argmax(axis=1) != 1)
                for i in controls
            ])),
            inference_seconds=reports[model]["elapsed_seconds"],
        )
        metrics.append(metric)
    csv_write(OUT / "belief_curve.csv", curves)
    csv_write(OUT / "trial_beliefs.csv", trial_rows)
    csv_write(OUT / "metrics.csv", metrics)
    render_figure(curves)

    text = [
        "# Experiment 2: replayed hidden-tilt belief tracking",
        "",
        "The first ten executed transitions use 0° lateral tilt; the next twenty use +15°. "
        "Observation 11 is the first posterior that incorporates a switched transition. "
        "Each of twelve held-out source episodes has a matched no-switch replay, and all "
        "models receive identical recorded actions and public observations.",
        "",
        "Seed-13 weights, sampling settings, calibrated likelihoods, and the development-selected "
        "total hazard h=0.06 are frozen. There is no posterior reset, likelihood tempering, "
        "test-time refitting, action adaptation, or transition selection.",
        "",
        "| Model | Delay median [min,max] | Identified by six | Correct at observation 30 | "
        "Maintained ≥0.9 over observations 16–30 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        text.append(
            f"| {MODEL_LABELS[metric['model']]} | "
            f"{metric['delay_median']:g} [{metric['delay_min']},{metric['delay_max']}] | "
            f"{metric['identified_by6']}/12 | {metric['correct_observation30']}/12 | "
            f"{metric['maintained_from16']}/12 |"
        )
    text += [
        "",
        "Delay is the first post-switch observation at which p(+15°) reaches 0.9. "
        "Later declines occur even though +15° remains unchanged and every trajectory supplies "
        "all 30 observations. Thus rapid initial identification does not imply sustained belief tracking.",
        "",
        "Confidence intervals resample the twelve source episodes 20,000 times with seed 260917. "
        "They do not include retraining variation.",
    ]
    (OUT / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the figure from an existing belief_curve.csv without model inference.",
    )
    args = parser.parse_args()
    OUT = args.output if args.output.is_absolute() else ROOT / args.output
    if args.plot_only:
        render_figure(csv_read(OUT / "belief_curve.csv"))
        print(OUT / "figure_belief_tracking.pdf")
        return
    import evaluation.switching_dynamics as benchmark
    benchmark.CONFIG_PATH = CONFIG
    benchmark.OUTPUT = OUT
    manifest = benchmark.generate_trajectories("test")
    reports = {}
    for model in json.loads(CONFIG.read_text())["models"]:
        print(f"Running {model}", flush=True)
        reports[model] = benchmark.infer_likelihoods(model, "test", args.device)
    plot_and_summarize(manifest, reports)


if __name__ == "__main__":
    main()
