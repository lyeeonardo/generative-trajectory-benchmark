#!/usr/bin/env python3
"""Reproduce the E2 recovery-data overlay from archived measured predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aif.likelihood import wrapped_gaussian_log_likelihood
from scripts.experiment2_switching_plus15 import csv_read, csv_write
from scripts.paper_figure_style import MODEL_COLORS, PALETTE, apply_style

PAPER = ROOT / "results/paper/experiment2"
OUT = ROOT / "results/paper/experiment2_recovery"
SOURCE = OUT / "source"
FIGURE = ROOT / "results/paper/figures/figure3_belief_tracking_with_recovery"
ORDER = ("diffusion", "autoregressive", "cvae", "flow_matching", "diffusion_recovery")
LABELS = ("Diffusion", "AR", "Joint CVAE", "Flow", "Diffusion + recovery data")
STYLES = (("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"), ("P", "-"))
COLORS = {**MODEL_COLORS, "diffusion_recovery": "#bf7628"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def validate_inputs():
    contract = read_json(SOURCE / "contract.json")
    receipt = read_json(SOURCE / "inference_receipt.json")
    if receipt["status"] != "COMPLETE":
        raise ValueError("Archived inference is incomplete")
    expected = {
        SOURCE / "contract.json": receipt["contract"],
        SOURCE / "inference.npz": receipt["inference_sha256"],
        PAPER / "belief_curve.csv": contract["original_curve_csv_sha256"],
        PAPER / "trajectories/test/manifest.json": contract["manifest_sha256"],
        **{ROOT / path: digest for path, digest in contract["bank_hashes"].items()},
        **{PAPER / f"{model}_beliefs.npz": digest
           for model, digest in contract["original_belief_hashes"].items()},
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"Source fingerprint mismatch: {path}")
    rows = read_json(PAPER / "trajectories/test/manifest.json")["rows"]
    switched = np.asarray([row["switch_step"] is not None for row in rows])
    if len(rows) != 24 or switched.sum() != 12:
        raise ValueError("Expected 12 switched trials and 12 unchanged companions")
    if len({row["source_episode_id"] for row, flag in zip(rows, switched) if flag}) != 12:
        raise ValueError("Switched trials must have 12 unique source episodes")
    with np.load(SOURCE / "inference.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(data["switched"], switched)
    for index, row in enumerate(rows):
        with np.load(ROOT / row["trajectory"], allow_pickle=False) as trial:
            np.testing.assert_array_equal(data["observations"][index], trial["public_observations"][1:, :7])
            np.testing.assert_array_equal(data["true_tilt_index"][index], trial["true_tilt_index"][1:])
    recalculated = wrapped_gaussian_log_likelihood(
        data["observations"][:, :, None], data["prediction_mean"], data["prediction_variance"],
    )
    np.testing.assert_allclose(recalculated, data["log_likelihood"], rtol=3e-5, atol=2e-4)

    # Independently recompute all posterior updates in probability space.
    hazard = contract["hazard"]
    np.testing.assert_allclose(hazard, read_json(ROOT / "configs/experiment2_switching_plus15.json")["fixed_total_hazard"])
    transition = np.full((3, 3), hazard / 2)
    np.fill_diagonal(transition, 1 - hazard)
    probability = np.full((24, 3), 1 / 3)
    rebuilt = [probability.copy()]
    for observation in range(30):
        log_likelihood = data["log_likelihood"][:, observation]
        weights = (probability @ transition.T) * np.exp(log_likelihood - log_likelihood.max(1, keepdims=True))
        probability = weights / weights.sum(1, keepdims=True)
        rebuilt.append(probability.copy())
    np.testing.assert_allclose(np.stack(rebuilt, axis=1), data["posterior"], atol=3e-14, rtol=1e-12)
    return contract, rows, switched, data, expected


def render(curves):
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.linewidth": .7})
    fig, ax = plt.subplots(figsize=(170 / 25.4, 80 / 25.4))
    fig.subplots_adjust(left=.115, right=.985, bottom=.15, top=.91)
    for model, label, (marker, linestyle) in zip(ORDER, LABELS, STYLES):
        rows = [row for row in curves if row["model"] == model]
        x = np.asarray([row["observation"] for row in rows])
        recovery = model == "diffusion_recovery"
        color = COLORS[model]
        ax.fill_between(x, [r["ci95_low"] for r in rows], [r["ci95_high"] for r in rows],
                        color=color, alpha=.16 if recovery else .14, linewidth=0)
        ax.plot(x, [r["mean_p_plus15"] for r in rows], color=color, marker=marker,
                linestyle=linestyle, markevery=(1, 2) if recovery else 2,
                markersize=5 if recovery else 4, linewidth=1.9 if recovery else 1.7,
                label=label, markerfacecolor="white" if recovery else color,
                markeredgewidth=.9 if recovery else .5)
    ax.axvline(10, color=PALETTE["ink"], linestyle="--", linewidth=1)
    ax.axhline(1 / 3, color="#8f8a7d", linestyle=(0, (2, 2)), linewidth=.9)
    ax.text(5, 1.025, "0°", transform=ax.get_xaxis_transform(), ha="center", va="bottom")
    ax.text(20, 1.025, "+15°", transform=ax.get_xaxis_transform(), ha="center", va="bottom")
    ax.set(xlabel="Number of observations", ylabel="Posterior probability of +15° tilt",
           xlim=(0, 30), ylim=(0, 1.02))
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linewidth=.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.6, handlelength=2.5,
              borderaxespad=.7, labelspacing=.6)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(FIGURE.with_suffix(f".{extension}"), dpi=450, facecolor="white")
    plt.close(fig)


def main():
    contract, trials, switched, data, inputs = validate_inputs()
    curves = csv_read(PAPER / "belief_curve.csv")
    draws = np.random.default_rng(260917).integers(0, 12, (20000, 12))
    posteriors = {"diffusion_recovery": data["posterior"]}
    for model in ORDER[:-1]:
        with np.load(PAPER / f"{model}_beliefs.npz", allow_pickle=False) as saved:
            posteriors[model] = saved["posterior"]
    for model in ORDER:
        values = posteriors[model][switched, :, 2]
        if values.shape != (12, 31) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete posterior series for {model}")
        mean = values.mean(axis=0)
        bounds = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
        derived = [dict(model=model, observation=t, available_blocks=12, mean_p_plus15=float(mean[t]),
                        ci95_low=float(bounds[0, t]), ci95_high=float(bounds[1, t])) for t in range(31)]
        if model == "diffusion_recovery":
            curves.extend(derived)
        else:
            saved = [row for row in curves if row["model"] == model]
            for before, after in zip(saved, derived):
                for key in ("observation", "available_blocks", "mean_p_plus15", "ci95_low", "ci95_high"):
                    np.testing.assert_allclose(before[key], after[key], atol=2e-14, rtol=1e-12)
    archived_curves = csv_read(SOURCE / "belief_curve_with_recovery.csv")
    if len(curves) != len(archived_curves) or len(curves) != 155:
        raise ValueError("Expected 31 observations for each of five models")
    for rebuilt, archived in zip(curves, archived_curves):
        if rebuilt["model"] != archived["model"]:
            raise ValueError("Archived model order differs")
        for key in ("observation", "available_blocks", "mean_p_plus15", "ci95_low", "ci95_high"):
            np.testing.assert_allclose(rebuilt[key], archived[key], atol=2e-14, rtol=1e-12)
    recovery = data["posterior"][switched, :, 2]
    # Observation 11 is one observation after the switch, matching Table 3.
    delays = [int(crossing[0]) + 1 for trial in recovery
              if len(crossing := np.flatnonzero(trial[11:] >= .9))]
    metrics = {
        "switched_trajectories": int(switched.sum()), "unchanged_trajectories": int((~switched).sum()),
        "delay_median": float(np.median(delays)) if delays else None,
        "delay_min": min(delays) if delays else None,
        "delay_max": max(delays) if delays else None,
        "correct_final_tilt": int((data["posterior"][switched, 30].argmax(-1) == 2).sum()),
        "identified_within_six": int((recovery[:, 11:17] >= .9).any(1).sum()),
        "maintained_p90_observations16_30": int((recovery[:, 16:] >= .9).all(1).sum()),
        "mean_p_plus15_observation30": float(recovery[:, 30].mean()),
        "unchanged_false_alarms": int((data["posterior"][~switched, 10:].argmax(-1) != 1).any(1).sum()),
        "minimum_individual_late_probability": float(recovery[:, 16:].min()),
    }
    trial_rows = [dict(trial_id=trial["trial_id"], source_episode_id=trial["source_episode_id"],
                       switched=bool(switched[i]), observation=t,
                       p_minus15=float(p[0]), p_flat=float(p[1]), p_plus15=float(p[2]))
                  for i, trial in enumerate(trials) for t, p in enumerate(data["posterior"][i])]
    csv_write(OUT / "belief_curve.csv", curves)
    csv_write(OUT / "recovery_trial_beliefs.csv", trial_rows)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    table = csv_read(ROOT / "results/paper/tables/table3_e2.csv")
    table.append(dict(
        model="Diffusion + recovery data",
        delay_median=f"{metrics['delay_median']:g}" if delays else "NA",
        delay_min=metrics["delay_min"], delay_max=metrics["delay_max"],
        identified_by_six=f"{metrics['identified_within_six']}/{metrics['switched_trajectories']}",
        correct_observation_30=f"{metrics['correct_final_tilt']}/{metrics['switched_trajectories']}",
    ))
    csv_write(ROOT / "results/paper/tables/table3_e2_with_recovery.csv", table)
    render(curves)
    provenance = {
        "source_directory": "dissertation/chapter3/side_experiments/e2_recovery_retraining/figure_extension",
        "input_sha256": {str(path.relative_to(ROOT)): digest for path, digest in inputs.items()},
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "calibration_sha256": contract["calibration_sha256"],
        "original_four_curves_unchanged": True, "recovery_curve_matches_archive": True,
        "bootstrap_draws": 20000, "bootstrap_seed": 260917, "bootstrap_unit": "source trajectory",
        "likelihoods_and_filter_recomputed": True, "new_training": False, "new_model_inference": False,
        "scope": "Previously inspected E2 trajectories, one reconstructed seed and its own calibration; no control-success claim.",
        "outputs": {str(FIGURE.with_suffix(f'.{ext}').relative_to(ROOT)): sha(FIGURE.with_suffix(f'.{ext}'))
                    for ext in ("png", "pdf", "svg")},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figure": str(FIGURE.with_suffix('.png')), **metrics}, indent=2))


if __name__ == "__main__":
    main()
