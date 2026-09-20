#!/usr/bin/env python3
"""Build the combined two-panel E1 benchmark figure from frozen outputs, without inference."""
from pathlib import Path
import argparse
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_figure_style import apply_style, PALETTE, MODEL_COLORS
from evaluation.uncertainty import fixed_case_trial_interval

OUT = ROOT / "results/paper/figures"
DATA = ROOT / "results/paper/source/experiment1"
FITS = ROOT / "outputs/e1_full/fits"
BANK = ROOT / "datasets/evaluation"
MODELS = ["diffusion", "autoregressive", "cvae", "flow_matching"]
LABELS = ["Diffusion", "AR", "Joint CVAE", "Flow"]
DRAWS = 20000
SEED = 260916
SOURCES = {}


def read_source(path):
    path = Path(path)
    stat = path.stat()
    SOURCES[str(path.relative_to(ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return path


def read_json(path):
    return json.loads(read_source(path).read_text())


def write_csv(name, rows):
    with (DATA / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def control_results():
    summary, episodes = [], []
    expected = [(30, 81, 9), (2, 2, 0), (25, 49, 1), (15, 43, 9)]
    for model, checks in zip(MODELS, expected):
        arms = {}
        for arm, subdir in [("correct", "behavior"), ("wrong", "wrong_tilt")]:
            folder = FITS / model / "seed_13/test" / subdir / "every_step"
            paths = sorted(folder.glob("*/episode_summary.json"))
            assert len(paths) == 90
            matrix = np.full((9, 10), np.nan)
            for path in paths:
                r = read_json(path)
                case = int(r["case_id"].split("_")[-1])
                trial = r["evaluation_rng_seed"] - 101
                assert np.isnan(matrix[case, trial]), "Duplicate case/trial"
                assert r["success"] == r["primary_success"] == (r["terminal"] == "success")
                assert r["true_tilt"] == [[-15, 0, 15][case % 3], 20]
                if arm == "wrong":
                    assert not r["conditioning_correct"]
                matrix[case, trial] = r["success"]
                episodes.append(dict(model=model, arm=arm, case=case,
                    trial_seed=r["evaluation_rng_seed"], lateral_tilt_degrees=r["true_tilt"][0],
                    success=int(r["success"]), terminal=r["terminal"], source=str(path.relative_to(ROOT))))
            assert np.isfinite(matrix).all()
            arms[arm] = matrix
        cells = [("zero_lateral_correct", arms["correct"][[1, 4, 7]]),
                 ("all_correct", arms["correct"]), ("all_wrong", arms["wrong"])]
        for (group, values), expected_success in zip(cells, checks):
            assert int(values.sum()) == expected_success
            interval = fixed_case_trial_interval(values, draws=DRAWS, seed=SEED)
            lo, hi = np.asarray(interval["ci95"]) * 100
            summary.append(dict(model=model, group=group, successes=int(values.sum()),
                episodes=values.size, success_percent=100 * values.mean(),
                ci95_low_percent=lo, ci95_high_percent=hi))
    write_csv("figure1_control.csv", summary)
    write_csv("figure1_episodes.csv", episodes)
    return summary


def prediction_results():
    cases = [c for c in read_json(BANK / "manifest.json")["cases"] if c["role"] == "test"]
    assert len(cases) == len({c["source_episode_id"] for c in cases}) == 18
    bank_actions = []
    for case in cases:
        with np.load(read_source(BANK / "public" / (case["case_id"] + ".npz"))) as z:
            assert z["actions"].shape == (3, 16, 6, 3)
            bank_actions.append(z["actions"])
    bank_actions = np.concatenate(bank_actions).reshape(864, 6, 3)
    with np.load(read_source(BANK / "truth/test_fixed16.npz")) as z:
        truth = z["observations"][..., :2].astype(np.float64)
        valid = z["valid"]
    assert truth.shape == (54, 16, 6, 2) and valid.shape == (54, 16, 6)
    assert np.all(np.diff(valid.astype(int), axis=-1) <= 0), "Non-prefix validity"
    np.testing.assert_array_equal(valid.sum((0, 1)), [864, 857, 855, 852, 848, 841])
    assert valid.sum() == 5117
    block_count = valid.reshape(18, 3, 16, 6).sum((1, 2))
    # One common parent resampling schedule for every model and horizon.
    indices = np.random.default_rng(SEED).integers(0, 18, (DRAWS, 18))
    boot_counts = block_count[indices].sum(1)
    summaries, blocks, endpoints = [], [], []
    for model in MODELS:
        folder = FITS / model / "seed_13/test/capabilities"
        with np.load(read_source(folder / "fixed_H6.npz")) as z:
            samples = z["observations"]
            assert samples.shape == (864, 32, 6, 7) and np.isfinite(samples).all()
            np.testing.assert_allclose(z["actions"], np.broadcast_to(bank_actions[:, None], (864, 32, 6, 3)), atol=1e-7, rtol=0)
            mean_xy = samples[..., :2].mean(1, dtype=np.float64).reshape(54, 16, 6, 2)
        errors = np.linalg.norm(mean_xy - truth, axis=-1) * 1000
        # Independently reconcile the H6 endpoint with the original RMSE report.
        rmse = np.sqrt(np.mean(errors[:, :, 5][valid[:, :, 5]] ** 2))
        recorded = read_json(folder / "report.json")["prediction"]["6"]["ball_position_vector_rmse"] * 1000
        np.testing.assert_allclose(rmse, recorded, atol=1e-4, rtol=1e-5)
        block_sum = np.where(valid, errors, 0).reshape(18, 3, 16, 6).sum((1, 2))
        bootstrap = block_sum[indices].sum(1) / boot_counts
        bounds = np.quantile(bootstrap, [.025, .975], axis=0)
        means = block_sum.sum(0) / block_count.sum(0)
        for h in range(6):
            np.testing.assert_allclose(means[h], errors[:, :, h][valid[:, :, h]].mean())
            summaries.append(dict(model=model, prediction_step=h + 1, query_horizon=6,
                valid_endpoints=int(valid[:, :, h].sum()), reference_trajectories=18,
                mean_euclidean_position_error_mm=means[h], ci95_low_mm=bounds[0, h], ci95_high_mm=bounds[1, h]))
        for i, case in enumerate(cases):
            for h in range(6):
                blocks.append(dict(model=model, reference_trajectory=case["case_id"],
                    source_episode_id=case["source_episode_id"], prediction_step=h + 1,
                    valid_endpoints=int(block_count[i, h]), error_sum_mm=block_sum[i, h]))
            for anchor in range(3):
                snapshot = i * 3 + anchor
                for branch in range(16):
                    for h in range(6):
                        ok = bool(valid[snapshot, branch, h])
                        endpoints.append(dict(model=model, reference_trajectory=case["case_id"],
                            anchor=anchor, branch=branch, prediction_step=h + 1, valid=int(ok),
                            predicted_mean_x_m=mean_xy[snapshot, branch, h, 0],
                            predicted_mean_y_m=mean_xy[snapshot, branch, h, 1],
                            actual_x_m=truth[snapshot, branch, h, 0] if ok else "",
                            actual_y_m=truth[snapshot, branch, h, 1] if ok else "",
                            euclidean_error_mm=errors[snapshot, branch, h] if ok else ""))
    write_csv("figure2_prediction.csv", summaries)
    write_csv("figure2_trajectory_blocks.csv", blocks)
    write_csv("figure2_endpoints.csv", endpoints)
    return summaries


def save_figure(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=450, facecolor="white")
    plt.close(fig)


def plot_control(rows, ax):
    ax.set_ylabel("Success rate (%)", labelpad=6)
    settings = [("zero_lateral_correct", "0° lateral, correct", PALETTE["green"], ""),
                ("all_correct", "All tilts, correct", PALETTE["teal"], "//"),
                ("all_wrong", "All tilts, wrong", PALETTE["ink"], "..")]
    width = .235
    for j, (group, label, color, hatch) in enumerate(settings):
        r = [next(r for r in rows if r["model"] == model and r["group"] == group) for model in MODELS]
        values = np.array([v["success_percent"] for v in r])
        lo = np.array([v["ci95_low_percent"] for v in r])
        hi = np.array([v["ci95_high_percent"] for v in r])
        x = np.arange(4) + (j - 1) * width
        ax.bar(x, values, width * .92, color=color, hatch=hatch, edgecolor="white", linewidth=.6, label=label, zorder=3)
        ax.errorbar(x, values, yerr=[values - lo, hi - values], fmt="none", color=PALETTE["ink"], elinewidth=.85, capsize=2, capthick=.85, zorder=4)
        for xi, value, upper in zip(x, values, hi):
            ax.text(xi, upper + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(4), LABELS)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 112)
    ax.set_xlim(-.53, 3.53)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025),
              ncol=3, frameon=False, fontsize=6.2, handlelength=1.3, columnspacing=.8, handletextpad=.4,
              borderaxespad=0, labelspacing=.35)


def plot_prediction(rows, ax):
    ax.set_ylabel("Mean Euclidean position\nerror (mm)", labelpad=6)
    for model, label, marker, style in zip(MODELS, LABELS, ["o", "s", "^", "D"], ["-", "--", "-.", ":"]):
        r = [r for r in rows if r["model"] == model]
        x = [v["prediction_step"] for v in r]
        y = [v["mean_euclidean_position_error_mm"] for v in r]
        color = MODEL_COLORS[model]
        ax.fill_between(x, [v["ci95_low_mm"] for v in r], [v["ci95_high_mm"] for v in r], color=color, alpha=.16, linewidth=0, zorder=2)
        ax.plot(x, y, color=color, linewidth=1.8, marker=marker, markersize=4.7, linestyle=style, label=label, zorder=3)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Prediction step H", labelpad=6)
    ax.set_xlim(.9, 6.1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=4,
              frameon=False, fontsize=7, columnspacing=1.0, handlelength=1.7,
              borderaxespad=0, labelspacing=.6)


def plot_combined(control, prediction):
    fig = plt.figure(figsize=(190 / 25.4, 88 / 25.4), facecolor="white")
    axes = [fig.add_axes([.075, .16, .395, .69]),
            fig.add_axes([.595, .16, .39, .69])]
    titles = ["a. Action-generation and control capability",
              "b. Controlled prediction benchmark"]
    for ax, title in zip(axes, titles):
        fig.text(ax.get_position().x0, .965, title, fontsize=8.5,
                 fontweight="bold", va="top")
        ax.grid(axis="y", linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.7)
    plot_control(control, axes[0])
    plot_prediction(prediction, axes[1])
    save_figure(fig, "figure2_world_models")



def load_summary(name):
    rows = []
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                if key in {"model", "group"}:
                    converted[key] = value
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute-source",
        action="store_true",
        help="Recompute source CSVs from a completed outputs/e1_full campaign.",
    )
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 8, "axes.linewidth": .7, "hatch.linewidth": .45})
    if args.recompute_source:
        if not FITS.exists():
            parser.error("--recompute-source requires outputs/e1_full")
        control = control_results()
        prediction = prediction_results()
        for name, before in SOURCES.items():
            stat = (ROOT / name).stat()
            assert before == (stat.st_size, stat.st_mtime_ns), f"Source changed: {name}"
        receipt = dict(
            bootstrap_draws=DRAWS,
            bootstrap_seed=SEED,
            figure1_resampling="10 proposal-trial seeds; retain all fixed cases in each sampled trial",
            figure2_resampling="18 reference-trajectory blocks with replacement",
            figure2_estimator="pooled Euclidean error of the 32-sample predictive mean",
            training_seed=13,
            retraining_variation_included=False,
            model_queries=0,
            simulator_steps=0,
            sources=list(SOURCES),
        )
        (DATA / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        control = load_summary("figure1_control.csv")
        prediction = load_summary("figure2_prediction.csv")
    plot_combined(control, prediction)
    print(json.dumps({
        "figure": str(OUT / "figure2_world_models.pdf"),
        "source_mode": "recomputed" if args.recompute_source else "retained",
        "control_rows": len(control),
        "prediction_rows": len(prediction),
    }, indent=2))


if __name__ == "__main__":
    main()
