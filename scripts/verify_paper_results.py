#!/usr/bin/env python3
"""Derive the five final-paper tables from retained machine-readable artifacts."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper"
APPENDIX = ROOT / "appendix_proposal" / "results"

MODELS = ("diffusion", "autoregressive", "cvae", "flow_matching")
DISPLAY = {
    "diffusion": "Diffusion",
    "autoregressive": "Autoregressive",
    "cvae": "Joint CVAE",
    "flow_matching": "Flow matching",
}
APPENDIX_DISPLAY = {
    "bc_mdn_aif": "BC-MDN",
    "cvae_aif": "CVAE",
    "diffusion_policy_aif": "Diffusion",
    "flow_matching_aif": "Flow",
    "transformer_aif": "Autoregressive",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def table1() -> list[dict]:
    expected = {
        "diffusion": (81, 68, 14.295, 57.123, 9.794),
        "autoregressive": (2, 2, 65.809, 48.735, 12.982),
        "cvae": (49, 61, 15.264, 5.643, 1.236),
        "flow_matching": (43, 34, 24.776, 199.571, 37.037),
    }
    rows = []
    for model in MODELS:
        base = RESULTS / "experiment1" / model
        direct = read_json(base / "every_step" / "report.json")
        reuse = read_json(base / "agreement_reuse" / "report.json")
        capability = read_json(base / "capabilities" / "report.json")
        values = (
            int(direct["successes"]),
            int(reuse["successes"]),
            round(1000 * capability["prediction"]["6"]["ball_position_vector_rmse"], 3),
            round(1000 * direct["model_checker_seconds_per_action"], 3),
            round(1000 * reuse["model_checker_seconds_per_action"], 3),
        )
        assert_equal(f"Table 1/{model}", values, expected[model])
        rows.append(dict(model=DISPLAY[model], direct_success=f"{values[0]}/90",
                         reuse_success=f"{values[1]}/90", h6_rmse_mm=values[2],
                         direct_ms=values[3], reuse_ms=values[4]))
    return rows


def table2() -> list[dict]:
    expected = {
        "diffusion": (98.96, 0.540, 0.062),
        "autoregressive": (79.28, 0.959, 0.127),
        "cvae": (99.07, 0.434, 0.057),
        "flow_matching": (99.07, 0.474, 0.069),
    }
    rows = []
    for model in MODELS:
        metrics = read_json(RESULTS / "experiment1" / model / "capabilities" / "report.json")["tilt_sensing"]
        values = (
            round(100 * metrics["one_step_accuracy"], 2),
            round(metrics["categorical_nll"], 3),
            round(metrics["ECE_10_bins"], 3),
        )
        assert_equal(f"Table 2/{model}", values, expected[model])
        rows.append(dict(model=DISPLAY[model], accuracy_percent=values[0],
                         log_loss=values[1], ece=values[2]))
    return rows


def table3() -> list[dict]:
    expected = {
        "diffusion": (3, 2, 4, 12, 5),
        "autoregressive": (5, 5, 8, 10, 7),
        "cvae": (4, 3, 10, 10, 0),
        "flow_matching": (4, 3, 4, 12, 0),
    }
    rows = read_csv(RESULTS / "experiment2" / "trial_beliefs.csv")
    output = []
    for model in MODELS:
        trials = defaultdict(list)
        for row in rows:
            if row["model"] == model and row["switched"].lower() == "true":
                trials[row["trial_id"]].append(row)
        assert_equal(f"Table 3/{model}/trial count", len(trials), 12)
        delays, final_correct, sustained = [], 0, 0
        for trial in trials.values():
            trial.sort(key=lambda row: int(row["observation"]))
            crossings = [int(row["observation"]) - 10 for row in trial
                         if int(row["observation"]) >= 11 and float(row["p_plus15"]) >= 0.9]
            if crossings:
                delays.append(min(crossings))
            final = next(row for row in trial if int(row["observation"]) == 30)
            final_correct += int(float(final["p_plus15"]) == max(
                float(final["p_minus15"]), float(final["p_flat"]), float(final["p_plus15"])))
            after_six = [row for row in trial if 16 <= int(row["observation"]) <= 30]
            sustained += int(after_six and all(float(row["p_plus15"]) >= 0.9 for row in after_six))
        values = (
            int(statistics.median(delays)),
            min(delays),
            max(delays),
            sum(delay <= 6 for delay in delays),
            final_correct,
        )
        assert_equal(f"Table 3/{model}", values, expected[model])
        assert_equal(f"Table 3/{model}/sustained observations 16-30", sustained, 0)
        output.append(dict(model=DISPLAY[model], delay_median=values[0],
                           delay_range=[values[1], values[2]],
                           identified_by_six=f"{values[3]}/12",
                           correct_observation_30=f"{values[4]}/12"))
    return output


def table_a1() -> list[dict]:
    expected = {
        "bc_mdn_aif": (11, 82.4, 0.75, 346),
        "cvae_aif": (17, 92.1, 0.39, 352),
        "diffusion_policy_aif": (18, 100.0, 50.05, 411),
        "flow_matching_aif": (14, 83.9, 3.86, 354),
        "transformer_aif": (14, 82.1, 14.03, 360),
    }
    rows = read_csv(APPENDIX / "id_n_sweep_summary.csv")
    output = []
    for model in APPENDIX_DISPLAY:
        row = next(row for row in rows if row["variant"] == "generator_aif_score"
                   and row["model"] == model and int(row["K"]) == 128)
        values = (
            round(18 * float(row["success_rate"])),
            round(100 * float(row["mean_feasible_proposal_rate"]), 1),
            round(1000 * float(row["mean_proposal_time_per_decision"]), 2),
            round(1000 * float(row["mean_planning_time_per_decision"])),
        )
        assert_equal(f"Table A1/{model}", values, expected[model])
        output.append(dict(family=APPENDIX_DISPLAY[model], success=f"{values[0]}/18",
                           feasible_percent=values[1], proposal_ms=values[2], plan_ms=values[3]))
    return output


def table_a2() -> list[dict]:
    rows = read_csv(APPENDIX / "per_episode.csv")
    expected = {
        "cem_aif": (8, 0, 0, 44.5, 44.4),
        "cvae_aif": (16, 16, 18, 43.8, 43.5),
    }
    output = []
    for model in ("cem_aif", "cvae_aif"):
        k, nominal_success, transfer_success, nominal_ms, transfer_ms = expected[model]
        subsets = []
        for experiment in ("id_n_sweep", "heldout_start_goal"):
            subset = [row for row in rows
                      if row["experiment"] == experiment
                      and row["model"] == model
                      and int(row["K"]) == k
                      and int(row["equal_wall_clock_budget_ms"]) == 50]
            assert_equal(f"Table A2/{model}/{experiment}/episodes", len(subset), 18)
            successes = sum(row["success"].lower() == "true" for row in subset)
            timing = round(1000 * statistics.fmean(
                float(row["planning_time_per_decision"]) for row in subset), 1)
            subsets.append((successes, timing))
        values = (k, subsets[0][0], subsets[1][0], subsets[0][1], subsets[1][1])
        assert_equal(f"Table A2/{model}", values, expected[model])
        output.append(dict(method="CEM" if model == "cem_aif" else "CVAE", K=k,
                           nominal_success=f"{values[1]}/18", transfer_success=f"{values[2]}/18",
                           nominal_plan_ms=values[3], transfer_plan_ms=values[4]))
    return output


def main() -> None:
    tables = {
        "table1": table1(),
        "table2": table2(),
        "table3": table3(),
        "table_a1": table_a1(),
        "table_a2": table_a2(),
    }
    print(json.dumps({"status": "PASS", **tables}, indent=2))


if __name__ == "__main__":
    main()
