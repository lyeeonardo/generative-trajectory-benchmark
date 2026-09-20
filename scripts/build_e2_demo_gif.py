#!/usr/bin/env python3
"""Build the README E2 belief-update/MuJoCo replay animation."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import replace
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.task import UphillTask
from evaluation.switching_dynamics import rebuild_task_preserving_public_state
from mujoco_task.sim.scene import SceneSpec
from scripts.paper_figure_style import apply_style, PALETTE

RESULTS = ROOT / "results/paper/experiment2"
OUTPUT = ROOT / "results/paper/media/e2_belief_demo.gif"
MODEL_LABELS = {
    "diffusion": "Diffusion/DiT",
    "autoregressive": "Autoregressive Transformer",
    "cvae": "Joint CVAE",
    "flow_matching": "Flow Matching",
}
TILT_LABELS = ("−15°", "0°", "+15°")
TILT_COLORS = ("#287779", "#8f8a7d", "#d97732")
SWITCH_STEP = 10
WIDTH, HEIGHT = 1080, 480
PLOT_WIDTH = 520
RENDER_WIDTH = WIDTH - PLOT_WIDTH
FRAME_MS = 170


def load_beliefs() -> dict[tuple[str, str], np.ndarray]:
    rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    with (RESULTS / "trial_beliefs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["switched"] == "True":
                rows.setdefault((row["trial_id"], row["model"]), []).append(row)
    beliefs = {}
    for key, values in rows.items():
        values.sort(key=lambda row: int(row["observation"]))
        beliefs[key] = np.asarray(
            [[float(row["p_minus15"]), float(row["p_flat"]), float(row["p_plus15"])] for row in values],
            dtype=np.float64,
        )
    return beliefs


def select_trial(beliefs: dict[tuple[str, str], np.ndarray], manifest: dict) -> tuple[dict, np.ndarray, dict]:
    """Choose the correct-final model/trial pair with the strongest late belief."""
    records = {row["trial_id"]: row for row in manifest["rows"] if row["switch_step"] is not None}
    candidates = []
    for (trial_id, model), posterior in beliefs.items():
        if int(np.argmax(posterior[-1])) != 2:
            continue
        crossing = np.flatnonzero(posterior[SWITCH_STEP + 1 :, 2] >= 0.9)
        delay = int(crossing[0] + 1) if len(crossing) else 10_000
        late_mean = float(np.mean(posterior[16:31, 2]))
        candidates.append((late_mean, -delay, float(posterior[-1, 2]), model, trial_id))
    if not candidates:
        raise RuntimeError("No correct-final switched trial is available for the demo")
    late_mean, negative_delay, final_probability, model, trial_id = max(candidates)
    posterior = beliefs[(trial_id, model)]
    record = records[trial_id]
    selection = {
        "selection_rule": "Among all switched model/trial pairs ending in the correct +15° class, maximize mean p(+15°) over observations 16–30; then prefer shorter threshold delay and larger final probability.",
        "trial_id": trial_id,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "source_episode_id": record["source_episode_id"],
        "mean_p_plus15_observations_16_30": late_mean,
        "threshold_0_9_delay_after_switch": -negative_delay,
        "final_p_plus15": final_probability,
        "switch_step": SWITCH_STEP,
    }
    return record, posterior, selection


def replay_states(record: dict) -> list[UphillTask]:
    source = ROOT / record["source"]
    meta = json.loads(source.with_suffix(".json").read_text())
    scene = SceneSpec(**meta["scene"])
    with np.load(ROOT / record["trajectory"], allow_pickle=False) as saved:
        actions = saved["actions"].copy()
        expected = saved["public_observations"].copy()
    env = UphillTask(max_steps=100)
    env.reset(scene)
    states = [env.copy()]
    for step, action in enumerate(actions):
        if step == SWITCH_STEP:
            switched_scene = replace(scene, lateral_tilt=float(np.deg2rad(15)))
            env, _ = rebuild_task_preserving_public_state(env, switched_scene)
        env.step(action)
        np.testing.assert_allclose(env.current_observation()[:7], expected[step + 1], atol=2e-6, rtol=0)
        states.append(env.copy())
    return states


def render_mujoco(env: UphillTask) -> Image.Image:
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=RENDER_WIDTH)
    try:
        renderer.update_scene(env.data, camera="orbit")
        frame = renderer.render().copy()
    finally:
        renderer.close()
    return Image.fromarray(frame)


def render_belief(posterior: np.ndarray, observation: int) -> Image.Image:
    apply_style()
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.subplots_adjust(left=.14, right=.97, bottom=.16, top=.83)
    x = np.arange(len(posterior))
    for index, (label, color) in enumerate(zip(TILT_LABELS, TILT_COLORS)):
        ax.plot(x[: observation + 1], posterior[: observation + 1, index], color=color, linewidth=2.6, label=label)
        ax.scatter([observation], [posterior[observation, index]], color=color, s=30, zorder=5)
    ax.axvline(SWITCH_STEP, color=PALETTE["ink"], linestyle="--", linewidth=1.3)
    ax.axhline(.9, color="#aaa394", linestyle=(0, (2, 2)), linewidth=1)
    ax.set(xlim=(0, 30), ylim=(0, 1.02), xlabel="Observation", ylabel="Posterior probability")
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False, fontsize=10)
    phase = "physical tilt: 0°" if observation <= SWITCH_STEP else "physical tilt: +15°"
    ax.set_title(f"Live hidden-tilt belief  ·  {phase}", fontsize=12, pad=10)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def annotate(frame: Image.Image, observation: int, probability: float) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(frame, (PLOT_WIDTH, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.rectangle((PLOT_WIDTH, 0, WIDTH, 64), fill=(255, 255, 255, 228))
    draw.text((PLOT_WIDTH + 18, 12), "Matched MuJoCo replay", fill="#274753", font=font)
    draw.text((PLOT_WIDTH + 18, 38), f"observation {observation:02d}/30   p(+15°)={probability:.3f}", fill="#274753", font=small)
    if observation == SWITCH_STEP:
        draw.rectangle((PLOT_WIDTH + 12, HEIGHT - 48, WIDTH - 12, HEIGHT - 12), fill="#d97732")
        draw.text((PLOT_WIDTH + 25, HEIGHT - 40), "UNANNOUNCED 0° → +15° TILT SWITCH", fill="white", font=small)
    return canvas


def main() -> None:
    manifest = json.loads((RESULTS / "trajectories/test/manifest.json").read_text())
    record, posterior, selection = select_trial(load_beliefs(), manifest)
    states = replay_states(record)
    if len(states) != len(posterior):
        raise AssertionError("Replay-state and belief lengths differ")
    frames = []
    for observation, (env, probability) in enumerate(zip(states, posterior[:, 2])):
        plot = render_belief(posterior, observation)
        rendered = render_mujoco(env)
        frame = annotate(rendered, observation, float(probability))
        frame.paste(plot, (0, 0))
        frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    durations = [FRAME_MS] * len(frames)
    durations[SWITCH_STEP] = 850
    durations[-1] = 1200
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)
    selection.update({"frames": len(frames), "frame_duration_ms": FRAME_MS, "output": str(OUTPUT.relative_to(ROOT))})
    OUTPUT.with_suffix(".json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
