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
from evaluation.switching_dynamics import _rotated_state, rebuild_task_preserving_public_state
from envs.mujoco_tilted_board import MujocoRigidState
from mujoco_task.sim.scene import SceneSpec
from scripts.paper_figure_style import apply_style, PALETTE
from scripts.render_environment_figure import RESOLUTION, make_display_model

RESULTS = ROOT / "results/paper/experiment2"
OUTPUT = ROOT / "results/paper/media/e2_belief_demo.gif"
MODEL_LABELS = {
    "diffusion": "Diffusion/DiT",
    "autoregressive": "Autoregressive Transformer",
    "cvae": "Joint CVAE",
    "flow_matching": "Flow Matching",
}
TILT_LABELS = ("−15°", "0°", "+15°")
TILT_COLORS = (PALETTE["teal"], PALETTE["green"], PALETTE["ink"])
SWITCH_STEP = 10
WIDTH, HEIGHT = 1080, 480
PLOT_WIDTH = 520
RENDER_WIDTH = WIDTH - PLOT_WIDTH
FRAME_MS = 170
TRANSITION_STEPS = 18
TRANSITION_FRAME_MS = 50


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


def interpolate_display_state(before: UphillTask, after: UphillTask, fraction: float):
    """Ease the visible attitude and board-local poses between two saved states."""
    if not 0 <= fraction <= 1:
        raise ValueError("Display interpolation fraction must be in [0, 1]")
    if fraction == 0:
        return before.scene, before.state.copy()
    if fraction == 1:
        return after.scene, after.state.copy()
    weight = fraction * fraction * (3 - 2 * fraction)
    scene = replace(before.scene, lateral_tilt=(
        (1 - weight) * before.scene.lateral_tilt + weight * after.scene.lateral_tilt
    ))
    first = _rotated_state(before.state, before.scene, scene)
    last = _rotated_state(after.state, after.scene, scene)

    def blend(a, b):
        return (1 - weight) * a + weight * b

    def quaternion(a, b):
        # Match quaternion signs before interpolating along the shortest arc.
        b = b if np.dot(a, b) >= 0 else -b
        result = blend(a, b)
        return result / np.linalg.norm(result)

    qpos = blend(first.qpos, last.qpos)
    qpos[3:7] = quaternion(first.qpos[3:7], last.qpos[3:7])
    yaw_delta = (last.rod_yaw - first.rod_yaw + np.pi) % (2 * np.pi) - np.pi
    state = MujocoRigidState(
        qpos=qpos, qvel=blend(first.qvel, last.qvel),
        mocap_pos=blend(first.mocap_pos, last.mocap_pos),
        mocap_quat=np.asarray([quaternion(a, b) for a, b in zip(first.mocap_quat, last.mocap_quat)]),
        rod_yaw=first.rod_yaw + weight * yaw_delta,
        step=first.step, time=blend(first.time, last.time),
    )
    return scene, state


def render_mujoco(env: UphillTask, *, display_scene=None, display_state=None) -> Image.Image:
    # Restore into a separate display model so styling cannot affect the replay.
    scene = env.scene if display_scene is None else display_scene
    state = env.state if display_state is None else display_state
    model = make_display_model(scene, env.config, env.physics_config)
    data = mujoco.MjData(model)
    data.qpos[:] = state.qpos
    data.qvel[:] = state.qvel
    data.mocap_pos[:] = state.mocap_pos
    data.mocap_quat[:] = state.mocap_quat
    data.time = state.time
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qpos, state.qpos)
    with mujoco.Renderer(model, height=RESOLUTION, width=RESOLUTION) as renderer:
        renderer.update_scene(data, camera="orbit")
        frame = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "display_ground")
    floor = ((segmentation[:, :, 0] == ground_id)
             & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
    source = ROOT / "results/paper/source/environment/provenance.json"
    left, top, right, bottom = json.loads(source.read_text())["shared_crop"]
    frame = frame[top:bottom, left:right].copy()
    floor = floor[top:bottom, left:right]
    h, w = floor.shape
    footprint = Image.new("L", (w, h), 0)
    ImageDraw.Draw(footprint).polygon(
        [(33, int(h * .23)), (w - 33, int(h * .23)), (w - 4, h - 24), (4, h - 24)], fill=255,
    )
    frame[floor & (np.asarray(footprint) == 0)] = 255
    image = Image.fromarray(frame)
    image.thumbnail((RENDER_WIDTH - 24, HEIGHT - 76), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (RENDER_WIDTH, HEIGHT), "white")
    canvas.paste(image, ((RENDER_WIDTH - image.width) // 2, 64))
    return canvas


def render_belief(posterior: np.ndarray, observation: int, *, transitioning: bool = False) -> Image.Image:
    apply_style()
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.subplots_adjust(left=.14, right=.97, bottom=.16, top=.83)
    x = np.arange(len(posterior))
    for index, (label, color) in enumerate(zip(TILT_LABELS, TILT_COLORS)):
        ax.plot(x[: observation + 1], posterior[: observation + 1, index], color=color, linewidth=2.6, label=label)
        ax.scatter([observation], [posterior[observation, index]], color=color, s=30, zorder=5)
    ax.axvline(SWITCH_STEP, color=PALETTE["ink"], linestyle="--", linewidth=1.3)
    ax.axhline(.9, color=PALETTE["deep_teal"], alpha=.45, linestyle=(0, (2, 2)), linewidth=1)
    ax.set(xlim=(0, 30), ylim=(0, 1.02), xlabel="Observation", ylabel="Posterior probability")
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False, fontsize=10)
    phase = "physical tilt: 0°" if observation <= SWITCH_STEP else "physical tilt: +15°"
    if transitioning:
        phase = "tilt transition"
    ax.set_title(f"Live hidden-tilt belief  ·  {phase}", fontsize=12, pad=10)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def annotate(frame: Image.Image, observation: int, probability: float, *, transition_tilt=None) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(frame, (PLOT_WIDTH, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.rectangle((PLOT_WIDTH, 0, WIDTH, 64), fill="white")
    draw.text((PLOT_WIDTH + 18, 12), "Matched MuJoCo replay", fill=PALETTE["ink"], font=font)
    detail = f"observation {observation:02d}/30   p(+15°)={probability:.3f}"
    if transition_tilt is not None:
        detail = f"0° → +15°  ·  visual transition ({transition_tilt:+.1f}°)"
    draw.text((PLOT_WIDTH + 18, 38), detail, fill=PALETTE["ink"], font=small)
    return canvas


def quantize_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Use a stable GIF palette with exact paper colors and visible orange balls."""
    samples = Image.new("RGB", (WIDTH // 2, HEIGHT // 2 * len(frames)), "white")
    for index, frame in enumerate(frames):
        samples.paste(frame.resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS),
                      (0, index * HEIGHT // 2))
    palette = samples.quantize(colors=240, method=Image.Quantize.MEDIANCUT)
    reserved = [
        "#ffffff", *PALETTE.values(), "#f08212", "#c46a0f", "#974f0b",
        "#1a47b8", "#292b29", "#8ab07d", "#b8b8b8", "#999999",
    ]
    rgb = palette.getpalette()[:720]
    for color in reserved[:16]:
        rgb.extend(int(color[1 + i:3 + i], 16) for i in (0, 2, 4))
    palette.putpalette((rgb + [255] * 768)[:768])
    quantized = []
    for frame in frames:
        mapped = frame.quantize(palette=palette, dither=Image.Dither.NONE)
        source = np.asarray(frame)
        indices = np.asarray(mapped).copy()
        # Pillow's palette lookup can round white to near-white; preserve exact
        # background and paper colors wherever they occur in the source image.
        for offset, color in enumerate(reserved[:16]):
            rgb_color = tuple(int(color[1 + i:3 + i], 16) for i in (0, 2, 4))
            indices[np.all(source == rgb_color, axis=-1)] = 240 + offset
        encoded = Image.fromarray(indices)
        encoded.putpalette(palette.getpalette())
        quantized.append(encoded)
    return quantized


def main() -> None:
    manifest = json.loads((RESULTS / "trajectories/test/manifest.json").read_text())
    record, posterior, selection = select_trial(load_beliefs(), manifest)
    states = replay_states(record)
    if len(states) != len(posterior):
        raise AssertionError("Replay-state and belief lengths differ")
    frames = []
    durations = []
    transition_frame_indices = []
    transition_angles = []
    for observation, (env, probability) in enumerate(zip(states, posterior[:, 2])):
        plot = render_belief(posterior, observation)
        rendered = render_mujoco(env)
        frame = annotate(rendered, observation, float(probability))
        frame.paste(plot, (0, 0))
        frames.append(frame)
        durations.append(FRAME_MS)
        if observation == SWITCH_STEP:
            durations[-1] = TRANSITION_FRAME_MS
            transition_frame_indices.append(len(frames) - 1)
            transition_angles.append(float(np.rad2deg(env.scene.lateral_tilt)))
            plot = render_belief(posterior, observation, transitioning=True)
            for step in range(1, TRANSITION_STEPS):
                scene, state = interpolate_display_state(env, states[observation + 1], step / TRANSITION_STEPS)
                angle = float(np.rad2deg(scene.lateral_tilt))
                rendered = render_mujoco(env, display_scene=scene, display_state=state)
                frame = annotate(rendered, observation, float(probability), transition_tilt=angle)
                frame.paste(plot, (0, 0))
                transition_frame_indices.append(len(frames))
                transition_angles.append(angle)
                frames.append(frame)
                durations.append(TRANSITION_FRAME_MS)
            transition_frame_indices.append(len(frames))
            transition_angles.append(float(np.rad2deg(states[observation + 1].scene.lateral_tilt)))
    frames = quantize_frames(frames)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    durations[-1] = 1200
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)
    selection.update({
        "frames": len(frames), "frame_duration_ms": FRAME_MS, "output": str(OUTPUT.relative_to(ROOT)),
        "tilt_colors": dict(zip(TILT_LABELS, TILT_COLORS)),
        "plot_style": "Shared paper palette and chart background.",
        "replay_style": "Shared environment-figure display model: gray floor, white background, orange ball, blue pusher.",
        "replay_camera": "Fixed head-on paper environment camera, 35 degree elevation.",
        "style_changes_affect_dynamics": False,
        "gif_palette": "One shared 256-color palette with reserved paper and object colors.",
        "source_observations": len(states),
        "frame_durations_ms": durations,
        "tilt_transition": {
            "kind": "Display-only smoothstep interpolation between recorded observations 10 and 11.",
            "duration_ms": TRANSITION_STEPS * TRANSITION_FRAME_MS,
            "frame_interval_ms": TRANSITION_FRAME_MS,
            "frame_indices": transition_frame_indices,
            "display_tilt_degrees": transition_angles,
            "switch_pause_ms": 0,
            "belief_handling": "Hold the last measured posterior during interpolation; update at observation 11.",
            "experiment_switch": "The recorded physical experiment still uses an instantaneous 0° to +15° switch.",
        },
    })
    OUTPUT.with_suffix(".json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
