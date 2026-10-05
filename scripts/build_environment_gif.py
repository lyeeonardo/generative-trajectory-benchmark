#!/usr/bin/env python3
"""Animate the three-tilt environment figure using its original successful runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_environment_figure import PANELS, add_trajectory, render_panel
from envs.mujoco_tilted_board import MujocoRigidState


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper/media/environment.gif")
    args = parser.parse_args()
    if args.width < 300:
        parser.error("--width must be at least 300")

    original = json.loads((ROOT / "results/paper/source/environment/provenance.json").read_text())
    crop = tuple(original["shared_crop"])
    panel_width, panel_height = crop[2] - crop[0], crop[3] - crop[1]
    margin = 24
    total_width = 3 * panel_width
    output_size = (args.width, round(args.width * (panel_height + 2 * margin) / (total_width + 2 * margin)))
    footprint = Image.new("L", (total_width, panel_height), 0)
    ImageDraw.Draw(footprint).polygon(
        [(100, int(panel_height * .23)), (total_width - 100, int(panel_height * .23)),
         (total_width - 12, panel_height - 24), (12, panel_height - 24)],
        fill=255,
    )
    ground_mask = np.asarray(footprint) > 0
    panel_sequences = []
    still_panels = []
    records = []
    source_hashes = {}
    frame_interval_ms = None

    for panel_index, panel_args in enumerate(PANELS):
        path = ROOT / original["panels"][panel_index]["source"]
        for source in (path, path.with_suffix(".json")):
            source_hashes[str(source.relative_to(ROOT))] = fingerprint(source)

        sequence = []
        frame_times = []
        ground = ground_mask[:, panel_index * panel_width:(panel_index + 1) * panel_width]

        def crop_panel(rgb, floor):
            rgb = rgb[crop[1]:crop[3], crop[0]:crop[2]].copy()
            floor = floor[crop[1]:crop[3], crop[0]:crop[2]]
            rgb[floor & ~ground] = 255
            return Image.fromarray(rgb)

        def capture(env, renderer, data, end, world, rotation, floor):
            for step in range(end + 1):
                env.set_state(MujocoRigidState(
                    qpos=data["qpos"][step], qvel=data["qvel"][step],
                    mocap_pos=data["mocap_pos"][step], mocap_quat=data["mocap_quat"][step],
                    rod_yaw=float(data["rod_yaw"][step]), step=int(data["step"][step]),
                    time=float(data["time"][step]), integration_state=data["integration_state"][step],
                    integration_spec=int(data["integration_spec"]),
                ))
                np.testing.assert_allclose(
                    env.current_observation()[:4], data["observations"][step, :4], atol=2e-7, rtol=0,
                )
                renderer.update_scene(env.data, camera="orbit")
                add_trajectory(renderer, world, rotation, end)
                sequence.append(crop_panel(renderer.render(), floor))
                frame_times.append(float(data["time"][step]))
            np.testing.assert_allclose(np.diff(frame_times), .05, atol=1e-10, rtol=0)

        frame, floor, record, _ = render_panel(*panel_args, frame_callback=capture)
        still_panels.append(crop_panel(frame, floor))
        panel_sequences.append(sequence)
        frame_interval_ms = round(1000 * (frame_times[1] - frame_times[0]))
        record.update({
            "animation_state_indices": [0, record["success_step"]],
            "animation_end_seconds": frame_times[-1],
            "hold_after_success": True,
        })
        records.append(record)
        print(f"Rendered {record['lateral_tilt_degrees']:+d} degrees: {len(sequence)} recorded states, success.", flush=True)

    def compose(panels):
        canvas = Image.new("RGB", (total_width + 2 * margin, panel_height + 2 * margin), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (margin + index * panel_width, margin))
        return canvas

    # Confirm the shared renderer still reproduces the source figure's composition.
    reference = np.asarray(Image.open(ROOT / "results/paper/figures/environment.png").convert("RGB"), dtype=np.int16)
    reproduced = np.asarray(compose(still_panels), dtype=np.int16)
    if reference.shape != reproduced.shape:
        raise AssertionError("Animation and original figure framing differ")
    still_mean_absolute_error = float(np.abs(reference - reproduced).mean())
    if still_mean_absolute_error > 2:
        raise AssertionError(f"Source figure appearance changed: mean absolute RGB error {still_mean_absolute_error}")

    count = max(map(len, panel_sequences))
    frames = [
        compose([sequence[min(step, len(sequence) - 1)] for sequence in panel_sequences]).resize(
            output_size, Image.Resampling.LANCZOS,
        )
        for step in range(count)
    ]
    panel_sequences.clear()

    # Share one palette across the loop to avoid flickering colors in static surfaces.
    sample_size = (min(768, args.width), round(min(768, args.width) * output_size[1] / output_size[0]))
    palette_samples = Image.new("RGB", (sample_size[0] * 3, sample_size[1] * 3), "white")
    for index, step in enumerate(np.linspace(0, count - 1, 9, dtype=int)):
        palette_samples.paste(frames[step].resize(sample_size, Image.Resampling.LANCZOS),
                              ((index % 3) * sample_size[0], (index // 3) * sample_size[1]))
    # The ball occupies little of the canvas; give its orange shades enough
    # weight that GIF quantization cannot replace them with neutral board colors.
    sample_pixels = np.asarray(palette_samples).astype(np.int16)
    orange = sample_pixels[
        (sample_pixels[:, :, 0] > 1.3 * sample_pixels[:, :, 1])
        & (sample_pixels[:, :, 1] > 1.3 * sample_pixels[:, :, 2])
        & (sample_pixels[:, :, 0] > 80)
    ].astype(np.uint8)
    if len(orange):
        weighted_samples = Image.new("RGB", (palette_samples.width, palette_samples.height + 48), "white")
        weighted_samples.paste(palette_samples, (0, 0))
        weighted_samples.paste(Image.fromarray(np.resize(orange, (48, palette_samples.width, 3))),
                               (0, palette_samples.height))
        palette_samples = weighted_samples
    palette = palette_samples.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [frame_interval_ms] * count
    durations[0] += 600
    durations[-1] = 1800
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)

    with Image.open(args.output) as gif:
        if gif.n_frames != count or gif.size != output_size or gif.info.get("loop") != 0:
            raise AssertionError("Encoded GIF differs from the intended frame count, dimensions, or loop")
        total_duration = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            gif.load()
            total_duration += gif.info["duration"]
        if total_duration != sum(durations):
            raise AssertionError("Encoded GIF timing differs from the recorded replay")

    for source, before in source_hashes.items():
        if fingerprint(ROOT / source) != before:
            raise AssertionError(f"Source changed during rendering: {source}")

    metadata = {
        "source_figure": "results/paper/figures/environment.png",
        "controller": "Recorded reference feedback controller; illustrative successful replay, not a learned-policy evaluation.",
        "panels": records,
        "frame_count": count,
        "frame_interval_ms": frame_interval_ms,
        "pixel_size": list(output_size),
        "duration_ms": total_duration,
        "start_pause_ms": 600,
        "final_hold_ms": 1800,
        "time_alignment": "Shared real simulation time; each panel holds its first successful terminal state.",
        "interpolation": False,
        "still_reproduction_mean_absolute_rgb_error": still_mean_absolute_error,
        "source_sha256": source_hashes,
        "dynamics_steps": 0,
        "model_queries": 0,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": count, "pixels": output_size,
                      "duration_ms": total_duration, "bytes": args.output.stat().st_size,
                      "still_mean_absolute_rgb_error": still_mean_absolute_error}), flush=True)


if __name__ == "__main__":
    main()
