"""Shared trajectory generation and likelihood inference for final E2."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import torch

from aif.observation import law_from_samples, moments
from data.preparation import ROOT, write_json
from environment.task import UphillTask
from envs.mujoco_tilted_board import (
    MujocoRigidState,
    _matrix_from_quat,
    _quat_from_matrix,
    board_rotation,
)
from evaluation.common import configure, load_model, public_context, atomic_npz
from mujoco_task.sim.scene import SceneSpec


CONFIG_PATH = ROOT / "configs/experiment2_switching_plus15.json"
OUTPUT = ROOT / "outputs/e2_switching_plus15"
MODEL_LABELS = {
    "diffusion": "Diffusion/DiT",
    "autoregressive": "Autoregressive Transformer",
    "cvae": "Joint CVAE",
    "flow_matching": "Flow Matching",
}
TILT_GRID = np.deg2rad(np.asarray([[-15, 20], [0, 20], [15, 20]], np.float32))
FLAT_INDEX = 1
SWITCHED_INDEX = 2


def read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _source_paths(split: str) -> list[Path]:
    if split not in ("development", "test"):
        raise ValueError("Switching benchmark split must be development or test")
    paths = []
    for condition in (1, 4, 7):
        for side in (-1, 1):
            for episode in (0, 1):
                path = ROOT / f"datasets/uphill_push_v1/episodes/{split}/c{condition}_side{side:+d}/episode_{episode:02d}.npz"
                if not path.exists():
                    raise FileNotFoundError(path)
                paths.append(path)
    if len(paths) != 12:
        raise AssertionError("Expected twelve balanced flat-tilt source episodes")
    return paths


def _rotated_state(state: MujocoRigidState, old_scene: SceneSpec, new_scene: SceneSpec) -> MujocoRigidState:
    """Preserve complete board-local rigid state while changing board attitude."""
    delta = board_rotation(new_scene) @ board_rotation(old_scene).T
    qpos = state.qpos.copy()
    qvel = state.qvel.copy()
    mocap_pos = state.mocap_pos.copy()
    mocap_quat = state.mocap_quat.copy()
    qpos[:3] = delta @ qpos[:3]
    qpos[3:7] = _quat_from_matrix(delta @ _matrix_from_quat(qpos[3:7]))
    qvel[:3] = delta @ qvel[:3]
    qvel[3:6] = delta @ qvel[3:6]
    mocap_pos[:] = (delta @ mocap_pos.T).T
    for i in range(len(mocap_quat)):
        mocap_quat[i] = _quat_from_matrix(delta @ _matrix_from_quat(mocap_quat[i]))
    return MujocoRigidState(
        qpos=qpos,
        qvel=qvel,
        mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
        rod_yaw=float(state.rod_yaw),
        step=int(state.step),
        time=float(state.time),
        integration_state=None,
        integration_spec=None,
    )


def rebuild_task_preserving_public_state(env: UphillTask, new_scene: SceneSpec) -> tuple[UphillTask, dict]:
    """Rebuild MuJoCo at a new tilt without a public-state, time, or dwell reset."""
    if env.scene is None or env.state is None:
        raise RuntimeError("Cannot switch an uninitialized environment")
    old_scene = env.scene
    old_state = env.state.copy()
    old_observation = env.current_observation().copy()
    old_settle = int(env.settle_count)
    rebuilt = UphillTask(max_steps=int(env.config.max_steps))
    rebuilt.reset(new_scene)
    rebuilt.set_state(_rotated_state(old_state, old_scene, new_scene))
    rebuilt.settle_count = old_settle
    new_observation = rebuilt.current_observation().copy()
    public_error = float(np.max(np.abs(old_observation[:12] - new_observation[:12])))
    if public_error > 2e-6:
        raise AssertionError(f"Tilt switch changed public state by {public_error}")
    if rebuilt.state.step != old_state.step or rebuilt.state.time != old_state.time or rebuilt.settle_count != old_settle:
        raise AssertionError("Tilt switch reset clock or dwell state")
    return rebuilt, {
        "maximum_public_state_jump": public_error,
        "step_preserved": True,
        "time_preserved": True,
        "dwell_preserved": True,
        "solver_memory_reset": True,
    }


def _clone_task(env: UphillTask, *, keep_integration: bool) -> UphillTask:
    assert env.scene is not None and env.state is not None
    clone = UphillTask(max_steps=int(env.config.max_steps))
    clone.reset(env.scene)
    state = env.state.copy()
    if not keep_integration:
        state.integration_state = None
        state.integration_spec = None
    clone.set_state(state)
    clone.settle_count = int(env.settle_count)
    return clone


def _trial_id(source_meta: dict, schedule: str) -> str:
    return f"{source_meta['split']}_c{source_meta['case_id']}_side{source_meta['side']:+d}_episode{source_meta['episode_id']}_{schedule}"


def _generate_trial(source: Path, switch_step: int | None, destination: Path, *, switched_tilt_degrees: float = -15.0, max_transitions: int = 100) -> dict:
    if switched_tilt_degrees not in (-15.0, 15.0) or not 1 <= max_transitions <= 100:
        raise ValueError("Expected a trained non-flat tilt and at most 100 transitions")
    switched_index = 0 if switched_tilt_degrees < 0 else 2
    source_meta = json.loads(source.with_suffix(".json").read_text())
    with np.load(source, allow_pickle=False) as saved:
        source_actions = saved["actions"].copy()
        source_observations = saved["observations"].copy()
    if source_meta["lateral_degrees"] != 0 or source_meta["longitudinal_degrees"] != 20:
        raise ValueError("Switching sources must begin at 0-degree lateral tilt")
    scene = SceneSpec(**source_meta["scene"])
    env = UphillTask(max_steps=100)
    initial = env.reset(scene)
    np.testing.assert_allclose(initial[:7], source_observations[0, :7], atol=2e-6, rtol=0)
    schedule = "no_switch" if switch_step is None else f"switch_{switch_step:02d}"
    observations = [initial[:7].copy()]
    actions, events = [], []
    true_indices = [FLAT_INDEX]
    switch_audit = None
    no_op_next_error = None
    terminal = "source_action_exhaustion"
    for step, action in enumerate(source_actions[:max_transitions]):
        if switch_step is not None and step == switch_step:
            continuous = _clone_task(env, keep_integration=True)
            rebuilt_flat = _clone_task(env, keep_integration=False)
            continuous_result = continuous.step(action)
            rebuilt_result = rebuilt_flat.step(action)
            no_op_next_error = np.abs(continuous_result.observation[:7] - rebuilt_result.observation[:7])
            switched_scene = replace(scene, lateral_tilt=float(np.deg2rad(switched_tilt_degrees)))
            env, switch_audit = rebuild_task_preserving_public_state(env, switched_scene)
        result = env.step(action)
        actions.append(np.asarray(action, np.float32))
        observations.append(result.observation[:7].copy())
        true_indices.append(switched_index if switch_step is not None and step >= switch_step else FLAT_INDEX)
        events.append([result.success, result.collision, result.info["fall_out"], result.timeout, result.info["rod_ball_contact"]])
        if result.collision:
            terminal = "collision"
            break
        if result.info["fall_out"]:
            terminal = "fall"
            break
    if terminal == "source_action_exhaustion" and len(actions) == max_transitions:
        terminal = "observation_window_complete"
    observations = np.asarray(observations, np.float32)
    actions = np.asarray(actions, np.float32)
    events = np.asarray(events, bool)
    true_indices = np.asarray(true_indices, np.int8)
    if switch_step is None:
        np.testing.assert_allclose(observations, source_observations[: len(observations), :7], atol=2e-6, rtol=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        destination,
        public_observations=observations,
        actions=actions,
        events=events,
        true_tilt_index=true_indices,
        geometry=np.asarray(initial[7:12], np.float32),
    )
    record = {
        "trial_id": destination.stem,
        "source_episode_id": int(source_meta["episode_id"]),
        "source": str(source.relative_to(ROOT)),
        "split": source_meta["split"],
        "condition_id": int(source_meta["case_id"]),
        "obstacle_x": float(source_meta["obstacle_x"]),
        "route_side": int(source_meta["side"]),
        "schedule": schedule,
        "switch_step": switch_step,
        "transitions": int(len(actions)),
        "terminal": terminal,
        "trajectory": str(destination.relative_to(ROOT)),
        "public_state_jump_at_switch": None if switch_audit is None else switch_audit["maximum_public_state_jump"],
        "no_op_rebuild_next_position_error_m": None if no_op_next_error is None else float(np.linalg.norm(no_op_next_error[:2])),
        "no_op_rebuild_next_velocity_error_m_s": None if no_op_next_error is None else float(np.linalg.norm(no_op_next_error[2:4])),
        "clock_and_dwell_preserved": switch_audit is None or all(switch_audit[k] for k in ("step_preserved", "time_preserved", "dwell_preserved")),
        "hidden_tilt_stored_in_model_input": False,
    }
    return record


def generate_trajectories(split: str) -> dict:
    config = read_config()
    root = OUTPUT / "trajectories" / split
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    rows = []
    schedules = [None] + [int(x) for x in config["switch_steps"]]
    for source in _source_paths(split):
        meta = json.loads(source.with_suffix(".json").read_text())
        for switch_step in schedules:
            schedule = "no_switch" if switch_step is None else f"switch_{switch_step:02d}"
            tid = _trial_id(meta, schedule)
            rows.append(_generate_trial(source, switch_step, root / f"{tid}.npz",
                switched_tilt_degrees=float(config["switched_lateral_tilt_degrees"]),
                max_transitions=int(config.get("maximum_transitions", 100))))
    no_op_pos = [r["no_op_rebuild_next_position_error_m"] for r in rows if r["switch_step"] is not None]
    no_op_vel = [r["no_op_rebuild_next_velocity_error_m_s"] for r in rows if r["switch_step"] is not None]
    report = {
        "status": "COMPLETE",
        "split": split,
        "trials": len(rows),
        "source_episode_blocks": len({r["source_episode_id"] for r in rows}),
        "switch_trials": sum(r["switch_step"] is not None for r in rows),
        "no_switch_trials": sum(r["switch_step"] is None for r in rows),
        "safety_terminated": sum(r["terminal"] in ("collision", "fall") for r in rows),
        "maximum_public_state_jump": max(r["public_state_jump_at_switch"] or 0.0 for r in rows),
        "maximum_no_op_rebuild_next_position_error_m": max(no_op_pos),
        "maximum_no_op_rebuild_next_velocity_error_m_s": max(no_op_vel),
        "model_specific_trajectories": False,
        "model_inputs_contain_hidden_tilt": False,
        "rows": rows,
    }
    write_json(manifest_path, report)
    return report


def _load_trials(split: str) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
    report = generate_trajectories(split)
    arrays = []
    for row in report["rows"]:
        with np.load(ROOT / row["trajectory"], allow_pickle=False) as saved:
            arrays.append({key: saved[key] for key in saved.files})
    return report["rows"], arrays


def _model_paths(model: str) -> tuple[Path, Path]:
    root = ROOT / f"outputs/e1_full/fits/{model}/seed_13"
    return root / "selected/checkpoint.pt", root / "calibration/calibration.json"


def _numpy(outputs: dict) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value) for key, value in outputs.items()}


@torch.no_grad()
def infer_likelihoods(model_name: str, split: str, device: str = "cuda") -> dict:
    destination = OUTPUT / "inference" / split / model_name
    report_path = destination / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    config = read_config()
    checkpoint, calibration_path = _model_paths(model_name)
    calibration = json.loads(calibration_path.read_text())
    rows, trials = _load_trials(split)
    configure(7300 + list(MODEL_LABELS).index(model_name), device)
    model, _ = load_model(checkpoint, device)
    model.sampling_temperature = float(calibration["sampling_temperature"])
    model.guidance_scale = float(calibration["guidance_scale"])
    samples_per_hypothesis = int(config["prediction_samples_per_hypothesis"])
    max_steps = max(len(t["actions"]) for t in trials)
    log_likelihood = np.full((len(trials), max_steps, 3), np.nan, np.float64)
    prediction_mean = np.full((len(trials), max_steps, 3, 7), np.nan, np.float32)
    prediction_variance = np.full_like(prediction_mean, np.nan)
    valid = np.zeros((len(trials), max_steps), bool)
    started = time.perf_counter()
    batches = 0
    predicted_samples = 0
    for step in range(max_steps):
        active = [i for i, trial in enumerate(trials) if step < len(trial["actions"])]
        if not active:
            continue
        actions = np.stack([trials[i]["actions"][step] for i in active])[:, None]
        truth = torch.as_tensor(np.stack([trials[i]["public_observations"][step + 1] for i in active]), device=device)
        for hypothesis, tilt in enumerate(TILT_GRID):
            contexts = [
                public_context(
                    trials[i]["public_observations"][: step + 1],
                    trials[i]["actions"][:step],
                    trials[i]["geometry"],
                    tilt,
                    step,
                )
                for i in active
            ]
            context = {key: np.concatenate([row[key] for row in contexts], axis=0) for key in contexts[0]}
            generator = torch.Generator(device=device).manual_seed(910000 + step)
            chunks = []
            for begin in range(0, len(active), 48):
                end = min(begin + 48, len(active))
                output = _numpy(model.predict(
                    {key: value[begin:end] for key, value in context.items()},
                    actions[begin:end],
                    samples=samples_per_hypothesis,
                    generator=generator,
                ))
                expected = np.broadcast_to(actions[begin:end, None], output["actions"].shape)
                if not np.array_equal(output["actions"], expected):
                    raise AssertionError("Controlled prediction changed an imposed action")
                chunks.append(output["observations"][:, :, 0])
                batches += 1
            sample = torch.as_tensor(np.concatenate(chunks), device=device)
            law = law_from_samples(sample, calibration["extra_variance"])
            score = law.log_prob(truth).detach().cpu().numpy()
            mean, raw_variance = moments(sample)
            mean = mean.detach().cpu().numpy()
            variance = (raw_variance + torch.as_tensor(calibration["extra_variance"], device=device)).detach().cpu().numpy()
            log_likelihood[active, step, hypothesis] = score
            prediction_mean[active, step, hypothesis] = mean
            prediction_variance[active, step, hypothesis] = variance
            predicted_samples += len(active) * samples_per_hypothesis
        valid[active, step] = True
    elapsed = time.perf_counter() - started
    if not np.isfinite(log_likelihood[valid]).all() or not np.isfinite(prediction_mean[valid]).all() or not np.isfinite(prediction_variance[valid]).all():
        raise FloatingPointError("Switching likelihood inference produced nonfinite values")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_npz(destination / "likelihoods.npz", log_likelihood=log_likelihood,
        prediction_mean=prediction_mean, prediction_variance=prediction_variance, valid=valid)
    checkpoint_stat = checkpoint.stat()
    calibration_stat = calibration_path.stat()
    report = {
        "status": "COMPLETE",
        "model": model_name,
        "display_name": MODEL_LABELS[model_name],
        "split": split,
        "trials": len(trials),
        "valid_transitions": int(valid.sum()),
        "hypotheses": 3,
        "samples_per_hypothesis": samples_per_hypothesis,
        "predictive_samples": predicted_samples,
        "model_batches": batches,
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "calibration": str(calibration_path.relative_to(ROOT)),
        "calibration_size": calibration_stat.st_size,
        "calibration_mtime_ns": calibration_stat.st_mtime_ns,
        "sampling_temperature": calibration["sampling_temperature"],
        "guidance_scale": calibration["guidance_scale"],
        "extra_variance": calibration["extra_variance"],
        "common_prediction_draws_across_hypotheses": True,
        "query_horizon": 1,
        "quality": "NULL",
        "true_tilt_available_to_model": False,
        "physical_simulator_queries_during_inference": 0,
    }
    write_json(report_path, report)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report
