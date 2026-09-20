"""Self-contained task configuration for the MuJoCo rod-push repo."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Literal


PresetName = Literal["smoke", "run"]
ModelName = Literal["vae", "diffuser", "ddiff", "flowmatch", "rssm_cem"]
SplitName = Literal["train", "val", "test"]

OBS_DIM = 14
DYNAMIC_OBS_DIM = 7
STATIC_OBS_DIM = OBS_DIM - DYNAMIC_OBS_DIM
ACTION_DIM = 3
STEP_FEATURE_DIM = DYNAMIC_OBS_DIM + ACTION_DIM
TILT_DIM = 2


@dataclass(frozen=True)
class PathsConfig:
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    results_dir: Path = root / "results"


@dataclass(frozen=True)
class SimulatorConfig:
    dt: float = 0.05
    max_steps: int = 90
    workspace_x: tuple[float, float] = (-0.8, 0.8)
    workspace_y: tuple[float, float] = (-1.0, 1.0)
    ball_radius: float = 0.04
    obstacle_radius: float = 0.10
    goal_radius: float = 0.12
    rod_radius: float = 0.03
    action_max_speed: float = 0.80
    action_max_omega: float = 4.0
    gravity_lateral_gain: float = 1.55
    gravity_longitudinal_gain: float = 1.25
    ball_damping: float = 0.22
    ball_velocity_noise_std: float = 0.004
    contact_push_gain: float = 7.5
    contact_carry_gain: float = 1.00
    controller_position_gain: float = 4.5
    controller_feedforward_gain: float = 0.75
    controller_yaw_gain: float = 5.0
    push_offset: float = 0.018
    success_speed_threshold: float = 1.50
    timeout_distance_penalty: float = 2.0


@dataclass(frozen=True)
class SceneConfig:
    start_x_range: tuple[float, float] = (-0.15, 0.15)
    goal_x_range: tuple[float, float] = (-0.15, 0.15)
    start_y_range: tuple[float, float] = (-0.90, -0.65)
    goal_y_range: tuple[float, float] = (0.65, 0.90)
    lateral_tilts: tuple[float, ...] = tuple(math.radians(value) for value in (-10.0, 0.0, 10.0))
    longitudinal_tilts: tuple[float, ...] = tuple(math.radians(value) for value in (0.0, 10.0, 20.0, 30.0))
    obstacle_center: tuple[float, float] = (0.0, 0.0)
    path_block_margin: float = 0.02
    route_midpoint_margin: float = 0.18


@dataclass(frozen=True)
class DatasetConfig:
    train_families: int = 18
    val_families: int = 4
    test_families: int = 4
    horizon: int = 16
    train_seed: int = 11
    val_seed: int = 17
    test_seed: int = 23
    perturbed_success_mode_period: int = 3


@dataclass(frozen=True)
class ExpertConfig:
    horizon: int = 16
    execute_steps: int = 1
    validation_attempts: int = 4
    cem_iterations: int = 3
    cem_population: int = 24
    cem_elites: int = 6
    cem_action_std: tuple[float, float, float] = (0.18, 0.18, 1.35)
    reference_progress_gain: float = 0.85


@dataclass(frozen=True)
class VAEConfig:
    hidden_dim: int = 192
    latent_dim: int = 20
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    kl_beta: float = 0.01


@dataclass(frozen=True)
class DiffuserConfig:
    hidden_dim: int = 256
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 2e-4
    diffusion_steps: int = 64
    sample_steps: int = 20
    cfg_dropout: float = 0.12
    guidance_scale: float = 2.5


@dataclass(frozen=True)
class FlowMatchConfig:
    hidden_dim: int = 256
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 2e-4
    default_nfe: int = 16


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 48
    kmeans_iters: int = 12


@dataclass(frozen=True)
class DDiffConfig:
    emb_dim: int = 96
    hidden_dim: int = 192
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    refine_steps: int = 6
    mask_schedule: tuple[float, ...] = (1.0, 0.75, 0.55, 0.35, 0.20, 0.0)


@dataclass(frozen=True)
class RSSMCEMConfig:
    latent_dim: int = 24
    deter_dim: int = 96
    hidden_dim: int = 192
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    kl_beta: float = 0.05
    action_std: tuple[float, float, float] = (0.18, 0.18, 1.25)
    default_population: int = 128
    default_iterations: int = 4
    default_elites: int = 16
    uniqueness_threshold: float = 0.55


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 16
    num_candidates: int = 16
    execute_steps: int = 1
    collision_cost: float = 100.0
    timeout_cost: float = 25.0
    goal_distance_cost: float = 12.0
    action_energy_cost: float = 0.08
    smoothness_cost: float = 0.05


@dataclass(frozen=True)
class EvalConfig:
    episodes_per_tilt: int = 6
    sample_candidates: int = 16
    guidance_scale: float = 2.5


@dataclass(frozen=True)
class StudyConfig:
    default_run_seeds: int = 3
    default_smoke_seeds: int = 1
    suite_workers_run: int = 24
    suite_workers_smoke: int = 1
    candidate_workers_run: int = 24
    candidate_workers_smoke: int = 1
    dataset_workers_run: int = 24
    dataset_workers_smoke: int = 1
    progress_log_interval_seconds: float = 30.0
    progress_log_episode_stride: int = 25
    equal_compute_families_run: int = 12
    equal_compute_families_smoke: int = 2
    multimodality_scenes_run: int = 16
    multimodality_scenes_smoke: int = 1
    latency_families_run: int = 6
    latency_families_smoke: int = 1
    latency_smoke_scene_limit: int = 3
    equal_compute_suite_seed: int = 101
    multimodality_suite_seed: int = 211
    conditioning_control_suite_seed: int = 257
    latency_suite_seed: int = 307
    route_band_half_height: float = 0.25
    multimodality_longitudinal_tilt: float = 0.35
    multimodality_candidates: int = 32
    conditioning_ambiguous_families_run: int = 16
    conditioning_ambiguous_families_smoke: int = 1
    conditioning_control_families_run: int = 8
    conditioning_control_families_smoke: int = 2
    conditioning_control_family_id_start: int = 32_000
    conditioning_control_cost_ratio: float = 1.20
    conditioning_smoothness_query_tilts: tuple[float, ...] = (-0.08, -0.04, 0.0, 0.04, 0.08)
    budget_calibration_scene_limit_run: int = 50
    budget_calibration_scene_limit_smoke: int = 6
    budget_calibration_time_tolerance: float = 0.20
    flowmatch_calibration_nfes: tuple[int, ...] = (4, 8, 16, 32)
    flowmatch_guidance_branching_nfes: tuple[int, ...] = (4, 8, 16)
    rssm_population_grid: tuple[int, ...] = (64, 128, 256)
    rssm_iteration_grid: tuple[int, ...] = (2, 4, 6)
    rssm_elite_grid: tuple[int, ...] = (8, 16, 32)
    latency_sample_steps: tuple[int, ...] = (5, 10, 20, 50)
    latency_candidate_counts: tuple[int, ...] = (1, 4, 8, 16)
    latency_execute_steps: tuple[int, ...] = (1, 4, 8)
    guidance_branching_cues: tuple[float, ...] = (-0.06, -0.03, 0.0, 0.03, 0.06)
    guidance_branching_sample_steps: tuple[int, ...] = (5, 10)
    guidance_branching_guidance_scales: tuple[float, ...] = (0.0, 1.0, 2.5, 4.0)
    guidance_branching_num_candidates: int = 16
    guidance_branching_execute_steps: int = 4
    fewshot_bank_seed: int = 401
    fewshot_family_id_start: int = 40_000
    fewshot_longitudinal_tilt: float = 0.35
    fewshot_source_abs_magnitude: float = 0.087266
    fewshot_target_abs_magnitude: float = 0.174533
    fewshot_eval_abs_magnitudes_run: tuple[float, ...] = (0.087266, 0.130900, 0.174533, 0.218166)
    fewshot_eval_abs_magnitudes_smoke: tuple[float, ...] = (0.087266, 0.174533)
    fewshot_source_train_families_run: int = 48
    fewshot_source_val_families_run: int = 8
    fewshot_adapt_pool_families_run: int = 64
    fewshot_eval_families_run: int = 20
    fewshot_source_train_families_smoke: int = 8
    fewshot_source_val_families_smoke: int = 2
    fewshot_adapt_pool_families_smoke: int = 6
    fewshot_eval_families_smoke: int = 4
    fewshot_adapt_successes_per_scene_run: int = 4
    fewshot_adapt_successes_per_scene_smoke: int = 2
    fewshot_budgets_run: tuple[int, ...] = (0, 5, 20, 100, 500)
    fewshot_budgets_smoke: tuple[int, ...] = (0, 5, 20)
    fewshot_finetune_passes: int = 20
    fewshot_finetune_max_steps: int = 2000
    fewshot_success_threshold: float = 0.85
    fewshot_plot_pre_post_budget_run: int = 100
    fewshot_plot_pre_post_budget_smoke: int = 20


@dataclass(frozen=True)
class Preset:
    name: PresetName
    paths: PathsConfig = PathsConfig()
    sim: SimulatorConfig = SimulatorConfig()
    scene: SceneConfig = SceneConfig()
    dataset: DatasetConfig = DatasetConfig()
    expert: ExpertConfig = ExpertConfig()
    vae: VAEConfig = VAEConfig()
    diffuser: DiffuserConfig = DiffuserConfig()
    flowmatch: FlowMatchConfig = FlowMatchConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    ddiff: DDiffConfig = DDiffConfig()
    rssm_cem: RSSMCEMConfig = RSSMCEMConfig()
    mpc: MPCConfig = MPCConfig()
    eval: EvalConfig = EvalConfig()
    study: StudyConfig = StudyConfig()


def _smoke_preset() -> Preset:
    preset = Preset(name="smoke")
    return replace(
        preset,
        sim=replace(preset.sim, max_steps=60, ball_velocity_noise_std=0.002),
        dataset=replace(preset.dataset, train_families=2, val_families=1, test_families=1, horizon=8),
        expert=replace(preset.expert, horizon=8, validation_attempts=4, cem_iterations=2, cem_population=10, cem_elites=3),
        vae=replace(preset.vae, hidden_dim=96, latent_dim=10, batch_size=16, epochs=1),
        diffuser=replace(preset.diffuser, hidden_dim=128, batch_size=16, epochs=1, diffusion_steps=64, sample_steps=8),
        flowmatch=replace(preset.flowmatch, hidden_dim=128, batch_size=16, epochs=1, default_nfe=8),
        tokenizer=replace(preset.tokenizer, vocab_size=24, kmeans_iters=6),
        ddiff=replace(preset.ddiff, emb_dim=48, hidden_dim=96, batch_size=16, epochs=1, refine_steps=4, mask_schedule=(1.0, 0.65, 0.3, 0.0)),
        rssm_cem=replace(preset.rssm_cem, latent_dim=16, deter_dim=48, hidden_dim=96, batch_size=16, epochs=1, default_population=48, default_iterations=2, default_elites=8, uniqueness_threshold=0.40),
        mpc=replace(preset.mpc, horizon=8, num_candidates=6),
        study=replace(
            preset.study,
            budget_calibration_scene_limit_smoke=4,
            flowmatch_calibration_nfes=(4, 8),
            flowmatch_guidance_branching_nfes=(4, 8),
            rssm_population_grid=(32, 64),
            rssm_iteration_grid=(2, 3),
            rssm_elite_grid=(8, 16),
            latency_sample_steps=(5, 10),
            latency_candidate_counts=(1, 4),
            latency_execute_steps=(1, 4),
            guidance_branching_sample_steps=(5,),
            guidance_branching_guidance_scales=(0.0, 2.5),
        ),
        eval=replace(preset.eval, episodes_per_tilt=2, sample_candidates=6, guidance_scale=2.0),
    )


def _run_preset() -> Preset:
    preset = Preset(name="run")
    return replace(preset, dataset=replace(preset.dataset, train_families=36, val_families=8, test_families=12))


def get_preset(name: PresetName | str) -> Preset:
    key = str(name).lower()
    if key == "smoke":
        return _smoke_preset()
    if key == "run":
        return _run_preset()
    raise KeyError(f"Unknown preset {name!r}. Expected one of: smoke, run.")
