"""Registry for Generator proposal generators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from data.archive_dataset import CONTEXT_DIM
from generators.cem import CEMGenerator
from generators.cvae import CVAEActionGenerator, CVAEConfig
from generators.diffuser import DiffuserConfig, DiffuserGenerator
from generators.diffusion_policy import DiffusionPolicyConfig, DiffusionPolicyGenerator
from generators.flow_matching import FlowMatchingConfig, FlowMatchingGenerator
from generators.normalizing_flow import NormalizingFlowConfig, NormalizingFlowGenerator
from generators.pets_cem import PETSCEMConfig, PETSCEMGenerator
from generators.prototype import ContextGaussianPrototypeGenerator
from generators.random import RandomShootingGenerator
from generators.transformer import TransformerActionGenerator, TransformerConfig
from generators.base import ProposalGenerator
from generators.bc_mdn import BCMDNConfig, BCMDNGenerator
from generators.composite import EnsembleGenerator, OracleSourceSelectorGenerator, RepairGenerator
from generators.probe_router import LinearRoutingModel, ProbeAndRouteGenerator

REQUIRED_GENERATOR_NAMES = (
    "cem_aif",
    "bc_mdn_aif",
    "cvae_aif",
    "diffusion_policy_aif",
    "diffuser_aif",
    "flow_matching_aif",
    "normalizing_flow_aif",
    "transformer_aif",
    "pets_cem_aif",
)

ALIASES = {
    "cem": "cem_aif",
    "cvae": "cvae_aif",
    "random": "random_shooting_aif",
    "random_aif": "random_shooting_aif",
}


def _cfg(config: dict[str, Any] | None, key: str, default: Any) -> Any:
    return default if config is None else config.get(key, default)


def get_generator(name: str, config: dict[str, Any] | None = None) -> ProposalGenerator:
    requested = str(name)
    key = ALIASES.get(requested, requested)
    config = dict(config or {})
    if key in {"ensemble_aif", "oracle_source_selector_aif"}:
        spec_key = "ensemble_sources" if key == "ensemble_aif" else "oracle_source_specs"
        specs = config.get(spec_key, config.get("source_specs", []))
        if not isinstance(specs, (list, tuple)) or len(specs) < 2:
            raise ValueError(f"{key} requires at least two {spec_key}")
        generators = [_generator_from_spec(spec, config) for spec in specs]
        if key == "ensemble_aif":
            return EnsembleGenerator(generators)
        return OracleSourceSelectorGenerator(generators)
    if key == "probe_router_aif":
        primary_spec = config.get("router_primary_spec", config.get("router_primary_model", "cvae_aif"))
        fallback_spec = config.get("router_fallback_spec", config.get("fallback_model", "cem_aif"))
        primary = _generator_from_spec(primary_spec, config)
        fallback = _generator_from_spec(fallback_spec, config)
        model_path = config.get("routing_model_path")
        routing_model = LinearRoutingModel.load(model_path) if model_path else LinearRoutingModel.default(
            rho_min=float(_cfg(config, "rho_min", 0.05)),
            rho_max=float(_cfg(config, "rho_max", 0.95)),
        )
        return ProbeAndRouteGenerator(
            primary,
            fallback,
            routing_model=routing_model,
            probe_per_source=int(_cfg(config, "router_probe_per_source", 8)),
            risk_mode=str(_cfg(config, "router_risk_mode", "cvar")),
            cvar_alpha=float(_cfg(config, "router_cvar_alpha", 0.25)),
            risk_beta=float(_cfg(config, "router_risk_beta", 1.0)),
            score_scale=float(_cfg(config, "router_score_scale", 50.0)),
            prediction_error_scale=float(_cfg(config, "router_prediction_error_scale", 1.0)),
            belief_kl_scale=float(_cfg(config, "router_belief_kl_scale", 1.0)),
            regret_scale=float(_cfg(config, "router_regret_scale", 50.0)),
            collect_oracle_labels=bool(_cfg(config, "router_collect_oracle_labels", False)),
            oracle_candidate_count=config.get("router_oracle_candidate_count"),
        )
    if key == "repair_aif":
        base_spec = config.get("repair_base_spec", config.get("repair_base_model", "cvae_aif"))
        base_generator = _generator_from_spec(base_spec, config)
        return RepairGenerator(
            base_generator,
            base_fraction=float(_cfg(config, "repair_base_fraction", 0.25)),
            population_multiplier=float(_cfg(config, "repair_population_multiplier", 4.0)),
            noise_std=tuple(_cfg(config, "repair_noise_std", [0.08, 0.08, 0.40])),
            risk_mode=str(_cfg(config, "repair_risk_mode", "cvar")),
            cvar_alpha=float(_cfg(config, "repair_cvar_alpha", 0.25)),
            risk_beta=float(_cfg(config, "repair_risk_beta", 1.0)),
        )
    if key == "cem_aif":
        gen = CEMGenerator(
            iterations=int(_cfg(config, "cem_iterations", 1)),
            population=int(_cfg(config, "cem_population", _cfg(config, "K", 32))),
            elites=int(_cfg(config, "cem_elites", 8)),
            elite_frac=config.get("cem_elite_frac"),
            init_std=tuple(_cfg(config, "init_std", [0.25, 0.25, 1.25])),
            min_std=tuple(_cfg(config, "min_std", [0.02, 0.02, 0.10])),
            smoothing_alpha=float(_cfg(config, "smoothing_alpha", 1.0)),
            action_squash=str(_cfg(config, "action_squash", "clip")),
            risk_mode=str(_cfg(config, "cem_risk_mode", "mean")),
            cvar_alpha=float(_cfg(config, "cem_cvar_alpha", 0.25)),
            risk_beta=float(_cfg(config, "cem_risk_beta", 1.0)),
        )
        gen.name = "cem_aif"
        return gen
    if key == "random_shooting_aif":
        gen = RandomShootingGenerator(action_std=tuple(_cfg(config, "random_action_std", [0.35, 0.35, 1.5])))
        gen.name = "random_shooting_aif"
        return gen
    if key == "cvae_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            gen = CVAEActionGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
            if config.get("cvae_sample_noise_std") is not None:
                gen.sample_noise_std = np.asarray(config["cvae_sample_noise_std"], dtype=np.float32)
        else:
            gen = CVAEActionGenerator(
                CVAEConfig(
                    context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                    horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                    action_dim=int(_cfg(config, "action_dim", 3)),
                    hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                    latent_dim=int(_cfg(config, "latent_dim", 8)),
                    beta_kl=float(_cfg(config, "beta_kl", 0.01)),
                    recon_weight=float(_cfg(config, "recon_weight", 1.0)),
                    sample_noise_std=tuple(_cfg(config, "cvae_sample_noise_std", [0.0, 0.0, 0.0])),
                ),
                device=config.get("device", "cpu"),
            )
        gen.name = "cvae_aif"
        gen.is_learned = True
        gen.supports_log_prob = False
        gen.supports_guidance = False
        gen.is_stochastic = True
        return gen
    if key == "bc_mdn_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return BCMDNGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
        return BCMDNGenerator(
            BCMDNConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                variant=str(_cfg(config, "variant", "mdn")),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_components=int(_cfg(config, "num_components", 5)),
                dropout=float(_cfg(config, "dropout", 0.0)),
                layer_norm=bool(_cfg(config, "layer_norm", False)),
            ),
            device=config.get("device", "cpu"),
        )
    if key == "diffusion_policy_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            gen = DiffusionPolicyGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
            if config.get("sample_steps") is not None:
                gen.config = replace(gen.config, sample_steps=int(config["sample_steps"]))
            return gen
        return DiffusionPolicyGenerator(
            DiffusionPolicyConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_layers=int(_cfg(config, "num_layers", 2)),
                num_heads=int(_cfg(config, "num_heads", 4)),
                dropout=float(_cfg(config, "dropout", 0.0)),
                diffusion_steps=int(_cfg(config, "diffusion_steps", 32)),
                sample_steps=int(_cfg(config, "sample_steps", 8)),
                context_dropout=float(_cfg(config, "context_dropout", 0.0)),
                backbone=str(_cfg(config, "backbone", "mlp")),
            ),
            device=config.get("device", "cpu"),
        )
    if key == "diffuser_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return DiffuserGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
        return DiffuserGenerator(
            DiffuserConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_layers=int(_cfg(config, "num_layers", 2)),
                num_heads=int(_cfg(config, "num_heads", 4)),
                dropout=float(_cfg(config, "dropout", 0.0)),
                diffusion_steps=int(_cfg(config, "diffusion_steps", 32)),
                sample_steps=int(_cfg(config, "sample_steps", 8)),
                context_dropout=float(_cfg(config, "context_dropout", 0.0)),
                backbone=str(_cfg(config, "backbone", "mlp")),
                mode=str(_cfg(config, "mode", _cfg(config, "target_mode", "action_only"))),
                state_dim=int(_cfg(config, "state_dim", 14)),
                state_unit_clip=float(_cfg(config, "state_unit_clip", 5.0)),
            ),
            device=config.get("device", "cpu"),
        )
    if key == "flow_matching_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return FlowMatchingGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
        return FlowMatchingGenerator(
            FlowMatchingConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_layers=int(_cfg(config, "num_layers", 2)),
                ode_steps=int(_cfg(config, "ode_steps", 8)),
                solver=str(_cfg(config, "solver", "euler")),
                path_type=str(_cfg(config, "path_type", "linear")),
                context_dropout=float(_cfg(config, "context_dropout", 0.0)),
                action_squash=str(_cfg(config, "action_squash", "clip")),
            ),
            device=config.get("device", "cpu"),
        )
    if key == "normalizing_flow_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return NormalizingFlowGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
        return NormalizingFlowGenerator(
            NormalizingFlowConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_layers=int(_cfg(config, "num_layers", 2)),
                num_coupling_layers=int(_cfg(config, "num_coupling_layers", 4)),
                scale_clip=float(_cfg(config, "scale_clip", 2.0)),
                min_log_std=float(_cfg(config, "min_log_std", -5.0)),
                max_log_std=float(_cfg(config, "max_log_std", 2.0)),
                action_squash_eps=float(_cfg(config, "action_squash_eps", 1e-5)),
            ),
            device=config.get("device", "cpu"),
        )
    if key == "transformer_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return TransformerActionGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
        return TransformerActionGenerator(
            TransformerConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_layers=int(_cfg(config, "num_layers", 1)),
                num_heads=int(_cfg(config, "num_heads", 2)),
                dropout=float(_cfg(config, "dropout", 0.0)),
                num_components=int(_cfg(config, "num_components", 3)),
                variant=str(_cfg(config, "variant", "gmm")),
                min_log_std=float(_cfg(config, "min_log_std", -5.0)),
                max_log_std=float(_cfg(config, "max_log_std", 2.0)),
                action_squash_eps=float(_cfg(config, "action_squash_eps", 1e-5)),
                temperature=float(_cfg(config, "temperature", 1.0)),
            ),
            device=config.get("device", "cpu"),
        )
    if key == "pets_cem_aif":
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return PETSCEMGenerator.load_checkpoint(checkpoint_path, device=config.get("device", "cpu"))
        return PETSCEMGenerator(
            PETSCEMConfig(
                context_dim=int(_cfg(config, "context_dim", CONTEXT_DIM)),
                horizon=int(_cfg(config, "H", _cfg(config, "horizon", 8))),
                action_dim=int(_cfg(config, "action_dim", 3)),
                obs_dim=int(_cfg(config, "obs_dim", 14)),
                ensemble_size=int(_cfg(config, "ensemble_size", 3)),
                hidden_dim=int(_cfg(config, "hidden_dim", 64)),
                num_layers=int(_cfg(config, "num_layers", 2)),
                cem_iterations=int(_cfg(config, "cem_iterations", 2)),
                cem_population=int(_cfg(config, "cem_population", _cfg(config, "K", 32))),
                cem_elite_frac=float(_cfg(config, "cem_elite_frac", 0.25)),
                init_std=tuple(_cfg(config, "init_std", [0.25, 0.25, 1.25])),
                min_std=tuple(_cfg(config, "min_std", [0.02, 0.02, 0.10])),
                smoothing_alpha=float(_cfg(config, "smoothing_alpha", 1.0)),
            ),
            device=config.get("device", "cpu"),
        )
    prototype_specs = {
    }
    if key in prototype_specs:
        return ContextGaussianPrototypeGenerator(name=key, **prototype_specs[key])
    raise KeyError(f"Unknown generator {name!r}. Expected one of: {', '.join(REQUIRED_GENERATOR_NAMES)}")


def _generator_from_spec(spec: Any, config: dict[str, Any]) -> ProposalGenerator:
    label: str | None = None
    if isinstance(spec, str):
        source_name = spec
        overrides: dict[str, Any] = {}
    elif isinstance(spec, dict):
        source_name = str(spec.get("name", ""))
        overrides = dict(spec.get("overrides", {}))
        label = None if spec.get("label") is None else str(spec["label"])
    else:
        raise TypeError(f"Generator source spec must be a name or mapping, got {type(spec).__name__}")
    if not source_name:
        raise ValueError("Generator source spec is missing name")
    source_config = {**config, **overrides}
    checkpoint_paths = config.get("checkpoint_paths", {})
    if isinstance(checkpoint_paths, dict) and checkpoint_paths.get(source_name):
        source_config["checkpoint_path"] = checkpoint_paths[source_name]
    generator = get_generator(source_name, source_config)
    if label:
        generator.name = label
    return generator


def all_generator_names(include_optional: bool = False) -> tuple[str, ...]:
    names = list(REQUIRED_GENERATOR_NAMES)
    if include_optional:
        names.append("random_shooting_aif")
    return tuple(names)
