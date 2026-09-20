#!/usr/bin/env python
"""Train the Step 4 paper generator checkpoints with a shared contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1_config import load_config
from training.train_generator import train_generator_model


PAPER_MODEL_CONFIGS = {
    "bc_mdn_aif": "configs/generator/full_bc_mdn.yaml",
    "cvae_aif": "configs/generator/full_cvae.yaml",
    "transformer_aif": "configs/generator/full_transformer.yaml",
    "diffusion_policy_aif": "configs/generator/full_diffusion_transformer.yaml",
    "flow_matching_aif": "configs/generator/full_flow_matching.yaml",
}


def _as_models(value: str | None) -> list[str]:
    if value is None:
        return list(PAPER_MODEL_CONFIGS)
    models = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [model for model in models if model not in PAPER_MODEL_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown paper generator model(s): {unknown}")
    return models


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    for key in (
        "output_dir",
        "dataset_root",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_train_windows",
        "max_val_windows",
        "max_failure_windows",
        "num_workers",
        "device",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg["failure_conditioned"] = True
    cfg.setdefault(
        "failure_sidecar_paths",
        "datasets/specialist_policy_approved_v1/runs/*/settings/*/failure_trajectories.npz",
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fresh Step 4 paper generator checkpoints.")
    parser.add_argument("--models", default=None, help="Comma-separated subset. Defaults to all paper models.")
    parser.add_argument("--output_dir", default="outputs/generators")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--max_failure_windows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifest", default="outputs/generators/paper_step4_training_manifest.json")
    parser.add_argument("--skip_completed", action="store_true")
    args = parser.parse_args()

    models = _as_models(args.models)
    manifest_path = Path(args.manifest)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_unix": time.time(),
        "models_requested": models,
        "results": {},
    }
    if args.skip_completed and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(previous.get("results"), dict):
            manifest["results"].update(previous["results"])
    _write_json(manifest_path, manifest)

    for model in models:
        current = manifest["results"].get(model, {})
        if args.skip_completed and current.get("status") == "complete":
            continue
        config_path = ROOT / PAPER_MODEL_CONFIGS[model]
        config = _apply_overrides(load_config(config_path), args)
        config["model"] = model
        started = time.time()
        manifest["results"][model] = {
            "status": "running",
            "config_path": str(config_path.relative_to(ROOT)),
            "started_at_unix": started,
        }
        _write_json(manifest_path, manifest)
        try:
            result = train_generator_model(config)
            manifest["results"][model] = {
                "status": "complete",
                "config_path": str(config_path.relative_to(ROOT)),
                "elapsed_sec": time.time() - started,
                "best_checkpoint_path": result["best_checkpoint_path"],
                "final_checkpoint_path": result["final_checkpoint_path"],
                "normalization_path": result["normalization_path"],
                "normalization_hash": result["normalization_hash"],
                "training_result_path": result["result_path"],
                "config_hash_path": result["config_hash_path"],
                "sample_diagnostics_path": result["sample_diagnostics_path"],
                "config_hash": result["config_hash"],
                "dataset": result["dataset"],
            }
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001 - persist partial status for detached runs.
            manifest["status"] = "failed"
            manifest["results"][model] = {
                "status": "failed",
                "config_path": str(config_path.relative_to(ROOT)),
                "elapsed_sec": time.time() - started,
                "error": repr(exc),
            }
            _write_json(manifest_path, manifest)
            raise
        _write_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["finished_at_unix"] = time.time()
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
