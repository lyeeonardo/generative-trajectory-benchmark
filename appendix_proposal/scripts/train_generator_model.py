#!/usr/bin/env python
"""Train any learned Generator proposal generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1_config import load_config
from training.train_generator import train_generator_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--debug_overfit", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--train_archive_root", default=None)
    parser.add_argument("--val_archive_root", default=None)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--failure_conditioned", action="store_true")
    parser.add_argument("--failure_sidecar_paths", default=None)
    parser.add_argument("--max_failure_windows", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}
    for key in (
        "model",
        "seed",
        "device",
        "resume",
        "num_workers",
        "output_dir",
        "dataset_root",
        "train_archive_root",
        "val_archive_root",
        "max_train_windows",
        "max_val_windows",
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
        "failure_sidecar_paths",
        "max_failure_windows",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.debug_overfit:
        config["debug_overfit"] = True
    if args.failure_conditioned:
        config["failure_conditioned"] = True
    if not config.get("model"):
        raise SystemExit("--model is required unless the config file sets model")
    result = train_generator_model(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
