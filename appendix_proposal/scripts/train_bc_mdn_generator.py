#!/usr/bin/env python
"""Train the Generator BC/MDN baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1_config import load_config
from training.train_bc_mdn import train_bc_mdn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/generator_bc_mdn.yaml")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--train_archive_root", default=None)
    parser.add_argument("--val_archive_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in (
        "dataset_root",
        "train_archive_root",
        "val_archive_root",
        "output_dir",
        "device",
        "epochs",
        "batch_size",
        "max_train_windows",
        "max_val_windows",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    result = train_bc_mdn(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
