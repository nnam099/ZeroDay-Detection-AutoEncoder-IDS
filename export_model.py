from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ids_v14_unswnb15 import CFG, run_full  # noqa: E402


def build_config(args: argparse.Namespace) -> SimpleNamespace:
    config = {
        key: value
        for key, value in vars(CFG).items()
        if not key.startswith("_") and not callable(value)
    }
    config.update(
        {
            "data_dir": args.data_dir,
            "save_dir": args.save_dir,
            "plot_dir": args.plot_dir,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "demo": args.demo,
            "seed": args.seed,
        }
    )
    return SimpleNamespace(**config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/export IDS v14 artifacts for the dashboard.")
    parser.add_argument("--data_dir", default=os.path.join(ROOT_DIR, "data"))
    parser.add_argument("--save_dir", default=os.path.join(ROOT_DIR, "checkpoints"))
    parser.add_argument("--plot_dir", default=os.path.join(ROOT_DIR, "plots"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows to avoid multiprocessing issues.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Train on synthetic demo data instead of UNSW-NB15.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    if not args.demo and not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")

    config = build_config(args)
    print(f"Training IDS v14 artifacts from data_dir={config.data_dir}")
    run_full(config)
    print(f"Artifacts saved under: {config.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
