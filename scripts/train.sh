#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-"data/"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/"}
PLOT_DIR=${PLOT_DIR:-"plots/"}

python train.py \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --plot_dir "$PLOT_DIR" \
  --epochs 100 \
  --batch_size 512 \
  --lr 3e-4 \
  --hidden 256 \
  --patience 20 \
  "$@"
