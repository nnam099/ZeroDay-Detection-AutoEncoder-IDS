#!/usr/bin/env bash
# IDS v15.0 — Training launcher
# Usage: bash scripts/train.sh [options]
set -e

DATA_DIR=${DATA_DIR:-"data/"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/"}
PLOT_DIR=${PLOT_DIR:-"plots/"}

python src/ids_v15_unswnb15.py \
    --data_dir   "$DATA_DIR"  \
    --save_dir   "$SAVE_DIR"  \
    --plot_dir   "$PLOT_DIR"  \
    --epochs     100          \
    --batch_size 512          \
    --lr         3e-4         \
    --hidden     256          \
    --ae_hidden  128          \
    --latent_dim 32           \
    --dos_weight 8.0          \
    --target_fpr 0.05         \
    "$@"
