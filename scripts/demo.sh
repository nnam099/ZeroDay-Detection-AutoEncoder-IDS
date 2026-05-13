#!/usr/bin/env bash
set -euo pipefail

python src/ids_v14_unswnb15.py --demo --save_dir checkpoints/ --plot_dir plots/ "$@"
