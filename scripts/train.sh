#!/bin/bash
python src/ids_v14_unswnb15.py \
  --data_dir data/ --save_dir checkpoints/ --plot_dir plots/ \
  --epochs 100 --batch_size 512 --lr 3e-4 --hidden 256 --patience 20
