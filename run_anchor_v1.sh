#!/bin/bash
# Opción C: regularizar la atencion hacia la agregacion fija (--attn anchor).
# FB15k-237 ind v1, best config sparse.
cd "$(dirname "$0")"
source env.sh
PY=$(which python)
$PY train.py --data_path ./data/inductive/fb15k-237_v1 --model sparse --attn anchor \
  --num_layer 6 --hidden_dim 64 --num_heads 8 --batch_size 16 --test_batch_size 16 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path ./experiments/sparse_anchor_v1
