#!/bin/bash
# Source-conditioned random-walk labeling (labeling trick nativo de transformer) en
# full attention. Best config (dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42), FB15k-237
# ind v1 y v2. Baselines full sin source_rw: v1 0.375, v2 0.491 (SESSION_NOTES).
set -e
source env.sh >/dev/null 2>&1
PY=$(which python)
mkdir -p logs experiments

COMMON="--model rfat --use_source_rw --source_rw_dim 8 \
  --num_layer 6 --hidden_dim 64 --num_heads 8 \
  --batch_size 16 --test_batch_size 16 --max_epochs 20 \
  --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42"

echo "=== source_rw full v1 ==="
$PY train.py --data_path ./data/inductive/fb15k-237_v1 $COMMON \
  --checkpoint_save_path ./experiments/source_rw_full_v1 2>&1 | tee logs/source_rw_full_v1.log

echo "=== source_rw full v2 ==="
$PY train.py --data_path ./data/inductive/fb15k-237_v2 $COMMON \
  --checkpoint_save_path ./experiments/source_rw_full_v2 2>&1 | tee logs/source_rw_full_v2.log

echo "=== DONE ==="
