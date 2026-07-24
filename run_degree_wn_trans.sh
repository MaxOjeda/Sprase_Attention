#!/usr/bin/env bash
# sparse degree en WN18RR TRANSDUCTIVO — 1 GPU, dim32/L4, batch 32.
# Corre como job step superpuesto sobre la GPU del job 812436 (SIGSTOP'd PID 2773478
# retiene la asignacion). NO tocar ese proceso.
set -euo pipefail
cd /local_scratch/mojeda_imfd/Doctorado/Attention
source env.sh
PY=$(which python)
$PY train.py \
  --data_path ./data/wn18rr \
  --model sparse --attn degree \
  --num_layer 4 --hidden_dim 32 --num_heads 8 \
  --devices 1 \
  --batch_size 32 --test_batch_size 32 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path ./experiments/sparse_degree_wn_trans
