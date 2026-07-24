#!/usr/bin/env bash
# sparse degree en FB15k-237 TRANSDUCTIVO (la mejor variante en inductivo).
# Config = dim32/L4 para ser comparable con los otros experimentos transductivos
# (sparse softmax dim32/L4 = test 0.3965). apples-to-apples degree vs softmax.
# Lanzado como job step superpuesto sobre la GPU del job 812436 (SIGSTOP'd PID 2773478
# retiene la asignacion). NO tocar ese proceso.
set -euo pipefail
cd /local_scratch/mojeda_imfd/Doctorado/Attention
source env.sh
PY=$(which python)
$PY train.py \
  --data_path ./data/fb15k-237 \
  --model sparse --attn degree \
  --num_layer 4 --hidden_dim 32 --num_heads 8 \
  --batch_size 16 --test_batch_size 16 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path ./experiments/sparse_degree_trans_fb237
