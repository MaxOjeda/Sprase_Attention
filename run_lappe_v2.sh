#!/bin/bash
# LapPE (Laplacian positional encoding) en FB15k-237 ind v2 para los 3 modelos de atencion
# (best config: dim64/drop0/lr1e-3). Apples-to-apples vs baselines sin PE (v2): full 0.491,
# sparse 0.450, sparse_nbfv 0.527. Mismo protocolo que run_lappe_v1.sh (solo cambia el dataset).
set -e
source env.sh
PY=$(which python)
COMMON="--data_path ./data/inductive/fb15k-237_v2 --num_layer 6 --hidden_dim 64 \
  --num_heads 8 --batch_size 16 --test_batch_size 16 --max_epochs 20 \
  --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --use_lappe --lappe_dim 16"

echo "===== RFAT (full) + LapPE [v2] ====="
$PY train.py --model rfat        $COMMON --checkpoint_save_path ./experiments/lappe_full_v2   2>&1 | tee logs/lappe_full_v2.log

echo "===== Sparse adyacencia + LapPE [v2] ====="
$PY train.py --model sparse      $COMMON --checkpoint_save_path ./experiments/lappe_sparse_v2 2>&1 | tee logs/lappe_sparse_v2.log

echo "===== Sparse_nbfv (V=NBF) + LapPE [v2] ====="
$PY train.py --model sparse_nbfv $COMMON --checkpoint_save_path ./experiments/lappe_sparse_nbfv_v2 2>&1 | tee logs/lappe_sparse_nbfv_v2.log

echo "===== DONE ====="
