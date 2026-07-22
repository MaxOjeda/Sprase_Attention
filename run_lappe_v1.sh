#!/bin/bash
# LapPE (Laplacian positional encoding) en FB15k-237 ind v1 para los 3 modelos de atencion
# (best config: dim64/drop0/lr1e-3). Apples-to-apples vs baselines sin PE: full 0.375,
# sparse 0.338, sparse_nbfv 0.421. Mismo protocolo que run_rwse_v1.sh (solo cambia el flag).
set -e
source env.sh
PY=$(which python)
COMMON="--data_path ./data/inductive/fb15k-237_v1 --num_layer 6 --hidden_dim 64 \
  --num_heads 8 --batch_size 16 --test_batch_size 16 --max_epochs 20 \
  --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --use_lappe --lappe_dim 16"

echo "===== RFAT (full) + LapPE ====="
$PY train.py --model rfat        $COMMON --checkpoint_save_path ./experiments/lappe_full_v1   2>&1 | tee logs/lappe_full_v1.log

echo "===== Sparse adyacencia + LapPE ====="
$PY train.py --model sparse      $COMMON --checkpoint_save_path ./experiments/lappe_sparse_v1 2>&1 | tee logs/lappe_sparse_v1.log

echo "===== Sparse_nbfv (V=NBF) + LapPE ====="
$PY train.py --model sparse_nbfv $COMMON --checkpoint_save_path ./experiments/lappe_sparse_nbfv_v1 2>&1 | tee logs/lappe_sparse_nbfv_v1.log

echo "===== DONE ====="
