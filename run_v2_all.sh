#!/bin/bash
set -e
source env.sh
PY=$(which python)
D=./data/inductive/fb15k-237_v2
COMMON="--data_path $D --num_layer 6 --num_heads 8 --batch_size 16 --test_batch_size 16 --max_epochs 20 --weight_decay 1e-4 --seed 42"

# 1) NBFNet (pna): dim 32, drop 0.1, lr 5e-3
$PY train.py $COMMON --model nbfnet --aggregate pna --hidden_dim 32 --drop 0.1 --learning_rate 5e-3 \
  --checkpoint_save_path ./experiments/v2_nbfnet 2>&1 | tee experiments/v2_nbfnet.log

# 2) RFAT full attention: dim 64, drop 0.0, lr 1e-3
$PY train.py $COMMON --model rfat --hidden_dim 64 --drop 0.0 --learning_rate 1e-3 \
  --checkpoint_save_path ./experiments/v2_rfat 2>&1 | tee experiments/v2_rfat.log

# 3) Sparse adyacencia: dim 64, drop 0.0, lr 1e-3
$PY train.py $COMMON --model sparse --hidden_dim 64 --drop 0.0 --learning_rate 1e-3 \
  --checkpoint_save_path ./experiments/v2_sparse 2>&1 | tee experiments/v2_sparse.log

# 4) Sparse NBF-value: dim 64, drop 0.0, lr 1e-3
$PY train.py $COMMON --model sparse_nbfv --hidden_dim 64 --drop 0.0 --learning_rate 1e-3 \
  --checkpoint_save_path ./experiments/v2_sparse_nbfv 2>&1 | tee experiments/v2_sparse_nbfv.log

echo "ALL_V2_DONE"
