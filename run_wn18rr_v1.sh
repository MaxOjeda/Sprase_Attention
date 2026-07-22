#!/bin/bash
# Los 4 modelos en WN18RR ind v1, cada uno con su mejor config (de FB15k-237).
source env.sh
PY=$(which python)
DATA=./data/inductive/wn18rr_v1
COMMON="--data_path $DATA --max_epochs 20 --num_heads 8 --weight_decay 1e-4 \
  --batch_size 16 --test_batch_size 16 --seed 42 --num_workers 8 --num_layer 6"

echo "########## NBFNet ##########"
$PY train.py $COMMON --model nbfnet --aggregate pna --hidden_dim 32 --drop 0.1 \
  --learning_rate 5e-3 --checkpoint_save_path ./experiments/wn_nbf_v1 \
  > logs/wn_nbf_v1.log 2>&1

echo "########## Full attention (RFAT) ##########"
$PY train.py $COMMON --model rfat --hidden_dim 64 --drop 0.0 \
  --learning_rate 1e-3 --checkpoint_save_path ./experiments/wn_full_v1 \
  > logs/wn_full_v1.log 2>&1

echo "########## Sparse attention ##########"
$PY train.py $COMMON --model sparse --hidden_dim 64 --drop 0.0 \
  --learning_rate 1e-3 --checkpoint_save_path ./experiments/wn_sparse_v1 \
  > logs/wn_sparse_v1.log 2>&1

echo "########## Sparse_nbfv (V desde NBFNet) ##########"
$PY train.py $COMMON --model sparse_nbfv --aggregate pna --hidden_dim 64 --drop 0.0 \
  --learning_rate 1e-3 --checkpoint_save_path ./experiments/wn_sparse_nbfv_v1 \
  > logs/wn_sparse_nbfv_v1.log 2>&1

echo "########## WN18RR v1 DONE ##########"
