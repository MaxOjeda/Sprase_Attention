#!/bin/bash
# Los 4 modelos en FB15k-237 ind v2, cada uno con su mejor config (misma que v1/wn18rr).
source env.sh
PY=$(which python)
DATA=./data/inductive/fb15k-237_v2
COMMON="--data_path $DATA --max_epochs 15 --num_heads 8 --weight_decay 1e-4 \
  --batch_size 16 --test_batch_size 16 --seed 42 --num_workers 8 --num_layer 6"

echo "########## NBFNet ##########"
$PY train.py $COMMON --model nbfnet --aggregate pna --hidden_dim 32 --drop 0.1 \
  --learning_rate 5e-3 --checkpoint_save_path ./experiments/fb_nbf_v2 \
  > logs/fb_nbf_v2.log 2>&1

echo "########## Full attention (RFAT) ##########"
$PY train.py $COMMON --model rfat --hidden_dim 64 --drop 0.0 \
  --learning_rate 1e-3 --checkpoint_save_path ./experiments/fb_full_v2 \
  > logs/fb_full_v2.log 2>&1

echo "########## Sparse attention ##########"
$PY train.py $COMMON --model sparse --hidden_dim 64 --drop 0.0 \
  --learning_rate 1e-3 --checkpoint_save_path ./experiments/fb_sparse_v2 \
  > logs/fb_sparse_v2.log 2>&1

echo "########## Sparse_nbfv (V desde NBFNet) ##########"
$PY train.py $COMMON --model sparse_nbfv --aggregate pna --hidden_dim 64 --drop 0.0 \
  --learning_rate 1e-3 --checkpoint_save_path ./experiments/fb_sparse_nbfv_v2 \
  > logs/fb_sparse_nbfv_v2.log 2>&1

echo "########## FB15k-237 v2 DONE ##########"
