#!/bin/bash
source env.sh
PY=$(which python)
DATA=./data/inductive/fb15k-237_v1
COMMON="--model sparse --data_path $DATA --max_epochs 20 --hidden_dim 64 --num_heads 8 --drop 0.0 --weight_decay 1e-4 --batch_size 16 --test_batch_size 16 --seed 42 --num_workers 8"
run () { name=$1; shift
  echo "########## $name ##########"
  $PY train.py $COMMON "$@" --checkpoint_save_path ./experiments/sparse_$name > logs/sparse_$name.log 2>&1
}
run A --num_layer 6 --learning_rate 5e-3
run B --num_layer 6 --learning_rate 1e-3
run C --num_layer 3 --learning_rate 5e-3
echo "########## SPARSE SWEEP DONE ##########"
