#!/bin/bash
# Paquete QC-Exphormer (--attn qc) en FB15k-237 ind v1 (regimen inductivo).
# Prediccion: Q anclado a x0 ayuda (quita medio canal dependiente de estado) pero
# K = W_K(h) sigue leyendo el estado acumulado => no deberia cerrar el gap del todo.
# Comparar vs sparse softmax (0.338), sparse rel y NBFNet (0.459).
cd "$(dirname "$0")"
source env.sh
PY=$(which python)
$PY train.py --data_path ./data/inductive/fb15k-237_v1 --model sparse --attn qc \
  --num_layer 6 --hidden_dim 64 --num_heads 8 --batch_size 16 --test_batch_size 16 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path ./experiments/sparse_qc_v1
