#!/bin/bash
# Paquete QC-Exphormer (--attn qc): port limpio de la arquitectura del proyecto viejo
# (Exphormer_Max, run FB15k-237 trans 0.456 CON fuga en el loss). Aqui SIN fuga:
# el harness usa CE de grafo completo y --filtered_ce (si se activa) filtra SOLO train.
# Config alineada al run viejo: L5, dim 64, lr 2e-4 (AdamW-ish), drop 0.1.
# Comparar vs sparse softmax transductivo (0.4028 dim64/L6; 0.3965 dim32/L4) y NBFNet.
cd "$(dirname "$0")"
source env.sh
PY=$(which python)
$PY train.py --data_path ./data/fb15k-237 --model sparse --attn qc \
  --num_layer 5 --hidden_dim 64 --num_heads 4 --batch_size 16 --test_batch_size 16 \
  --max_epochs 20 --learning_rate 2e-4 --weight_decay 1e-4 --drop 0.1 --seed 42 \
  --checkpoint_save_path ./experiments/sparse_qc_trans_fb237
