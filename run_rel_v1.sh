#!/bin/bash
# Ablation de transferencia (--attn rel): elimina el termino q.k del logit y deja la
# atencion puramente relacional (b[head,rel] + compatibilidad r_q x rel), transferible
# por construccion. Pregunta: ¿el overfit inductivo del sparse vive en q.k?
# FB15k-237 ind v1, best config sparse. Comparar vs sparse softmax (0.338) y NBFNet (0.459).
cd "$(dirname "$0")"
source env.sh
PY=$(which python)
$PY train.py --data_path ./data/inductive/fb15k-237_v1 --model sparse --attn rel \
  --num_layer 6 --hidden_dim 64 --num_heads 8 --batch_size 16 --test_batch_size 16 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path ./experiments/sparse_rel_v1
