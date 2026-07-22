#!/bin/bash
# Sparse attention (SparseGraphTransformer) sobre FB15k-237 TRANSDUCTIVO — 1 GPU, 24h,
# ENCADENABLE (auto-resume). Best config del sparse (SESSION_NOTES): hidden_dim 32,
# drop 0.0, lr 1e-3, L3, seed 42.
#
# Medido en H100 NVL (2026-07-09, smoke test batch 8): ~2.42 it/s, ~39.8 GB,
# 272,115 queries/época = 34,015 batches => ~3.9 h/época train + ~0.5 h val full-filtered.
# => en 24h se entrenan ~5 épocas. La meta es max_epochs=20 (estándar del proyecto).
#
# CÓMO ENCADENAR: el MISMO script sirve para el primer lanzamiento y para continuar.
#   - Si NO existe last.ckpt  => arranca de cero (época 0).
#   - Si existe last.ckpt     => reanuda desde la última época completada (estado del
#                                 optimizador + época restaurados por PL Trainer.fit).
# Cada época guarda 'last.ckpt' (estado completo) y el mejor 'valid_mrr' en
# experiments/sparse_trans_fb237/. Si SLURM corta el job por tiempo a mitad de época, se
# pierde solo esa época parcial; el próximo job reanuda desde el último last.ckpt.
# El test final SOLO corre en el job que alcanza la época 20.
#
# Uso:  sbatch sbatch_sparse_trans.sh   (relánzalo tantas veces como haga falta hasta 20 ep)
#
# NOTA batch: 39.8 GB a batch 8 (seguro en los 95 GB). batch 16 (~80 GB) casi duplicaría la
# velocidad pero con riesgo de OOM en la validación full-filtered. Se usa batch 8 (probado).
#SBATCH --job-name=sparse_trans_fb237
#SBATCH --partition=compute-gpu-h100
#SBATCH --account=imfd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sparse_trans_fb237_%j.log

cd "$SLURM_SUBMIT_DIR"
source env.sh

mkdir -p logs experiments
PY=$(which python)

CKPT_DIR=./experiments/sparse_trans_fb237_3_32
RESUME_ARG=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
  RESUME_ARG="--resume_from $CKPT_DIR/last.ckpt"
  echo "=== RESUME desde $CKPT_DIR/last.ckpt ==="
else
  echo "=== ARRANQUE DE CERO (no hay last.ckpt) ==="
fi

echo "=== sparse attention FB15k-237 transductivo | job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY train.py \
  --data_path ./data/fb15k-237 \
  --model sparse \
  --num_layer 3 --hidden_dim 32 --num_heads 8 \
  --batch_size 8 --test_batch_size 8 \
  --max_epochs 20 \
  --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path "$CKPT_DIR" \
  $RESUME_ARG

echo "=== DONE job $SLURM_JOB_ID | $(date) ==="
