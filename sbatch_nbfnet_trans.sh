#!/bin/bash
# NBFNet transductivo FB15k-237 — 1 GPU, 24h, ENCADENABLE (auto-resume) para llegar a 20 ep.
# Config alineada a NBFNet/config/knowledge_graph/fb15k237.yaml (L6 dim32 distmult pna
# short_cut layer_norm dependent, lr 5e-3) y a la best config validada del harness.
# ~2.5 h/epoca en H100 => ~6-7 epocas por job de 24h. Relanzalo hasta que alcance la epoca
# 20: al completar fit corre trainer.test automaticamente sobre el best-valid ckpt.
#   - Si NO existe last.ckpt => arranca de cero.
#   - Si existe last.ckpt    => reanuda desde la ultima epoca completada.
# Uso:  sbatch sbatch_nbfnet_trans.sh
#SBATCH --job-name=nbfnet_trans_fb237
#SBATCH --partition=compute-gpu-h100
#SBATCH --account=imfd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nbfnet_trans_fb237_%j.log

cd "$SLURM_SUBMIT_DIR"
source env.sh
mkdir -p logs experiments
PY=$(which python)

CKPT_DIR=./experiments/nbfnet_trans_fb237
RESUME_ARG=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
  RESUME_ARG="--resume_from $CKPT_DIR/last.ckpt"
  echo "=== RESUME desde $CKPT_DIR/last.ckpt ==="
else
  echo "=== ARRANQUE DE CERO (no hay last.ckpt) ==="
fi

echo "=== NBFNet FB15k-237 transductivo | job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY train.py \
  --data_path ./data/fb15k-237 \
  --model nbfnet --aggregate pna \
  --num_layer 6 --hidden_dim 32 --num_heads 8 \
  --batch_size 16 --test_batch_size 16 --max_epochs 20 \
  --learning_rate 5e-3 --weight_decay 1e-4 --drop 0.1 --seed 42 \
  --checkpoint_save_path "$CKPT_DIR" \
  $RESUME_ARG

echo "=== DONE job $SLURM_JOB_ID | $(date) ==="
