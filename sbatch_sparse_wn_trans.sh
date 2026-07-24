#!/bin/bash
# sparse (atencion softmax NORMAL, sin degree) en WN18RR TRANSDUCTIVO — 1 GPU, 24h,
# ENCADENABLE (auto-resume). Config dim32/L4, batch 32, single-GPU (sin DDP).
#   - Si NO existe last.ckpt => arranca de cero.  Si existe => reanuda.
# Uso:  sbatch sbatch_sparse_wn_trans.sh
#SBATCH --job-name=sparse_wn_trans
#SBATCH --partition=compute-gpu-h100
#SBATCH --account=imfd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sparse_wn_trans_%j.log

cd "$SLURM_SUBMIT_DIR"
source env.sh
mkdir -p logs experiments
PY=$(which python)

CKPT_DIR=./experiments/sparse_wn_trans
RESUME_ARG=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
  RESUME_ARG="--resume_from $CKPT_DIR/last.ckpt"
  echo "=== RESUME desde $CKPT_DIR/last.ckpt ==="
else
  echo "=== ARRANQUE DE CERO (no hay last.ckpt) ==="
fi

echo "=== sparse (softmax) WN18RR transductivo | job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

$PY train.py \
  --data_path ./data/wn18rr \
  --model sparse \
  --num_layer 4 --hidden_dim 32 --num_heads 8 \
  --devices 1 \
  --batch_size 32 --test_batch_size 32 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path "$CKPT_DIR" $RESUME_ARG
