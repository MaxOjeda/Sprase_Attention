#!/bin/bash
# sparse degree en FB15k-237 TRANSDUCTIVO — 2 GPUs (DDP), 24h, ENCADENABLE (auto-resume).
# degree fue la mejor variante de atencion en inductivo; se prueba en transductivo.
# Config dim32/L4. batch 16/GPU x 2 GPUs = batch GLOBAL 32 (a peticion; NOTA: el baseline
# sparse softmax dim32/L4 = test 0.3965 usa batch 16 => esta corrida NO es apples-to-apples
# en batch, es un regimen de batch 2x).
# DDP: metricas MR/MRR/Hits usan dist_reduce_fx='sum' => agregan correcto entre ranks.
# ~0:45 h/epoca en 2xH100 => 20 epocas en ~15h (cabe en un job de 24h).
#   - Si NO existe last.ckpt => arranca de cero.  Si existe => reanuda.
# Uso:  sbatch sbatch_degree_trans.sh
#SBATCH --job-name=degree_trans_fb237
#SBATCH --partition=compute-gpu-h100
#SBATCH --account=imfd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/degree_trans_fb237_%j.log

cd "$SLURM_SUBMIT_DIR"
source env.sh
mkdir -p logs experiments
PY=$(which python)

CKPT_DIR=./experiments/sparse_degree_trans_fb237
RESUME_ARG=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
  RESUME_ARG="--resume_from $CKPT_DIR/last.ckpt"
  echo "=== RESUME desde $CKPT_DIR/last.ckpt ==="
else
  echo "=== ARRANQUE DE CERO (no hay last.ckpt) ==="
fi

echo "=== sparse degree FB15k-237 transductivo (2 GPU DDP) | job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# srun lanza 1 tarea por GPU; PL 1.9 detecta el entorno SLURM y arma DDP con --devices 2.
srun $PY train.py \
  --data_path ./data/fb15k-237 \
  --model sparse --attn degree \
  --num_layer 4 --hidden_dim 32 --num_heads 8 \
  --devices 2 \
  --batch_size 16 --test_batch_size 16 \
  --max_epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --drop 0.0 --seed 42 \
  --checkpoint_save_path "$CKPT_DIR" $RESUME_ARG
