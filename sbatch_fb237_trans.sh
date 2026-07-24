#!/bin/bash
# Baseline transductivo FB15k-237 (PLAN_MRR.md §7 paso 3) — 2 GPUs vía sbatch.
# La sesión interactiva tiene solo 1 GPU asignada (SLURM_GPUS_ON_NODE=1) y DDP rompe ahí
# (NCCL, [[nccl-ddp-broadcast-fails]]); este job pide 2 H100 del pool libre (IDX 5-7).
# Patrón canónico PL+SLURM: srun + ntasks-per-node=2 => 1 rank DDP por GPU (con ntasks=1 PL
# detecta SLURM y colapsa a world_size=1, corriendo en 1 sola GPU). --cpu-bind=none en srun
# evita "CPU binding outside of job step allocation" (las CPUs de las GPUs 5-7 están
# fragmentadas entre NUMA nodes y el bind por defecto falla).
# Recipe README FB15k-237 transductivo: hidden_dim 32, bce + adv_temp 0.5, --remove_all, neg 8.
# val=test ya arreglado (lightning.py) => model selection limpia.
#SBATCH --job-name=fb237_trans
#SBATCH --partition=compute-gpu-h100
#SBATCH --account=imfd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/autopsy/fb237_trans_%j.log

cd "$SLURM_SUBMIT_DIR"
source env.sh

# NCCL en este cluster falla detectando la NIC ("Attribute busid of node nic not found",
# [[nccl-ddp-broadcast-fails]]). Para 2 GPUs en el MISMO nodo NCCL debe usar NVLink/P2P y no
# la red => desactivar InfiniBand/NIC y forzar Socket evita el topology parse roto.
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# devices=2 + srun ntasks=2 => 1 rank por GPU. batch 96/GPU (efectivo 192; README usaba 4 GPUs
# => efectivo 384, lo anotamos: comparación de baseline, no idéntica).
srun --cpu-bind=none python main.py \
  --seed 42 --accelerator gpu --strategy ddp --precision 32 --devices 1 \
  --max_epochs 20 \
  --checkpoint_save_path ./experiments/autopsy/fb15k237_trans_seed42 \
  --data_path ./data/fb15k-237 \
  --batch_size 96 --test_batch_size 96 --num_workers 8 \
  --num_layer 3 --num_qk_layer 2 --num_v_layer 3 \
  --hidden_dim 32 --num_heads 4 \
  --loss_fn bce --adversarial_temperature 0.5 --remove_all --num_negative_sample 8 \
  --learning_rate 5e-3 --optimizer Adam --weight_decay 1e-4

echo "=== DONE fb237_trans job $SLURM_JOB_ID ==="
