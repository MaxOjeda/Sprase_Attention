#!/bin/bash
#SBATCH --job-name=rwse_v1
#SBATCH --partition=compute-gpu-h100
#SBATCH --account=imfd
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rwse_v1_sbatch_%j.log
# Recuperacion robusta a caidas del interactivo: corre las 3 corridas RWSE
# (full / sparse / sparse_nbfv) en FB15k-237 v1 con la best config de cada modelo.
cd /nfs_ssd/mojeda_imfd/Doctorado/Attention
srun ./run_rwse_v1.sh
