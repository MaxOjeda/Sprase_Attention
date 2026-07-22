# Setup obligatorio para el env knowformer. Uso: `source env.sh`
# Recreado a partir de la descripcion en CLAUDE.md (el archivo original faltaba).

# 1. Evitar que ~/.local/lib/python3.9/site-packages envenene el env.
export PYTHONNOUSERSITE=1

# 3. gcc 14 es demasiado nuevo para CUDA 12.6 -> usar gnu12/12.2.0.
module swap gnu14 gnu12/12.2.0 2>/dev/null || module load gnu12/12.2.0 2>/dev/null

# 4. CUDA 12.6.
module load cuda/12.6 2>/dev/null
export CUDA_HOME=/opt/ohpc/pub/cuda_home/cuda-12.6

# 2. python del env knowformer al frente del PATH.
export PATH=/nfs_ssd/mojeda_imfd/miniconda3/envs/knowformer/bin:$PATH
