#!/usr/bin/env bash
# Fase C (PLAN_SOLUCION.md): RWPE desde el head como PE estructural global del QK-stream.
# Contraste de DOS puntos de inyeccion + baseline (1 seed):
#   baseline  : sin PE.
#   pe_input  : RWPE sumado al INPUT del QK-RMPNN (riesgo de re-smoothing, como Fase A).
#   pe_post   : RWPE sumado tras el QK-RMPNN, antes de fc_to_qk (sobrevive al encoder).
# Hipotesis: pe_post preserva el rango (erank_K sube) y pe_input lo pierde (RMPNN lo
# re-smoothea). Senal global ortogonal al V-RMPNN local => si la atencion se vuelve
# load-bearing, es el caso de uso natural de un Graph Transformer.
# Inductivo WN18RR v2 (mismo regimen que el diagnostico). Single-device.
set -u
cd "$(dirname "$0")"
source env.sh

SEEDS=(42)
WALK_LEN=8
DATA=./data/inductive/wn18rr_v2
LOGDIR=logs/qkpe
CKPTROOT=experiments/qkpe
MANIFEST="$LOGDIR/manifest.csv"
mkdir -p "$LOGDIR" "$CKPTROOT"
echo "variant,seed,test_mrr,best_ckpt" > "$MANIFEST"

declare -A FLAGS=(
  [baseline]=""
  [pe_input]="--qk_pe input --qk_pe_walk_len $WALK_LEN"
  [pe_post]="--qk_pe post --qk_pe_walk_len $WALK_LEN"
)

run () {
  local variant=$1 seed=$2
  local flags=${FLAGS[$variant]}
  local log="$LOGDIR/${variant}_seed${seed}.log"
  echo "=== [$(date +%H:%M:%S)] variant=$variant seed=$seed flags='$flags' -> $log ==="
  python main.py \
    --seed "$seed" --accelerator gpu --precision 32 --devices 1 \
    --max_epochs 20 --checkpoint_save_path "$CKPTROOT/${variant}_seed${seed}" \
    --data_path "$DATA" \
    --batch_size 64 --test_batch_size 64 --num_workers 8 \
    --num_layer 3 --num_qk_layer 2 --num_v_layer 3 \
    --hidden_dim 32 --num_heads 4 \
    --loss_fn bce --adversarial_temperature 0.5 --num_negative_sample 8 \
    --learning_rate 5e-3 --optimizer Adam --weight_decay 1e-4 \
    $flags > "$log" 2>&1
  local mrr ckpt
  mrr=$(grep -aoE 'test_mrr[^0-9]*[0-9.]+' "$log" | grep -aoE '[0-9.]+$' | tail -1)
  ckpt=$(grep -aoE 'Loaded model weights from checkpoint at .*\.ckpt' "$log" | tail -1 | cut -d' ' -f7-)
  echo "    -> variant=$variant seed=$seed test_mrr=${mrr:-FAIL}"
  echo "${variant},${seed},${mrr:-FAIL},${ckpt:-NA}" >> "$MANIFEST"
}

for variant in baseline pe_input pe_post; do
  for seed in "${SEEDS[@]}"; do
    run "$variant" "$seed"
  done
done
echo "=== [$(date +%H:%M:%S)] DONE. Manifest: $MANIFEST ==="
