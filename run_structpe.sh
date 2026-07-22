#!/usr/bin/env bash
# Structural Query-PE (PLAN_STRUCTURAL_PE.md L1+L2+L3, SESSION_LOG Resultado 10).
# Punto de reanudacion: el toolkit (compute_reltyped_rwpe/centrality/laplacian_pe,
# build_structural_pe, --qk_pe_readout, control 'noise') esta implementado y paso smoke,
# pero NUNCA se corrio. Esta matriz lo corre.
#
# Diagnostico que lo motiva (Resultado 9): el RWPE head-anclado relation-agnostic (Fase C)
# volvio la atencion load-bearing (Delta-0.075) pero NO subio MRR -> REDUNDANTE con el
# V-RMPNN (ambos codifican alcanzabilidad desde h). Para subir MRR el PE debe cargar senal
# ORTOGONAL al V-stream head-anclado: identidad intrinseca del candidato (deg/lap), no mas
# alcanzabilidad. Esta matriz aisla cada componente:
#
#   celda    | --qk_pe_kind | aisla
#   ---------+--------------+------------------------------------------------------------
#   baseline | (sin PE)     | referencia limpia (sin PE en atencion ni readout)
#   noise    | noise        | CONTROL DE CAPACIDAD: mismo presupuesto de params de fc_pe,
#            |              | contenido estructural nulo. Si full>noise => el contenido
#            |              | importa, no los params (gate §3.3, OBLIGATORIO para el paper).
#   rwpe     | rwpe         | Fase C re-corrida (redundante, head-anclado rel-agnostic)
#   rel      | rel          | L1: landing-prob desde h POR GRUPO DE RELACION (menos redundante)
#   deg      | deg          | L2 intrinseco: firma grado/relacion (centrality, Graphormer)
#   lap      | lap          | L2 intrinseco/global: top-k eigvecs del Laplaciano (rol global)
#   full     | rel,deg,lap  | L1+L2 ensamblado (apuesta principal de MRR)
#
# Todas las celdas con PE usan --qk_pe post (sobrevive al RMPNN, Resultado 9) Y
# --qk_pe_readout (L3: PE del candidato tambien al mlp_out). baseline sin ninguno.
# Recipe canonico (CLAUDE.md), wn18rr_v2 ind, 1 seed para el gate mecanico; tras pasar,
# relanzar con SEEDS=(42 43 44) para el gate de MRR (Delta>0.01 multi-seed).
# Single-device: NO --strategy ddp (NCCL roto en srun-interactiva, Resultado 5).
set -u
cd "$(dirname "$0")"
source env.sh

SEEDS=(42)
WALK_LEN=8
PE_GROUPS=8   # OJO: NO usar el nombre GROUPS (variable read-only reservada de bash =
              # array de group-IDs del usuario; $GROUPS se expande al GID, no a 8).
LAP_K=8
DATA=./data/inductive/wn18rr_v2
LOGDIR=logs/structpe
CKPTROOT=experiments/structpe
MANIFEST="$LOGDIR/manifest.csv"
mkdir -p "$LOGDIR" "$CKPTROOT"
[ -f "$MANIFEST" ] || echo "variant,seed,test_mrr,best_ckpt" > "$MANIFEST"

# PE comun a todas las celdas estructurales: post-RMPNN + readout.
PE_COMMON="--qk_pe post --qk_pe_readout --qk_pe_walk_len $WALK_LEN --qk_pe_groups $PE_GROUPS --qk_pe_lap_k $LAP_K"

# orden: baseline primero (referencia), luego noise (control), luego los componentes.
VARIANTS=(baseline noise rwpe rel deg lap full)
declare -A FLAGS=(
  [baseline]=""
  [noise]="$PE_COMMON --qk_pe_kind noise"
  [rwpe]="$PE_COMMON --qk_pe_kind rwpe"
  [rel]="$PE_COMMON --qk_pe_kind rel"
  [deg]="$PE_COMMON --qk_pe_kind deg"
  [lap]="$PE_COMMON --qk_pe_kind lap"
  [full]="$PE_COMMON --qk_pe_kind rel,deg,lap"
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

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run "$variant" "$seed"
  done
done

echo "=== [$(date +%H:%M:%S)] DONE. Manifest: $MANIFEST ==="
echo
echo "Siguiente (gates, PLAN_STRUCTURAL_PE.md §3) sobre el ckpt de cada celda:"
echo "  # (2) ortogonalidad: CKA(out_attn,out_vN) debe BAJAR en deg/lap/full vs rwpe"
echo "  python diag_rala_mechanism.py --ckpt <ckpt> --data_path $DATA   # seccion (D)"
echo "  # (4) load-bearing:"
echo "  python eval_noattn.py --ckpt <ckpt> --data_path $DATA            # con atn"
echo "  python eval_noattn.py --ckpt <ckpt> --data_path $DATA --no_attn  # sin atn"
echo "  # (1)(3) MRR/capacidad: comparar full vs noise vs baseline en $MANIFEST"
