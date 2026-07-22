#!/bin/bash
# Watchdog para el run manual de NBFNet transductivo (job 798757, srun --overlap).
# Espera hasta ~1.9h antes del wall de 24h, mata el entrenamiento y corre el test
# eval-only sobre el best-valid ckpt (save_top_k=1 => el unico epoch=*.ckpt del dir).
# Resultado en logs/nbfnet_trans_test.log. last.ckpt queda intacto para resume a 20 ep.
cd /nfs_ssd/mojeda_imfd/Doctorado/Attention
CKPT_DIR=experiments/nbfnet_trans_fb237
WLOG=logs/nbfnet_trans_watchdog.log
TLOG=logs/nbfnet_trans_test.log
{
  echo "=== watchdog start $(date) | sleep 59000s ==="
  sleep 59000
  echo "=== watchdog FIRED $(date) : matando entrenamiento NBFNet ==="
  pkill -u "$USER" -f 'train.py .*--model nbfnet' 2>/dev/null
  sleep 25
  BEST=$(ls -t $CKPT_DIR/epoch=*.ckpt 2>/dev/null | head -1)
  echo "best-valid ckpt: $BEST"
  if [ -z "$BEST" ]; then echo "NO HAY best ckpt; abortando test"; exit 1; fi
  source env.sh
  PY=$(which python)
  echo "=== eval-only test $(date) ==="
  $PY train.py \
    --data_path ./data/fb15k-237 \
    --model nbfnet --aggregate pna \
    --num_layer 6 --hidden_dim 32 --num_heads 8 \
    --batch_size 16 --test_batch_size 16 --max_epochs 20 \
    --learning_rate 5e-3 --weight_decay 1e-4 --drop 0.1 --seed 42 \
    --checkpoint_save_path $CKPT_DIR \
    --eval_only --eval_ckpt "$BEST" > "$TLOG" 2>&1
  echo "=== watchdog DONE $(date) ; test en $TLOG ==="
} >> "$WLOG" 2>&1
