# SESSION_LOG — Diagnóstico de bajo rango en la atención de KnowFormer

Bitácora de resultados del proyecto Knowformer-Expander, línea "diagnóstico de rango"
(`PLAN_RANK_DIAGNOSTIC.md`, base teórica `breaking_low_rank.tex` / RALA).
Conciso a propósito; el detalle de hipótesis vive en `PLAN_RANK_DIAGNOSTIC.md`.

---

## TL;DR

La atención lineal de KnowFormer **es rango-1** (patología P2 real), pero **no es el
cuello de botella**. Tres ángulos independientes convergen en que la vía de atención
aporta **~1 punto de MRR como techo**, al borde del ruido de seed. Cualquier fix tipo
RALA (α_j, φ(X)) sobre el rango de la atención topa en ese ~1 punto ⇒ **no vale la pena**.
La performance (0.69/0.70) la carga la vía de valor (`Σv + v·N`) + el V-RMPNN.

**Confirmado por el factorial completo (Resultado 5, 2026-06-05)**: 6 variantes × 3 seeds,
ninguna supera al baseline (Δ −0.003 a −0.009, ruido de seed); y el análisis de rango
muestra que las operaciones RALA son **mecánicamente inertes** a `d_head=8` (α_j deja el KV
buffer en srank≈1, φ no reemplaza el rango de v·num_node) ⇒ STOP definitivo.

**Blindado con d_head=32 + diagnóstico mecánico (Resultado 6, 2026-06-05)**: el null NO era
artefacto de `d_head=8`. Con d_head=32 (techo de rango 32) α_j **sigue siendo un no-op exacto**
(`Neff/N = 1.000`, α≡1) y srank(kvs) sigue en 1.00. Causa raíz medida: los feature maps K/Q
salen **over-smoothed** del RMPNN (DCfrac_K = 0.999 ⇒ los N nodos son el mismo vector). El bajo
rango está en los **inputs** de la atención, no en su operador ⇒ RALA arregla el eslabón
equivocado. El +0.010 de MRR de `rala` (1 seed) es **capacidad de fc_phi, no rango** ⇒ STOP
intrínseco, no dimensional.

**Cierre de la línea QK-input — Causa 5 confirmada por un ángulo nuevo (Resultados 8-9, 2026-06-10)**:
arreglar el *input* del QK-stream (ancla NBFNet = Fase A; RWPE global desde el head = Fase C) **sí
vuelve la atención load-bearing** — con RWPE post-RMPNN, anular la atención cuesta Δ−0.075 vs Δ−0.001
en baseline (gate >0.03 pasa por 2.5×). Pero el **MRR no se mueve** (+0.004, ruido) y `erank_K` apenas
sube (2.10 « 4): la atención cargó señal **redundante** con el V-RMPNN (la disociación load-bearing↑ /
rango≈ / MRR≈ es el hallazgo). El techo es del **régimen** (KGC local en WN18RR), no del operador ni
del rango ⇒ lo único no-redundante por diseño es señal **ortogonal al MP local** (atención por pares /
Edge Transformer).

**Cierre definitivo de la línea QK-input (Resultado 10, 2026-06-11)**: el Structural Query-PE
(rel-typed / centrality / Laplacian, L1+L2+L3) probó la señal **ortogonal intrínseca** que faltaba.
Resultado terminal: ninguna celda sube MRR, la intrínseca (deg/full) **daña** (−0.036/colapso) por
romper la inductividad (test graph disjunto), y —el remate— `deg` **sí des-smootheó K** (DCfrac
0.99→0.47, erank 1.4→3.5, la única vez del proyecto) y el MRR **empeoró**. Subir el rango de K
empeora el MRR ⇒ se acabó la hipótesis "des-smoothear el QK-stream ayuda". **No más parches al
QK-input.** El pivote real es de **régimen** (FB15k-237 / composicional) o de **arquitectura**
(atención por pares / Edge Transformer), no de qué se le entrega al encoder de K.

**Fase 1 redefinida — arranque positivo (Resultado 11, 2026-06-14)**: fix del leak val=test
aplicado + re-baseline limpio (wn18rr_v2 ind = 0.6898±0.0027). **Autopsia de errores
(`autopsy.py`)**: el MRR del baseline **colapsa monótonamente con d(h,t)** en ambos datasets
(d=2: 0.85-0.97 → d≥5/∞: ~0.17), pero FB15k-237 tiene 52% de masa a d≥4/inalcanzable vs 19% en
WN18RR. Los errores se concentran más allá del horizonte del MP (L=3) ⇒ **luz verde con datos a
CPA** (composición 2L hops). Es la Figura 1 del paper. Desarrollo = FB15k-237; sostener = WN18RR.

**CPA CERRADO + pivote a A1 (candidate-set attention), Gate 0 POSITIVO (Resultado 13, 2026-06-17)**:
CPA dado por finalizado (Resultado 12 + fix center, decisión usuario: no más variantes pivote-lineal).
Nuevo plan A1 = atención softmax sobre el set de candidatos top-K con bias por par (pointwise→listwise).
**Gate 0 (oracle) PASA por 2.5–5×** sin construir nada: headroom@32 = +0.125 (WN18RR ind) / +0.205
(FB15k-237 ind); oracle-MRR@K≈recall@K ⇒ toda la brecha es masa YA en el top-K pero mal ordenada.
Baseline FB15k-237 **transductivo** (job 684912) terminó: test_mrr=0.4311. Régimen de desarrollo listo.

---

## Setup común

- **Dataset principal**: WN18RR v2 inductivo (N_test=2.757, N_train graph=6.954).
- **Recipe** (idéntico al baseline reproducido, CLAUDE.md): `--num_layer 3 --num_qk_layer 2
  --num_v_layer 3 --hidden_dim 32 --num_heads 4 --loss_fn bce --adversarial_temperature 0.5
  --num_negative_sample 8 --lr 5e-3 --optimizer Adam --weight_decay 1e-4 --max_epochs 20
  --batch_size 64 --precision 32 --seed 42`.
- **Techo dimensional clave**: d_head = 32/4 = **8** ⇒ kvs es 8×8 y el feature map por
  cabeza es N×8 (rango tope 8, no 64 como en el paper RALA). Output mergeado N×32.
- Métricas de rango: `erank = exp(H(p))` con `p=σ/Σσ` (Roy & Vetterli 2007); `srank =
  ‖A‖²_F/σ₁²`. El `matrix_rank` numérico engaña; erank/srank miden diversidad real.

---

## Resultado 0 — Baseline reproducido

KnowFormer lineal (original), WN18RR v2 ind, seed 42, 20 epochs:
**test_mrr = 0.7008** (best ckpt epoch 19). Punto de comparación para todo lo demás.

Ckpt: `experiments/rank/wn18rr_v2_linear/.../epoch=19_step=4780.ckpt`.

---

## Resultado 1 — Análisis de rango (Fig. estilo RALA)

Hook `KNOWFORMER_RANK_DUMP` en `src/model.py:attn` captura kvs, attn_only (=q·kvs) y
output. Script `analyze_rank.py`. 1 batch de test (32 queries). Figuras en
`figs/rank_wn18rr_v2/spectra_linear.png` y `featuremap_linear.png`.

| capa | tensor | shape | matrix_rank | erank | srank | tope |
|---|---|---|---|---|---|---|
| 0 | KV buffer | 8×8 | 6.69 | 1.01 | 1.00 | 8 |
| 0 | attn-only q·kvs | N×8 | 2.89 | 1.01 | 1.00 | 8 |
| 0 | output (cabeza) | N×8 | 8.00 | 2.50 | 1.04 | 8 |
| 0 | output (full) | N×32 | 32.0 | 3.41 | 1.02 | 32 |
| 1 | KV buffer | 8×8 | 6.26 | 1.01 | 1.00 | 8 |
| 1 | attn-only | N×8 | 1.48 | 1.00 | 1.00 | 8 |
| 1 | output (cabeza) | N×8 | 8.00 | 3.60 | 1.11 | 8 |
| 1 | output (full) | N×32 | 32.0 | 6.06 | 1.11 | 32 |
| 2 | KV buffer | 8×8 | 7.85 | 1.09 | 1.00 | 8 |
| 2 | attn-only | N×8 | 3.88 | 1.06 | 1.00 | 8 |
| 2 | output (cabeza) | N×8 | 8.00 | 4.31 | 1.23 | 8 |
| 2 | output (full) | N×32 | 32.0 | 9.23 | 1.21 | 32 |

**Lectura inicial (luego corregida)**: kvs y attn-only son rango-1 (σ₂/σ₁≈0.01 en el
espectro), el output sube a rango pleno numérico vía los términos aditivos. Se postuló
"atención decorativa" → revisado en Resultados 2-3.

---

## Resultado 2 — Contribución y rango centrado (revisión crítica)

Hook ampliado: captura los 3 términos del numerador (q·kvs, Σv, v·N) cada uno **dividido
por el mismo denominador** ⇒ suman exactamente al output, son comparables en escala.
Script `analyze_rank_contrib.py`. 10 forward passes (el ruido gaussiano de `model.py:214`
sigue activo en eval), batch 32.

### (a) Contribución al output final

| capa | ‖attn‖/‖out‖ | ‖sumv‖/‖out‖ | ‖vN‖/‖out‖ | frac_attn | frac_sumv | frac_vN |
|---|---|---|---|---|---|---|
| 0 | 0.382 | 0.517 | 0.541 | +0.026 | 0.475 | 0.500 |
| 1 | 0.939 | 0.940 | 0.999 | **−0.883** | 0.884 | 0.999 |
| 2 | 0.747 | 0.803 | 0.908 | **−0.601** | 0.710 | 0.891 |

`frac_* = <term,out>/‖out‖²` (las 3 suman 1). **La atención NO es de norma pequeña**
(38–94 % de la del output), pero su contribución proyectada es ≈0 o **negativa**: el
término es grande pero **se cancela** contra los términos de valor.

### (b) Rango crudo vs centrado por filas (output y attn-only, N×32)

| capa | erank_out | erankC_out | srank_out | srankC_out | erank_at | erankC_at | srank_at | srankC_at |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.41 | 9.86 | 1.02 | 2.29 | 1.11 | 2.16 | 1.00 | 1.11 |
| 1 | 6.06 | 11.16 | 1.11 | 1.68 | 1.00 | 3.02 | 1.00 | 1.48 |
| 2 | 9.23 | 15.10 | 1.21 | 2.14 | 1.54 | 1.61 | 1.03 | 1.02 |

Al centrar (restar media sobre nodos), srank_out sube ⇒ **el bajo rango crudo del output
era en parte artefacto DC** (el `Σv` broadcast de `model.py:189` suma un vector idéntico a
todos los nodos). El **attn-only sigue siendo rango-1 incluso centrado** (srank 1.0–1.5):
esa parte sí aguanta.

---

## Resultado 3 — Ablación decisiva (eliminar la atención)

Toggle `KNOWFORMER_NO_ATTN_TERM` anula el término `q·kvs` del numerador (deja solo
`Σv + v·N`). Reentreno completo, mismo recipe, seed 42.

| Modelo | test_mrr | hits@10 | hits@3 | hits@1 | MR |
|---|---|---|---|---|---|
| Baseline (con q·kvs) | **0.7008** | — | — | — | — |
| Sin atención (q·kvs≡0) | **0.6898** | 0.770 | 0.723 | 0.642 | 185 |

**Δmrr = −0.011** (retiene el 98.4 % del MRR sin ninguna atención).

---

## Resultado 4 — De dónde viene el rango del output (ablación de `v·num_node`)

Toggle `KNOWFORMER_NO_VN_TERM` deja fuera el término self/skip `v·num_node` (equivale a
comentar `+ v*num_node` en `model.py:numerator`). **Ablación en inferencia** sobre el
checkpoint baseline (entrenado *con* el término) vía `analyze_rank.py`, mismo batch/setup
que el Resultado 1. Aísla cuánto del rango del output aportaba ese término. Figuras en
`figs/rank_wn18rr_v2_novn/`.

erank del output (full, N×32) con vs sin `v·num_node`:

| capa | baseline (con) | sin v·num_node |
|---|---|---|
| 0 | 3.41 | **1.07** |
| 1 | 6.06 | **1.74** |
| 2 | 9.23 | **1.60** |

(out por cabeza: erank 2.5/3.6/4.3 → **1.02/1.05/1.19**; srank → 1.00. kvs y attn-only
quedan idénticos: no dependen del término.)

**Lectura**: confirma de forma directa el desglose del Resultado 2. **`v·num_node` es el
término que carga toda la diversidad/rango del feature map de salida.** Al quitarlo, el
output queda con solo `q·kvs` (rango-1) + `Σv` (broadcast DC, rango-1) ⇒ **colapsa a
rango-1**, tan plano como la atención pura. La "salida full-rank" de KnowFormer en la figura
estilo RALA NO la produce la atención, sino la inyección directa del valor por nodo
`v·num_node` (análogo funcional al `φ(X_i)⊙` de RALA, ya presente en el diseño original).

Caveat: ablación en inferencia, no un modelo reentrenado sin el término (pendiente opcional
abajo).

---

## Síntesis: los tres ángulos convergen

| Intervención sobre la atención | Δmrr | n |
|---|---|---|
| Cambiar su *forma* (lineal→softmax, Fase 0) | −0.0097 (p>0.05) | 5 seeds (v1) |
| **Eliminarla por completo** (ablación) | **−0.0110** | 1 seed (v2) |

La vía de atención aporta **~1 punto de MRR**, y su forma interna (lineal/softmax/rango) es
**irrelevante** dentro de ese techo. La norma grande engaña: casi todo cancela o lo absorbe
el `attn_norm` (LayerNorm `model.py:230`); el residuo neto al score es ~1 punto.

**Veredicto**: ni "decorativa = cero" ni "patología que conviene arreglar". La atención es
rango-1, de norma grande pero cancelatoria, y su aporte neto (~1 pt MRR) está al borde del
ruido de seed. **Techo cuantificado para cualquier fix RALA = ~1 punto** ⇒ confirma el STOP
del plan, ahora con un número y no solo un gate nulo.

---

## Pendiente (opcional, para blindar)

- Ablación no-attn con 4-5 seeds: el Δ=0.011 de 1 seed podría ser ruido (Fase 0 con 5 seeds
  dio p>0.05 para un efecto del mismo tamaño). Único paso que faltaría para cerrar con
  significancia estadística.
- Réplica fiel del Resultado 4: reentrenar v2 con `KNOWFORMER_NO_VN_TERM=1` y regenerar las
  figs (vs la ablación en inferencia actual), para confirmar el colapso de rango en un modelo
  entrenado sin el término.

## Resultado 5 — Ablation factorial RALA (COMPLETO, 2026-06-05)

**Estado**: 18 runs terminados (6 variantes × 3 seeds 42,43,44). El factorial **y** el
análisis de rango convergen con el resto del log: el fix RALA **no compra MRR y, peor, es
mecánicamente inerte** a `d_head=8`. Cierra definitivamente el STOP del plan.

### Qué se implementó
RALA (`breaking_low_rank.tex`) como **toggles opcionales, off por defecto**, cableados
CLI→Lightning→modelo (`main.py`, `lightning.py`, `src/model.py`):
- `--rala_kv` (mod 1): KV buffer ponderado con α_j = N·softmax(Q_g·K_j), Q_g=mean_v Q.
  Parameter-free. Reweighta K in place ⇒ afecta kvs y denominador. En `attn()`.
- `--rala_phi {none,x,v}` (mod 2): modulación Hadamard φ(X)⊙Y en `forward()`. `v`=stream
  V-RMPNN de la capa, `x`=input acumulado. `fc_phi` init constante 1 (no-op al inicio).
- Detalle clave: φ se aplica al output de `attn()` que YA contiene `v·num_node`. O sea
  φ y v·num_node coexisten salvo que se quite v·num_node con `KNOWFORMER_NO_VN_TERM=1`.

### Setup
WN18RR v2 ind, recipe idéntico al baseline (sección "Setup común"), **single-device (sin
--strategy ddp)**: NCCL roto en sesión srun-interactiva; con 1 GPU equivalente. Baseline
fresco aquí, NO las cifras ddp históricas. ~18 min/run. `run_rala.sh` corrió 10/18 y se
cortó; los 8 faltantes se relanzaron a mano con `run_rala_missing.sh` (appendea al
manifest, no sobreescribe). Resultados en `logs/rala/manifest.csv`.

| variante | flags | env (v·num_node) |
|---|---|---|
| baseline | — | ✓ |
| alpha | `--rala_kv` | ✓ |
| phi | `--rala_phi v` | ✓ (φ apilado encima) |
| rala | `--rala_kv --rala_phi v` | ✓ |
| phi_replace | `--rala_phi v` | ✗ `NO_VN_TERM=1` (φ reemplaza, fiel a eq.6) |
| rala_replace | `--rala_kv --rala_phi v` | ✗ `NO_VN_TERM=1` |

### (a) MRR — ninguna variante supera al baseline

| variante | seed42 | seed43 | seed44 | media | Δ vs baseline |
|---|---|---|---|---|---|
| **baseline** | 0.7050 | 0.7022 | 0.7041 | **0.7038** | — |
| phi_replace | 0.7016 | 0.6975 | 0.7042 | 0.7011 | −0.0027 |
| alpha | 0.6971 | 0.7073 | 0.6900 | 0.6981 | −0.0056 |
| rala_replace | 0.6915 | 0.7036 | 0.6959 | 0.6970 | −0.0068 |
| rala | 0.6905 | 0.6971 | 0.7002 | 0.6959 | −0.0079 |
| phi | 0.6941 | 0.6987 | 0.6912 | 0.6947 | −0.0091 |

Todas caen −0.003 a −0.009 MRR, dentro del ruido de seed. Apilar φ sobre v·num_node
(phi, rala) es lo peor; `phi_replace` (φ reemplazando v·N, fiel a eq.6) es la mejor de las
seis y prácticamente empata al baseline → φ y v·num_node son análogos intercambiables
(confirma Resultado 4 por MRR).

### (b) Análisis de rango (`analyze_rala.sh`, seed 42) — el mecanismo es inerte

**α_j NO sube el stable rank del KV buffer.** srank(kvs) por capa:
| variante | c0 | c1 | c2 |
|---|---|---|---|
| baseline | 1.00 | 1.00 | 1.00 |
| alpha | 1.01 | 1.00 | 1.00 |

Sigue rango-1. erank sube marginal y no-monótono (1.03→1.26 solo en c0). En el paper
(visión, d=64) α subía el rango sustancialmente; con `d_head=8` y la estructura por-relación
de KnowFormer (cada fila del batch ya es una query/relación ⇒ Q_g=mean_v Q tiene poca
diversidad que repartir) el softmax-reweighting apenas lo toca.

**attn_only (q·kvs) es rango-1 estructural** en todas las variantes (srank 1.00–1.14),
incluso en las `*_replace` donde el modelo, forzado a usar la atención al quitar v·N, sube
su matrix_rank de 2–3 a 6–7 pero **el stable rank no pasa de ~1.1**.

**φ NO reemplaza el rango de v·num_node.** erank del output full (N×32):
| variante | c0 | c1 | c2 |
|---|---|---|---|
| baseline (con v·N) | 5.34 | 6.63 | 8.27 |
| phi_replace (φ en vez de v·N) | 1.54 | 1.34 | 1.54 |
| rala_replace | 1.43 | 1.51 | 1.43 |

Al quitar v·num_node y poner φ, **el output colapsa a rango-1** (srank ~1.0). φ —gate de
canal por token— no regenera el rango que daba v·num_node. Confirmación directa del
Resultado 4 por rango.

### Remate
A pesar de ese colapso de rango, **phi_replace iguala al baseline en MRR (0.7011 vs
0.7038)**: el feature map de atención puede caer a rango-1 sin tocar el MRR porque el stream
residual + V-RMPNN llevan la señal. Es la formulación más fuerte de la tesis: **el rango de
la rama de atención es casi irrelevante para el MRR de KnowFormer.** Las dos operaciones de
RALA ni siquiera logran su objetivo de aumentar el rango en esta arquitectura ⇒ negativo
más fuerte que "el MRR no se movió": el fix es **inerte mecánicamente**, no solo neutro.

Figs por variante en `figs/rank_wn18rr_v2_<variant>/`; tablas en `logs/rala/rank_<variant>.log`
y `contrib_<variant>.log`.

---

## Resultado 6 — Factorial RALA d_head=32 + diagnóstico mecánico (COMPLETO, 2026-06-05)

**Motivación**: el factorial d_head=8 (Resultado 5) salió negativo y se atribuyó a que
`d_head=8` es demasiado chico (RALA fue diseñado para d=64). Con **d_head=32 (un solo head,
`--num_heads 1 --hidden_dim 32`)** el techo de rango del KV buffer es 32, no 8 ⇒ α_j tendría
margen real para subir el rango. Es el test que decide si el null de RALA es **intrínseco o
artefacto de dimensión**. 1 seed (42), 6 variantes, resto idéntico al baseline. Single-device.
Dirs separados `*_dhead32`. Scripts: `run_rala_dhead32.sh` (+ `run_rala_dhead32_missing.sh`
para los 3 que faltaron al cortarse la sesión), análisis `analyze_rala_dhead32.sh`.

### (a) MRR — 5 de 6 variantes SUBEN (al revés que d_head=8)

| variante | flags | test_mrr | Δ vs baseline d32 |
|---|---|---|---|
| baseline | — | 0.6946 | — |
| alpha | `--rala_kv` | 0.6965 | +0.0019 |
| phi | `--rala_phi v` | 0.6983 | +0.0037 |
| **rala** | `--rala_kv --rala_phi v` | **0.7046** | **+0.0100** |
| phi_replace | `--rala_phi v` + NO_VN | 0.6985 | +0.0039 |
| rala_replace | `--rala_kv --rala_phi v` + NO_VN | 0.6941 | −0.0005 |

Contraste con d_head=8 (todas caían). `rala` (α+φ) da +0.0100, el mayor efecto del proyecto, y
parece superaditivo (singles suman +0.0056). PERO: 1 seed; la Fase 0 mostró que Δ~0.01 da p>0.05
sobre 5 seeds ⇒ **no concluir sin réplica multi-seed**. `rala_replace` (φ reemplaza v·N) cae al
baseline ⇒ el boost necesita que φ se **apile** sobre v·N (capacidad), no que lo reemplace.

### (b) Análisis de rango (`analyze_rala_dhead32.sh`) — la atención SIGUE inerte

**srank(kvs) sigue clavado en 1.00 con α_j**, exactamente como a d_head=8:
| variante | c0 | c1 | c2 |  | erank(kvs) c0/c1/c2 |
|---|---|---|---|---|---|
| baseline | 1.00 | 1.00 | 1.00 | | 1.01 / 1.06 / 1.18 |
| alpha | 1.00 | 1.00 | 1.00 | | 1.01 / 1.03 / 1.18 (≤ baseline) |

`attn_only` srank 1.00 en todo. El output (erank 5.8/6.1/7.3) lo sigue cargando `v·num_node`:
quitarlo (`phi_replace`/`rala_replace`) colapsa el output a erank ~1-5 ⇒ re-confirma Resultado 4
a d_head=32. **Subir d_head de 8→32 no tocó nada.**

### (c) Diagnóstico mecánico — POR QUÉ es inerte (`diag_rala_mechanism.py`)

Hook ampliado captura `q_norm`, `k` post-norm pre-reweight, y `alpha`. Sobre baseline vs alpha:

| | logit_std (Q_g·K_j) | ‖Q_g‖ | **Neff(α)/N** | srank_K | **DCfrac_K** |
|---|---|---|---|---|---|
| alpha c0 | 0.001 | 0.999 | **1.000** | 1.00 | 0.999 |
| alpha c1 | 0.000 | 1.000 | **1.000** | 1.00 | 1.000 |
| alpha c2 | 0.064 | 0.987 | 0.997 | 1.04 | 0.951 |

Tres mecánicas convergentes, todas derivadas de **over-smoothing del RMPNN**:
1. **α_j es un no-op exacto** (`Neff/N=1.000`, α∈[0.999,1.006]). Causa próxima: `logit_std≈0.001`
   ⇒ los logits `Q_g·Kⱼ` son idénticos entre tokens ⇒ softmax plano sobre N=2757.
2. **K/Q son token-colineales** (`DCfrac_K=0.999`: 99.9% de la energía de K es su media sobre
   nodos; `srank_K=1.00`; `‖Q_g‖≈1` ⇒ q's casi idénticos). Los N nodos son **el mismo vector**
   ⇒ `B = Kᵀ diag(α) V` es rango-1 para **cualquier** α ⇒ reweightar tokens colineales no sube
   rango. Esto es over-smoothing de message-passing (Li 2018; Oono & Suzuki 2020).
3. **Σαⱼ=N congela el modo DC dominante** (σ₁ de B ∝ el outer product DC, invariante a α).

**Tesis**: RALA asume que el bajo rango está en el *operador* de atención (tokens diversos, la
linearización los colapsa). En KnowFormer el bajo rango está en los **inputs** (node features
ya colapsados, *aguas arriba* de la atención). RALA arregla el eslabón equivocado ⇒ inerte por
construcción. Contraste con visión: parches diversos (DCfrac moderado) + κ=`elu+1` sin normalizar
con escala (logits con rango) ⇒ allá α sí es peaked y sube el rango. KnowFormer hace
`F.normalize(q,k)` (logits ∈[-1,1] sin temperatura) sobre features ya over-smoothed ⇒ α≡1.

**Veredicto**: el null es **intrínseco**, no dimensional. El +0.010 de `rala` es capacidad de
`fc_phi`, ortogonal a la tesis del rango (srank no se movió). STOP blindado. El único punto de
ataque real sería des-smoothear el front-end (PairNorm, jumping-knowledge, menos capas RMPNN
antes de la atención), pero eso ya NO es RALA y choca con la lista negra (no tocar los RMPNN).

**Pendiente opcional**: réplicas multi-seed (baseline+rala, seeds 43/44) para confirmar si el
+0.010 es ruido de seed o un efecto real de capacidad. No cambia el veredicto del mecanismo.

## Resultado 7 — QK-stream vs V-stream: ambos over-smoothed (2026-06-05)

**Motivación**: el Resultado 6c midió `DCfrac_K=0.999` sobre K (stream QK), y se conjeturó
que el V-stream estaría *menos* colapsado por recibir el one-hot del source (`v_x[h_index]=1`),
mientras que el QK-stream solo recibe `x`+ruido gaussiano. Se midió para confirmar/refutar.

`diag_rala_mechanism.py` extendido para capturar `v` en el hook y medir K vs V lado a lado
(filas row-normalized para comparar dirección). Baseline d_head=32, wn18rr_v2, ckpt
`experiments/rala_dhead32/baseline_seed42/...final.ckpt`:

| capa | srank_K | erank_K | DCfrac_K | srank_V | erank_V | DCfrac_V |
|---|---|---|---|---|---|---|
| 0 | 1.06 | 2.30 | 0.712 | 1.11 | 5.69 | 0.890 |
| 1 | 1.00 | 1.34 | 0.999 | 1.01 | 2.68 | 0.988 |
| 2 | 1.04 | 2.07 | 0.956 | 1.21 | 8.41 | 0.823 |

**Corrige el matiz a medias**: NO es "QK roto, V sano". Por **srank ambos empatan en ~1**.
La única ventaja de V es **erank** (diversidad direccional de la cola): 2–4× la de K en cada
capa — coherente con el one-hot del source. Pero el one-hot **no rescata el modo dominante**
de V (srank~1); solo enriquece la cola. Conexión clave: `erank_V` (5.7/2.7/8.4) ≈ `erank` del
output (5.8/6.1/7.3, Resultado 6) ⇒ el output hereda su diversidad de V vía `v·num_node`, NO
del operador de atención. Confirma Resultado 4/6 por un ángulo nuevo. **Refuerza el STOP**: la
atención no tiene de dónde sacar rango en NINGUNO de los dos streams, no solo en el QK.

---

## Resultado 8 — Fase A: labeling trick (one-hot del head al QK-stream) — NEGATIVO MECÁNICO (2026-06-09)

**Motivación** (PLAN_SOLUCION.md, pivote del 2026-06-09): el QK-stream arranca con `x + ruido
gaussiano` (`model.py:295`) — **sin anclaje a la query**, a diferencia del V-stream
(`v_x[h_index]=1`). En la capa externa 0, `x=0` ⇒ el QK-encoder ve *solo ruido* ⇒ converge a un
vector DC (chicken-and-egg de query-conditioning). Fase A inyecta el one-hot del head al **input**
del QK-stream, espejo exacto del V-stream. Toggle `--qk_anchor_head` (off=baseline; `fc_qk_x`
in_features `hidden_dim+1`→`hidden_dim*2`). Receta canónica (num_heads 4, hidden_dim 32), wn18rr_v2
ind, 1 seed (42). Scripts `run_qkinput.sh`, `logs/qkinput/`, `experiments/qkinput/`.

### (a) MRR — sube +0.0097 pero dentro del ruido del propio baseline
| variante | test_mrr | hits@1 | hits@3 | hits@10 |
|---|---|---|---|---|
| baseline | 0.6895 | 0.645 | 0.720 | 0.764 |
| **anchor** | **0.6992** | 0.650 | 0.737 | 0.777 |

Δ=+0.0097, todas las métricas suben consistentes. PERO: este baseline (0.6895) cayó ~1.4pt bajo
los baselines single-device previos (0.7038 medio, Resultado 5) ⇒ **la varianza entre baselines
idénticos ya es mayor que la ganancia**. Piso de ruido (Fase 0: Δ~0.01 da p>0.05).

### (b) Gate mecánico (`diag_rala_mechanism.py` sobre ambos ckpts) — FALLA
| capa | erank_K base | erank_K **anchor** | DCfrac_K base | DCfrac_K **anchor** |
|---|---|---|---|---|
| 0 | 1.06 | 1.44 | 1.000 | 0.996 |
| 1 | 1.28 | 1.38 | 0.952 | 0.996 |
| 2 | 1.33 | **2.01** | 0.986 | 0.938 |

`srank_K`≈1 en ambos. El ancla **mueve erank_K en la dirección correcta** (c2: 1.33→2.01) pero
**no pasa el gate** (`erank_K>4` en ≥2 capas; máx 2.01). `DCfrac_K` sigue ~0.94–0.996: K continúa
DC-dominado. El erank_K del anchor (1.4–2.0) sigue **por debajo del erank_V que ya tenía el
baseline** (2.6–4.2).

**Veredicto**: el modo de falla **anticipado** por PLAN_SOLUCION.md — "necesario pero no
suficiente": el ancla rompe la simetría en el *input*, pero las 2 capas del QK-RMPNN la
**re-smoothean** (over-smoothing, Oono & Suzuki). No es la degeneración a V (cos K≈V); es el
encoder re-colapsando. El +0.0097 es capacidad de `fc_qk_x` (input más ancho), NO la atención
volviéndose load-bearing — mismo patrón del falso positivo de `rala` d_head=32 (+0.010 = capacidad).
**Lección para Fase C+**: inyectar señal al *input* del QK-RMPNN no sobrevive; hay que inyectarla
**post-RMPNN** (directo en Q/K) para que no se re-smoothee.

---

## Resultado 9 — Fase C: RWPE desde el head — COMPLETO: atención load-bearing SIN subir MRR (2026-06-10)

**Pivote sobre Fase B**: Fase B (embeddings pretrained ComplEx/RotatE) **descartada** — exige
entidades compartidas ⇒ fuerza transductivo, rompe el régimen inductivo del diagnóstico y el
requisito del manuscrito (una arquitectura trans+ind). Fase C la reemplaza: **se mantiene en
inductivo** (señal estructural por-grafo desde el head, sin lookup por entidad).

**Intervención**: RWPE desde el head (`compute_rwpe` en `model.py`) — para cada query y nodo v,
vector de probabilidades de aterrizaje `P^k e_h` a k=1..L pasos (esperanza exacta de "contar
visitas de L random walks", determinista vía L matvecs sparse; relation-agnostic). Anclado a h
como el V-stream **pero global** (ve paths largos/ciclos que el RMPNN de 3 capas no alcanza).
Toggles `--qk_pe {none,input,post}`, `--qk_pe_walk_len` (8); proyección `fc_pe` aditiva, init-a-0
(baseline anidado). `input`=al input del RMPNN (riesgo re-smoothing, Resultado 8); `post`=tras el
RMPNN antes de `fc_to_qk` (sobrevive). El run original (`run_qkpe.sh`) murió tras `baseline`;
`pe_input`+`pe_post` se relanzaron con `run_qkpe_rest.sh` (appendea al manifest). 1 seed (42),
wn18rr_v2 ind. `logs/qkpe/`, `experiments/qkpe/`.

### (a) MRR — plano (gate de éxito FALLA)
| variante | test_mrr | Δ vs baseline |
|---|---|---|
| baseline | 0.7013 | — |
| pe_input | 0.7041 | +0.0028 |
| **pe_post** | 0.7057 | +0.0044 |

Orden correcto según hipótesis (post > input > base), magnitudes en piso de ruido (Fase 0: p>0.05).

### (b) Gate mecánico `erank_K` (`diag_rala_mechanism.py`) — FALLA (como Fase A)
| capa | erank_K base | erank_K pe_input | erank_K **pe_post** | DCfrac_K base | DCfrac_K **pe_post** |
|---|---|---|---|---|---|
| 0 | 1.15 | 1.24 | **2.10** | 1.000 | **0.758** |
| 1 | 1.58 | 1.20 | 1.22 | 0.965 | 0.998 |
| 2 | 1.10 | 1.72 | 1.71 | 1.000 | 0.973 |

`pe_post` es el mejor variante de toda la línea QK-input (capa 0: DCfrac 1.0→0.758, erank
1.15→2.10, la única vez que K se des-smoothea de verdad). Pero máx erank_K=2.10 « gate >4, y solo
en 1 capa; capas 1-2 re-colapsan ⇒ el QK-RMPNN re-smoothea (Oono & Suzuki), mismo modo de falla
que Fase A pese a inyectar **post**-encoder.

### (c) Gate load-bearing (`eval_noattn.py`, ablación-en-inferencia `NO_ATTN_TERM=1`) — PASA 2.5×
| variante | con atn | sin atn (inf) | **Δ load-bearing** |
|---|---|---|---|
| baseline | 0.7013 | 0.6999 | **0.0014** |
| pe_input | 0.7036 | 0.6531 | **0.0505** |
| **pe_post** | 0.7071 | 0.6317 | **0.0754** |

Comparación apples-to-apples (ablación-en-inferencia en todos; NO el reentreno del Resultado 3).
En baseline, anular la atención en inferencia **no cuesta nada** (Δ=0.0014) ⇒ el modelo no enruta
señal por la atención. Con RWPE la atención pasa a cargar **5-7.5 pt** de MRR ⇒ **gate >0.03 pasa
por 2.5×**. *Matiz*: el RWPE entra al output **solo** por la vía de atención (K post-RMPNN →
`fc_to_qk` → q·kvs), así que anular la atención borra todo el PE ⇒ el operador de atención es ahora
el **conducto** load-bearing de la señal RWPE.

### Veredicto — la disociación es el hallazgo; confirma Causa 5
**Las dos métricas se separan**: `erank_K` NO subió (máx 2.10) pero la atención SÍ se volvió
load-bearing (Δ=0.075). Refuta `erank_K` como proxy y confirma la hipótesis secundaria del plan
**"rango ≠ información útil"** en su forma inversa: lo que volvió la atención load-bearing no fue el
rango de K sino que K por fin lleva **información query-conditioned** (probabilidades de aterrizaje
del head). Pero el **MRR neto es plano** (+0.004): pe_post *sin* su atención (0.6317) cae *por
debajo* del baseline sin atención (0.6999) ⇒ el modelo **redistribuyó** trabajo del V-stream a la
atención, no **añadió** capacidad. La señal RWPE (alcanzabilidad global desde h) es **redundante**
con lo que el V-RMPNN ya propaga localmente. Es el outcome #2 que PLAN_SOLUCION.md anticipó como
**publicable**: *"fixing the inputs is necessary but insufficient — KGC reasoning is local enough
that attention rank doesn't translate to MRR in WN18RR"*, con el refinamiento de que **ni siquiera
hizo falta subir el rango**: bastó info query-conditioned para que la atención cargara 7.5 pt, y aun
así el techo del régimen no se movió por redundancia con el V-stream.

**Cierra la línea QK-input (Fase A + C).** Lo único que por diseño NO sería redundante con el
V-RMPNN es señal **ortogonal al MP local** ⇒ pivote natural: atención por pares / Edge Transformer
(dirección futura, abajo), no más parches al input del QK-stream.

**Artefactos**: `run_qkpe_rest.sh` (relanza pe_input+pe_post, appendea al manifest),
`eval_noattn.py` (test-only con/sin `NO_ATTN_TERM`). `logs/qkpe/manifest.csv`,
`experiments/qkpe/{baseline,pe_input,pe_post}_seed42/`.

---

## Resultado 10 — Structural Query-PE (L1+L2+L3): COMPLETO — NEGATIVO TERMINAL (2026-06-11)

**Estado**: matriz de 7 celdas corrida (1 seed, wn18rr_v2 ind; `full` cortado tras confirmar
colapso, valid_mrr=0.444). **Ninguna celda estructural supera baseline; las intrínsecas (deg/full)
DAÑAN.** Y el remate mecánico: `deg` **sí des-smootheó K** (la mejor del proyecto) y el MRR
**empeoró** ⇒ confirmación terminal de que el rango de K no es el cuello de botella. Cierra la
línea de "arreglar los inputs del QK-stream". Plan: `PLAN_STRUCTURAL_PE.md`.

### Veredicto (TL;DR del resultado)
1. **MRR**: gate `Δ>0.01` FALLA en todas. Único positivo `rwpe` +0.005 (piso, = Resultado 9).
   `deg` −0.036 y `full` colapsa ⇒ la señal estructural **intrínseca** (centrality/lap) es
   contraproducente en inductivo (graph-específica, no transfiere al test graph disjunto).
2. **Des-smoothing ≠ MRR (remate)**: `deg` capa 2 logró `DCfrac_K 0.47 / erank_K 3.50`
   (de ~1.0/1.4; la única des-smoothing real del proyecto) y aun así MRR −0.036.
   Subir el rango de K **empeora** el MRR ⇒ el rango nunca fue el cuello de botella.
3. **Gate CKA mal planteado**: `CKA(out_attn,out_sumv)=1.000` en TODAS las celdas/capas ⇒ la
   vía de atención sigue rango-1/DC pase lo que pase; `CKA(out_attn,out_vN)` bajo en todas
   (0.10–0.25) por diferencia de rango, no por ortogonalidad. El PE entra al score por el
   **readout (L3)**, no por volver informativa la atención. La hipótesis redundante↔CKA no aplica.

### Por qué (diagnóstico que lo motiva)
El RWPE de la Fase C (Resultado 9) volvió la atención load-bearing pero NO subió MRR porque es
**redundante** con el V-RMPNN: relation-agnostic + head-anclado = copia degradada de la
alcanzabilidad que NBFNet ya propaga. Para subir MRR el PE debe cargar señal **ortogonal** al
V-stream head-anclado (identidad intrínseca del candidato, estructura global, firma de relación).

### Qué se implementó (`src/model.py` + wiring `lightning.py`/`main.py`)
Toolkit de **Structural Query-PE**, componentes ensamblables, todo ablación anidada (default off =
baseline intacto):
- `compute_reltyped_rwpe` (**L1**): landing-prob desde h **por grupo de relación** (`rel % G`).
- `compute_centrality` (**L2 intrínseco**): firma grado in/out por grupo (centrality, Graphormer).
- `compute_laplacian_pe` (**L2 global**): top-k eigvecs del Laplaciano (`which='LA'` sobre
  adyacencia normalizada), sign-flip en train.
- `build_structural_pe`: ensambla en orden canónico `STRUCT_PE_ORDER=[rwpe,rel,deg,lap,noise]`.
- `'noise'`: **control de capacidad** (mismo presupuesto de params que rwpe, sin estructura).
- **L3**: `--qk_pe_readout` concatena el PE del candidato al `mlp_out` (columnas init-a-0).
- Flags: `--qk_pe_kind` (coma-sep), `--qk_pe_groups` (def 8), `--qk_pe_lap_k` (def 8),
  `--qk_pe_readout`. `--qk_pe {post,input}` sigue eligiendo punto de inyección al QK-stream.

### Decisiones de diseño (registrar en el paper)
- **Leakage**: rwpe/rel (head-anclados) usan el grafo **enmascarado** (`graph_mask` quita el triple
  objetivo, lightning.py:105-126) porque el landing-prob desde h ilumina el target. deg/lap
  (globales, no query-specific) usan el grafo **completo**, cacheados 1×/split (sin leak, rápido).
- **eigsh**: `which='SA'` sobre L no converge (ARPACK -1); fix = `which='LA'` sobre adyacencia
  normalizada (= menores de L), `maxiter=20N`, `tol=1e-3`.
- **Cache** (`_PE_GRAPH_CACHE`, `_LAP_CACHE`) por firma de grafo: sin él, reconstruir matrices
  sparse por batch daba ~0.74 it/s; con él **~4 it/s ≈ baseline** (~22 min/run de 20 epochs).

### Bugs ya resueltos (no re-descubrir)
1. PE reconstruido por batch → cache de matrices de transición / deg / lap.
2. eigsh no converge con 'SA' → 'LA' sobre adyacencia normalizada.
3. `graph_mask` varía por batch → split masked (rwpe/rel) vs full+cache (deg/lap).

### Matriz lanzada — `run_structpe.sh` (2026-06-11)
wn18rr_v2 ind, 1 seed (gate mecánico + MRR), todas con `--qk_pe post --qk_pe_readout`,
recipe canónico (hidden_dim 32, num_heads 4, batch 64, 20 epochs). Corriendo en background
(`logs/structpe_driver.log`, manifest `logs/structpe/manifest.csv`); orden de celdas
`baseline → noise → rwpe → rel → deg → lap → full`:

| celda | `--qk_pe_kind` | aísla |
|---|---|---|
| baseline | (sin `--qk_pe`) | referencia |
| noise | `noise` | **control de capacidad** (¿Δ = params de fc_pe?) |
| rwpe | `rwpe` | Fase C (redundante) |
| rel | `rel` | L1 |
| deg | `deg` | L2 intrínseco |
| lap | `lap` | L2 global |
| **full** | `rel,deg,lap` | L1+L2 |

Comando tipo (una celda):
```
python main.py --seed 42 --accelerator gpu --precision 32 --devices 1 --max_epochs 20 \
  --checkpoint_save_path ./experiments/structpe/full_seed42 --data_path ./data/inductive/wn18rr_v2 \
  --batch_size 64 --test_batch_size 64 --num_workers 8 \
  --num_layer 3 --num_qk_layer 2 --num_v_layer 3 --hidden_dim 32 --num_heads 4 \
  --loss_fn bce --adversarial_temperature 0.5 --num_negative_sample 8 \
  --learning_rate 5e-3 --optimizer Adam --weight_decay 1e-4 \
  --qk_pe post --qk_pe_kind rel,deg,lap --qk_pe_groups 8 --qk_pe_lap_k 8 --qk_pe_readout
```
(NO usar `--strategy ddp` en sesión srun-interactiva; NCCL roto, ver Resultado 5.)

### Gates (PLAN_STRUCTURAL_PE.md §3)
1. **MRR**: `Δmrr > 0.01` multi-seed (gate de éxito final). PENDIENTE (corriendo 1 seed).
2. **Ortogonalidad** ✅ CABLEADO: `linear_cka(out_attn, out_vN)` en `diag_rala_mechanism.py`
   sección (D), nodos=muestras, por capa. Debe **caer** en deg/lap/full vs rwpe.
3. **Capacidad**: `full` (o el ganador) debe superar a `noise` con significancia ⇒ el contenido
   estructural importa, no los params de `fc_pe`. PENDIENTE (corriendo).
4. **Load-bearing**: `eval_noattn.py` sobre el ganador (ya existe).

### Smoke del gate de ortogonalidad (2026-06-11, ckpt `qkpe/pe_post` = Fase C rwpe)
Validación end-to-end de la sección (D) nueva sobre el ckpt RWPE ya entrenado (reproduce exacto
`erank_K=2.10` cap0 del Resultado 9 ⇒ lee bien el ckpt). CKA(out_attn,out_vN) por capa:

| capa | CKA(attn,vN) | CKA(attn,sumv) |
|---|---|---|
| 0 | 0.716 | 1.000 |
| 1 | 0.044 | 1.000 |
| 2 | 0.231 | 1.000 |

media CKA(attn,vN) = **0.330**. Dos lecturas: (i) **CKA(attn,sumv)=1.000 en todas las capas**
corrobora por un ángulo nuevo que `out_attn` es rango-1 sobre nodos (se alinea perfecto con el
término DC, que es rango-1) — consistente con `srank(attn_only)≈1` de todo el log; (ii) el RWPE
redundante **ya da CKA=0.330 (no ~1)**, así que el "techo de redundancia" de referencia no es
alto ⇒ el gate de ortogonalidad habrá que leerlo **por-capa** y con margen, no esperar una caída
grande desde 1. (1 ckpt, 1 batch de 32 queries; indicativo, no concluyente.)

### Incidencia: bug `GROUPS` (variable reservada) — rel/deg/full murieron por OOM (2026-06-11)
El primer lanzamiento usó `GROUPS=8` en `run_structpe.sh`. **`GROUPS` es read-only en bash**
(array de group-IDs del usuario); la asignación se ignoró y `$GROUPS` se expandió al GID
(`686600032`). Las celdas que usan grupos de relación —`rel` (`walk_len*groups`) y `deg`
(`2*groups`)— pidieron ~5e9 canales de PE ⇒ **OOM ⇒ SIGKILL** (sin traceback Python). `full`
las incluye ⇒ también falla. **`baseline`/`noise`/`rwpe`/`lap` NO usan groups ⇒ resultados
VÁLIDOS, no se re-corren.** Fix: `GROUPS`→`PE_GROUPS` en `run_structpe.sh`; `run_structpe_missing.sh`
relanza solo rel/deg/full (appendea al manifest). Lección: no usar `GROUPS` como nombre de var en bash.

### (a) MRR — matriz completa (1 seed, wn18rr_v2 ind)
| celda | señal | test_mrr | Δ vs baseline |
|---|---|---|---|
| baseline | — | 0.7000 | — |
| rwpe | alcanzabilidad rel-agnostic (Fase C) | 0.7054 | +0.0054 (piso) |
| lap | rol global (Laplacian eigvecs) | 0.6993 | −0.0007 |
| rel | alcanzabilidad por relación (L1) | 0.6925 | −0.0076 |
| noise | ruido (control de capacidad) | 0.6795 | −0.0205 |
| deg | grado/relación intrínseco (L2) | 0.6644 | **−0.0356** |
| full | rel+deg+lap | cortado (valid 0.444) | colapsa |

`noise` cae −0.020 ⇒ inyectar canales al QK-stream+readout NO es neutral, *daña*; la vara es
superar **baseline**, no `noise`. `deg` cae por DEBAJO de `noise` (−0.036) ⇒ no es capacidad,
es la **señal** la que perjudica. Las dos intrínsecas (deg, lap) y rel son ≤ baseline. Hipótesis:
el PE intrínseco es **graph-específico** y rompe la generalización inductiva (test graph disjunto).

### (b) Des-smoothing de K + CKA (`diag_rala_mechanism.py` sobre los 4 ckpts, `logs/structpe/cka_all.log`)
`deg` es la ÚNICA intervención del proyecto que des-smootheó K de verdad — y empeoró el MRR:
| ckpt | capa | srank_K | erank_K | DCfrac_K |
|---|---|---|---|---|
| baseline | 2 | 1.01 | 1.41 | 0.986 |
| **deg** | 2 | **1.34** | **3.50** | **0.468** |

DCfrac_K 0.99→0.47, erank_K 1.4→3.5 (casi el gate >4 que nunca se alcanzó), y aun así MRR −0.036.
**Subir el rango de K empeora el MRR** ⇒ remate de la disociación rango≠MRR.

CKA(out_attn, ·) media por capa:
| celda | CKA(attn,vN) | CKA(attn,**sumv**) |
|---|---|---|
| baseline | 0.151 | **1.000** |
| rwpe | 0.098 | **1.000** |
| deg | 0.246 | **1.000** |
| lap | 0.144 | **1.000** |

`CKA(attn,sumv)=1.000` en todas ⇒ `out_attn` es rango-1/DC pase lo que pase (el PE no vuelve
informativa la atención; entra por el readout L3). `CKA(attn,vN)` bajo en todas por diferencia de
rango, NO por ortogonalidad ⇒ el gate de ortogonalidad CKA(attn,vN) **no separa** redundante de
ortogonal en esta arquitectura (de hecho `rwpe`, el "redundante", da el CKA más bajo). Gate retirado.

### Artefactos nuevos (2026-06-11)
- `run_structpe.sh`: matriz de 7 celdas, appendea a `logs/structpe/manifest.csv`, imprime los
  comandos de los 4 gates al terminar. (`GROUPS`→`PE_GROUPS` corregido.)
- `run_structpe_missing.sh`: relanza rel/deg/full con `PE_GROUPS=8` correcto (appendea).
- `diag_rala_mechanism.py`: `linear_cka()` + sección (D) CKA(out_attn,out_vN) y CKA(out_attn,out_sumv)
  por capa. Usa el hook existente (`out_attn`/`out_vN`/`out_sumv`, `model.py:543-545`).

### Conclusión y qué NO hacer a continuación
La línea "arreglar los inputs del QK-stream" (Fases A + C + Structural Query-PE) está **cerrada en
negativo** sobre WN18RR inductivo: ni la señal redundante (rwpe) ni la ortogonal intrínseca
(deg/lap) suben MRR; la intrínseca **daña** por romper la inductividad; y des-smoothear K (deg)
empeora el MRR. **No correr 3 seeds ni más componentes de PE en este régimen** — el techo es de la
tarea (Causa 5), no del PE. Direcciones que SÍ cambiarían el régimen (no más parches al QK-input):
- **Cambiar de dataset/régimen**: FB15k-237 (denso, popularity bias → deg podría ayudar en vez de
  dañar al ser transductivo, sin el problema de transferencia); o splits composicionales largos.
- **Cambiar la arquitectura**: atención por pares / Edge Transformer (la atención opera donde el MP
  no llega; no-redundante por construcción). Ver "Dirección futura" abajo.
- (Si se insiste en PE intrínseco: probarlo en **transductivo** primero, donde no hay test graph
  disjunto que rompa la transferencia — es la hipótesis que explicaría el daño de deg.)

---

## Resultado 11 — Re-baseline limpio (post-fix leak) + Autopsia de errores (Stage 0) — POSITIVO: luz verde a CPA (2026-06-14)

**Contexto**: arranque de la Fase 1 redefinida (CPA, sin expander). PLAN_MRR.md §7 paso 1-2.

### (a) Fix del leak val=test — APLICADO y re-baseline corrido
`lightning.py` ambos DataModules ahora usan `valid_triplets`+`valid_collate_fn` en
`val_dataloader()` (inductivo: valid sobre el train graph, sin `is_ind=True`). Verificado +
validación corre limpia (`valid_mrr≈0.585`). **Re-baseline wn18rr_v2 ind, 3 seeds**
(`run_rebaseline.sh`): test_mrr 42/43/44 = 0.6864/0.6928/0.6903 ⇒ **media 0.6898±0.0027**.
Histórico CON leak 0.7038 ⇒ −0.014 esperado (el leak capturaba varianza inter-epoch), NO
regresión. **Referencia para gates de CPA: Δ>0.01 ⇒ superar ~0.700.**

### (b) Baseline FB15k-237 ind v1 (recipe README: hidden_dim 64, neg 6, single-device)
test_mrr=**0.4440**, H@1=0.349, H@10=0.600, MR=131.9 (410 queries). Headroom real (vs WN18RR
saturado 0.69) ⇒ régimen de desarrollo correcto.

### (b') Baseline FB15k-237 TRANSDUCTIVO — EN CURSO (2 GPUs DDP, job 684912)
`sbatch_fb237_trans.sh`, recipe README trans (hidden_dim 32, bce+adv_temp 0.5, `--remove_all`,
neg 8), 6.1M params, 2×H100 DDP. ~23 min/epoch × 20 ⇒ ETA ~7.5-8 h. **Receta DDP 2-GPU que
funciona en este cluster** (los 3 fixes, no obvios — ver [[nccl-ddp-broadcast-fails]]):
1. **`srun --ntasks-per-node=2`** (NO plain-python): bajo SLURM, PL con `ntasks=1` colapsa a
   `world_size=1` y usa 1 sola GPU pese a `--devices 2`. Con 2 tasks ⇒ 1 rank DDP por GPU.
2. **`srun --cpu-bind=none`**: evita `CPU binding outside of job step allocation` (las CPUs de
   las GPUs no-contiguas, p.ej. IDX 5-7, están fragmentadas entre NUMA nodes).
3. **`NCCL_IB_DISABLE=1` + `NCCL_NET=Socket` + `NCCL_SOCKET_IFNAME=lo`**: evita el topology parse
   roto (`Attribute busid of node nic not found`) que mataba el primer `broadcast`. Single-node
   ⇒ NCCL usa NVLink/P2P para los datos y loopback para el rendezvous.
Número pendiente: leer `test_mrr` de `logs/autopsy/fb237_trans_684912.log` al terminar.

### (c) Autopsia (`autopsy.py`): MRR estratificado por d(h,t)/grado(t)/freq(r)
Harness nuevo: dump per-query (h,r,t,rank) gated por `KNOWFORMER_RANK_AUTOPSY` en
`lightning.py:test_step`; estratificación + Figura 1 en `autopsy.py`. d(h,t)=BFS no-dirigido en
el grafo de MP de test (test triples held-out ⇒ d=1 vacío, sin leak).

**MRR vs d(h,t) — colapso monótono universal, masa distinta por régimen:**

| d(h,t) | FB15k-237 v1 (MRR / %) | WN18RR v2 (MRR / %) |
|---|---|---|
| d=2 | 0.847 / 33% | 0.972 / 61% |
| d=3 | 0.426 | 0.552 |
| d=4 | 0.252 | 0.442 |
| d≥5 | 0.183 | 0.177 |
| d=∞ | 0.166 | 0.030 |
| **% d≥4 / ∞** | **52%** | **19%** |

El MP de L=3 alcanza ~3 hops; el MRR se cae por un acantilado en d≥3-4 en AMBOS datasets (el
mecanismo es universal). Pero FB15k-237 tiene 52% de masa a d≥4/inalcanzable vs 19% en WN18RR
(dominado por d=2 fácil). Confirma con datos: **desarrollar en FB15k-237 (headroom en
composición larga), sostener en WN18RR.** grado(t): MRR↑ con grado (esperado, no bias dañino).
freq(r): sin señal en FB15k-237 (no-monótono); degenerada en WN18RR (~11 rels) ⇒ **P2/contenido
NO indicado**.

### Veredicto
La autopsia es la **Figura 1 del paper** y la luz verde con datos a **CPA**: los errores del
baseline se concentran exactamente en queries de larga distancia composicional (>horizonte del
MP), que es lo que `x_{h,u} ⊙ x_{u,t}` (2L hops) ataca por construcción. Siguiente: related-work
pass + CPA mínima en fb15k-237_v1 (gates PLAN_MRR §6). Artefactos: `autopsy.py`,
`logs/autopsy/*_ranks.pt`, `figs/autopsy/*_mrr_vs_d.png`, ckpt baseline FB15k-237 v1.

---

## Resultado 12 — CPA mínima (Compositional Pivot Attention): NEGATIVO LIMPIO + causa mecánica (2026-06-14)

**Contexto**: primera implementación de la apuesta central de la Fase 1 (PLAN_MRR P1). CPA = Edge
Transformer restringido a la fila h × k pivotes: 1ª corrida V-RMPNN anclada en h → `x_{h,v}`;
selección top-k de pivotes u por `s(u)=w·x_{h,u}`; 2ª corrida (mismo V-RMPNN, pesos compartidos)
anclada en cada u → `x_{u,t}`; atención de composición `out(t)=Σ_u softmax_u(β(x_{h,u}))·g(x_{h,u}⊙x_{u,t})`.
Módulo top-level calculado 1 vez tras el stack, aditivo a x, gate `fc_out` init-0 ⇒ baseline anidado
(verificado: grad a v_layers de CPA = 0 al init). Toggles `--cpa_k/--cpa_compose/--cpa_mode/--cpa_random_pivots/--cpa_v_layer`.

### Setup
FB15k-237 ind v1 (régimen de desarrollo, Resultado 11), recipe README (hidden_dim 64, neg 6,
bce+adv_temp 0.5), 1 seed (42), 20 epochs, single-device GPU 0. **Baseline en este code-path
(cpa off) = 0.4626** (referencia interna apples-to-apples; el 0.4440 de Resultado 11 difiere por la
varianza del ruido gaussiano del QK-stream entre corridas — la comparación válida es vs 0.4626).
~13 s/época, 9.75 GB. `run_cpa.sh` + `run_cpa_variants.sh`, `logs/cpa/manifest.csv`.

### (a) MRR — TODAS las celdas por debajo del baseline (gate §6.1 FALLA, kill §6.5 cumplido)
| celda | flags | test_mrr | valid_mrr | Δ test |
|---|---|---|---|---|
| **baseline** | `--cpa_k 0` | **0.4626** | 0.478 | — |
| cpa_k16 | k16 distmult relcond | 0.4447 | 0.478 | −0.018 |
| cpa_k8_rand | k8 **pivotes aleatorios** | 0.4538 | 0.477 | −0.009 |
| cpa_k8_mlp | k8 **mlp** relcond | 0.4433 | 0.472 | −0.019 |
| cpa_k8_agnostic | k8 distmult **agnostic** | 0.4370 | 0.479 | −0.026 |
| cpa_k4 | k4 distmult relcond | 0.4362 | 0.479 | −0.026 |
| cpa_k8 | k8 distmult relcond | 0.4313 | 0.476 | −0.031 |

Kill-criterion §6.5 satisfecho: barrido k∈{4,8,16} + ambos compose (distmult/mlp) + ambos modos
(relcond/agnostic) + control de capacidad. **Ninguna supera baseline; todas dañan.** Dos firmas:
1. **valid ≈ baseline (0.472–0.479) pero test por debajo** ⇒ lo que CPA aprende NO transfiere al
   grafo de test disjunto (mismo modo de falla inductivo que `deg`, Resultado 10). Ni siquiera en
   valid (train graph, sin problema de transferencia) CPA aporta ⇒ no es solo transferencia.
2. **Control de capacidad INVERTIDO**: `cpa_k8_rand` (pivotes aleatorios, 0.4538) **> `cpa_k8`**
   (selección dirigida, 0.4313) por +0.022. La selección dirigida es PEOR que la aleatoria ⇒ la
   topología query-conditioned no aporta; el score de selección solo añade un sesgo train-específico
   que daña la transferencia. (Gate §6.3 falla al revés.)

### (b) Causa mecánica — las corridas ancladas son cuasi-invariantes al ancla (diagnóstico decisivo)
Instrumentación de `CPA.forward` sobre el ckpt `cpa_k8` (1 batch test, b=16, k=8, N=1093, d=64):

| tensor | srank | erank | DCfrac |
|---|---|---|---|
| `x_{h,v}` (corrida en h) | 1.32 | 25.8 | 0.001 |
| `x_{u,v}` (corridas en pivotes) | 1.32 | 25.7 | 0.001 |

Las corridas **NO** están over-smoothed sobre nodos (erank≈26, diversas por nodo). El problema es
otro: **las corridas ancladas en pivotes distintos son casi idénticas ENTRE SÍ**:
`cos(x_{u,·}, x_{u',·}) = 0.992` (misma query, pivotes distintos) ⇒ el ancla one-hot del pivote
**se lava** tras L=3 capas del V-RMPNN. Consecuencia directa: **solo el 7.4% de la energía de
`comp = g(x_{h,u}⊙x_{u,t})` varía entre pivotes**; el 93% es común a todos. CPA añade
efectivamente **un solo canal promediado**, no k evidencias composicionales distintas. Esto explica
las tres observaciones (a): no hay composición real ⇒ plano en valid; qué pivote elijas casi no
importa (7%) ⇒ random ≥ dirigido; el resto es ruido/sesgo que daña test.

### Veredicto
La apuesta "x_{h,u}⊙x_{u,t} sobre k pivotes" **no se sostiene con corridas NBFNet vanilla**: la
premisa de que NBFNet es "la fila h del Edge Transformer" es solo **débilmente** cierta — las filas
(corridas ancladas en distintas semillas) colapsan al ~99% en norma; la señal ancla-específica es un
término de 2º orden pequeño que el training gate-0 no extrae. Es un negativo mecánicamente
interpretable (publicable como tal), no solo un null de MRR.

**Salvamento posible (1 iteración, NO ejecutado — decisión del usuario)**: aislar el residuo
pivote-específico antes de componer (centrar `x_{u,v}` sobre los pivotes / sobre su media, o
re-inyectar el ancla en cada capa / menos capas internas para que no se lave). Si el 7% pasa a
dominar, CPA tendría material real que componer. Si no ⇒ STOP definitivo de la línea de pares
lineales y pivote a Edge Transformer pleno (P4, O(N²), estados de par reales) o reenfoque del paper
como diagnóstico+metodología.

### (c) Fix `--cpa_center` EJECUTADO (decisión usuario, 2026-06-14) — parcial pero NO cierra el gate
Centrar `x_{u,v}` sobre la dim de pivotes (resta el backbone comun, el 100% del residuo pasa a variar
entre pivotes). 1 seed, fb15k-237_v1. Comparar vs baseline 0.4626 y vanilla cpa_k8 0.4313:
| celda | test_mrr | valid_mrr |
|---|---|---|
| baseline | 0.4626 | 0.478 |
| cpa_k8_center | 0.4440 | **0.481** |
| cpa_k8_center_rand (control) | 0.4374 | **0.484** |
| cpa_k8_center_v2 (cpa_v_layer 2) | 0.4296 | 0.476 |
| cpa_k8 vanilla | 0.4313 | 0.476 |

Tres lecturas, las dos primeras a favor, la tercera fatal:
1. **El fix aisló señal útil EN-DISTRIBUCION**: por primera vez en esta rama el valid sube **sobre
   baseline** (0.481/0.484 > 0.478). El residuo pivote-específico SI tiene material composicional.
   Y recupera test vs vanilla (0.444 vs 0.431).
2. **No transfiere**: todas siguen **bajo baseline en test** (mejor −0.019). La señal es
   train-graph-específica (mismo split valid↑/test↓ que `deg`, Resultado 10).
3. **El gate de capacidad (§6.3) SIGUE sin pasar, incluso con el fix**: `center_rand` (pivotes
   ALEATORIOS) iguala o supera a `center` dirigido (valid 0.484>0.481; test 0.437≈0.444, dentro de
   ruido). **La selección query-conditioned no aporta sobre aleatoria** ⇒ no es "atención de pivotes
   composicional", es **capacidad añadida** que ayuda in-distribution y sobreajusta el grafo disjunto.

### Veredicto FINAL de CPA (post-fix)
La hipótesis central —"componer `x_{h,u}⊙x_{u,t}` sobre pivotes *seleccionados* añade MRR"— **no se
sostiene en fb15k-237_v1 inductivo**: (i) gate MRR §6.1 falla (toda celda < baseline en test);
(ii) gate capacidad §6.3 falla (dirigido ≈ aleatorio, incluso tras el fix que sí extrajo señal). Lo
único positivo es señal composicional in-distribution (valid↑) que NO transfiere. Kill-criterion
§6.5 cumplido a fondo (k-sweep + ambos compose + ambos modos + control + el fix dirigido).
Pista abierta (no perseguida sin decisión): el valid↑ sugiere que el régimen **transductivo**
(train=test entidades, sin transferencia a grafo disjunto) podría capturar esa señal — pero el
fallo dirigido≈aleatorio socava la novedad del mecanismo (la selección "con quién") independiente del
régimen. Fallbacks: Edge Transformer pleno (P4, estados de par reales) o paper como diagnóstico+metodología.

Artefactos del fix: `--cpa_center` (`src/model.py` CPA, centra x_{u,v} sobre pivotes), `run_cpa_fix.sh`,
celdas `cpa_k8_center*` en `logs/cpa/manifest.csv` y `experiments/cpa/`.

### Artefactos
- `src/model.py`: clase `CPA` + wiring en `Knowformer` (toggles off por defecto, baseline anidado).
- `lightning.py`/`main.py`: args `--cpa_k/--cpa_compose/--cpa_mode/--cpa_random_pivots/--cpa_v_layer`.
- `run_cpa.sh` (barrido k + control), `run_cpa_variants.sh` (compose/mode). `logs/cpa/manifest.csv`,
  `experiments/cpa/*_seed42/`.

---

## Resultado 13 — CPA CERRADO + pivote a A1 (candidate-set attention): Gate 0 oracle POSITIVO (2026-06-17)

**Contexto**: decisión del usuario — CPA dado por finalizado (Resultado 12 + fix center: kill-criterion
§6.5 cumplido a fondo, dirigido≈aleatorio mata la novedad del mecanismo). Nuevo plan: **A1 = atención
softmax sobre el set de candidatos top-K con bias por par estructural relativo** (el reranker
pointwise→listwise). Diagnóstico que lo motiva: el readout actual puntúa cada nodo **pointwise**
(`mlp_out(x_v)`); el MRR es **listwise** (orden relativo entre hard-negatives confundibles). Ningún
número de capas de MP single-source compara candidatos entre sí; la atención sobre el set de candidatos
sí. Es el único lugar provablemente **no-redundante** (esquiva R9) e **inductivo-safe** (features de
par relativas, esquiva R10) que queda sin expander.

### (a) Gate 0 — análisis oracle (PASA por 2.5–5×, sin construir nada)
`eval_oracle.py` (nuevo): sobre los ranks per-query ya volcados por `autopsy.py` (no corre el modelo),
mide recall@K y **oracle-MRR@K** = MRR si el top-K quedara perfectamente ordenado
(`RR=1` si rank≤K, si no `1/rank` intacto: el reranker no mueve al gold fuera del top-K).

| dataset | MRR_base | K=16 | K=32 | K=64 | K=128 |
|---|---|---|---|---|---|
| WN18RR v2 ind | 0.6866 | recall .776 / **head +0.091** | .811 / **+0.125** | .829 / +0.143 | .848 / +0.162 |
| FB15k-237 v1 ind | 0.4440 | .617 / **+0.177** | .646 / **+0.205** | .702 / +0.260 | .793 / +0.349 |

Umbral del plan: headroom>0.05 ⇒ construir A1. Se supera por 2.5–5×. Lectura clave:
**oracle-MRR@K ≈ recall@K** (la cola fuera del top-K aporta ~0) ⇒ toda la brecha es masa **ya
recuperada en el top-K pero mal ordenada** (ranks 2..K). El reranker no necesita subir recall, solo
reordenar. Caveat honesto: es la cota superior (reordenamiento perfecto); confirma que el techo
EXISTE, no que sea capturable — eso lo decide A1-v0.

### (b) Baseline FB15k-237 transductivo (job 684912, 2×H100 DDP) — terminó
test_mrr=**0.4311** (H@1=0.338, H@3=0.473, H@10=0.611, MR=109.8; ckpt epoch 12). Régimen de
desarrollo del plan (headroom real vs WN18RR saturado) con su baseline limpio. Receta DDP del cluster
validada de punta a punta (sbatch_fb237_trans.sh, los 3 fixes NCCL/srun).

### Veredicto
Gate 0 da luz verde a A1 con datos. Siguiente: A1-v0 = `CandidateReranker` readout-only aditivo (delta
init-0 ⇒ baseline anidado), `--rerank_k/--rerank_layers`, loss conjunta (CE full + CE restringida a
top-K∪{gold}), candidatos con grafo enmascarado; wn18rr_v2 ind + fb15k-237 trans, 1 seed. Luego
ablación del bias por par (el claim) + control de capacidad (shuffled/noise). Artefacto: `eval_oracle.py`.

---

## Resultado 14 — A1-v0 mínima (candidate-set attention SIN bias por par): NEGATIVO + mecanismo (2026-06-17)

**Contexto**: primera implementación de A1 (PLAN A1). Decisión usuario: v0 mínima (sin bias por par,
para aislar si el sustrato listwise solo mueve MRR) sobre FB15k-237 ind v1 (1 GPU). `CandidateReranker`
(`src/model.py`): top-K por score base → tokens {query=rel-emb, candidatos=estado del nodo} →
`rerank_layers` capas `nn.TransformerEncoderLayer` softmax (sin bias por par) → delta escalar,
`delta_head` init-0 (baseline anidado, verificado: delta≡0 al init). Loss conjunta (CE full base +
CE restringida top-K∪{gold}). Toggles `--rerank_k/--rerank_layers/--rerank_heads/--rerank_weight`.

### (a) MRR — el reranker DAÑA (gate A1 falla por mucho; en valid TAMBIÉN)
| celda | test_mrr | valid_mrr |
|---|---|---|
| baseline (rerank_k 0) | 0.4686 | 0.4826 |
| rerank_k32 | 0.3722 | 0.4312 |
| rerank_k64 | 0.3700 | 0.4238 |
| rerank_k128 | 0.4203 | 0.4488 |

### (b) Diagnóstico decisivo — el daño es el REORDENAMIENTO, no el modelo base
Sobre ckpt `rerank_k32`, eval con vs sin reranker:
- **sin reranker (solo score base)**: test_mrr=**0.4535** ⇒ el modelo base está intacto (−0.015 vs
  baseline, dentro del ruido; la loss conjunta apenas lo perturbó).
- **con reranker**: 0.3727 ⇒ el reordenamiento en eval cuesta **−0.08**.

Restringido a las 267 queries con gold en top-32 (donde el reranker PUEDE ayudar):
- pick del reranker == gold: **124/267 = 46.4%** (peor que el pick del base, que ya tiene MRR alto in-set).
- rank del gold al reordenar: **EMPEORA 99, MEJORA 2**, igual 166 ⇒ estrictamente destructivo.

### Veredicto
El readout base `mlp_out(x_v)` ya extrae la mejor señal **pointwise** de las features del nodo. La
atención softmax sobre esas MISMAS features de candidato, **sin estructura entre pares**, no añade nada
discriminativo (mismo input que el base + mezcla) ⇒ solo ruido confiado que degrada el orden base ya
bueno. NO es calibración (un reranker calibrado sería neutro; aquí el *pick* mismo, 46%, es peor que el
base). **El sustrato listwise por sí solo no captura el headroom del oracle.** Ningún ajuste de training
lo rescata: el reranker no puede *identificar* el gold con features pointwise. Es un negativo
mecánicamente interpretable (eco de R9/R12: lo que se le da a la atención es redundante con lo que el
readout/V-stream ya tiene). **Redirige al paso 2 del plan**: bias por par estructural relativo (vecinos
compartidos / co-alcanzabilidad entre candidatos) — señal NO computable por MP single-source, el claim
real del paper. Decisión pendiente (usuario): construir el bias por par o reconsiderar A1.
Artefactos: `CandidateReranker` + `--rerank_*`, `run_rerank.sh`, `logs/rerank/manifest.csv`,
`experiments/rerank/*_seed42/`.

---

## Resultado 15 — A1 paso 2: bias por par estructural (el claim): señal REAL pero techo ~baseline (2026-06-17)

**Contexto**: R14 (v0 sin par) negativa ⇒ decisión usuario: construir el bias por par (paso 2, el
claim). `PairBiasEncoderLayer` (atención hand-rolled con bias aditivo por cabeza en los logits
candidato×candidato) + `pair_features` (adyacencia `A[u,v]` + vecinos comunes `log|N(u)∩N(v)|`,
intrínsecas/relativas, inductivo-safe, cacheadas). Control de capacidad `--rerank_pair_shuffle`
(baraja el par). `delta_head` init-0 ⇒ baseline anidado preservado (verificado). FB15k-237 ind v1, 1 seed.

### (a) MRR — el bias por par aporta señal REAL pero todo sigue < baseline
| celda | test_mrr | valid_mrr |
|---|---|---|
| baseline | 0.4663 | 0.4768 |
| k64_nobias (v0, este code-path) | 0.3924 | 0.4333 |
| **k64_pair** | **0.4018** | 0.4332 |
| k64_pair_shuffle (control capacidad) | 0.3827 | 0.4147 |
| k32_pair | 0.3731 | 0.3994 |

- **Gate 2 (capacidad) PASA**: k64_pair (0.4018) > k64_pair_shuffle (0.3827, +0.019) y > k64_nobias
  (0.3924, +0.009) ⇒ la estructura de par es señal NO-redundante real (no son params).
- **Gate 1 (MRR) FALLA por mucho**: toda celda < baseline; el reranker es net-destructivo (−0.06).

### (b) Diagnóstico — cuello de botella = pick accuracy, no calibración (ckpt k64_pair)
De 287 queries con gold en top-64: **53% ya tiene el gold en rank 1 por base** (el reranker solo puede
dañarlas). **pick reranker==gold = 44.3%** (v0 era 46% ⇒ el par NO mejoró la identificación).
Cambios de rank in-set: **MEJORA 16, EMPEORA 105**, igual 166 (vs 2/99 en v0 ⇒ el par sí arregló ×8
más casos duros, pero sigue perturbando la mayoría ya correcta). MRR_base(sin reranker)=0.4536 intacto.

**Cálculo del techo**: aun sin daño y conservando las 16 mejoras ⇒ +16×(1−0.4)/410 ≈ **+0.023** ≈
baseline. El headroom del oracle (+0.26@64) asumía pick PERFECTO (100%); a 44% el techo real es ~baseline.

### Veredicto
El bias por par estructural (adyacencia + vecinos comunes) es señal **real y no-redundante** (pair >
shuffle/nobias, MEJORA 2→16) PERO **demasiado débil para discriminar el gold de los hard-negatives**
(pick 44%). El reranker es net-destructivo porque perturba el 53% ya-correcto y solo acierta el pick
44%. **El cuello de botella es la pick accuracy, no la calibración**: aun con reranking perfecto-no-dañino
el techo es ~baseline. Es el tema recurrente (R9/R12/R14): la señal discriminativa para los negativos
duros de KGC es esquiva/redundante con lo que el base ya extrae. Opciones (decisión usuario): (i) features
de par más ricas (co-alcanzabilidad h→u/h→v, paths rel-typed entre candidatos) + regularización de delta
para frenar el daño — pero el techo ~baseline acota el upside; (ii) pivote a paper diagnóstico+metodología
(R1-R15 = cadena A* fuerte: la atención en KGC es inerte/redundante en todos los sustratos probados).
Artefactos: `PairBiasEncoderLayer`/`pair_features`/`_dense_adj_sym`, `--rerank_pair_bias/--rerank_pair_shuffle`,
`run_rerank_pair.sh`, `logs/rerank_pair/manifest.csv`, `experiments/rerank_pair/*_seed42/`.

---

## Resultado 16 — Gate §3 RCT (relpath vs node-state, kill-criterion front-loaded): NEGATIVO DECISIVO (2026-06-18)

**Contexto**: arranque de RCT (PLAN_FASE1_RCT) en rama git nueva `rct`. El plan §3 exige correr el
**gate §3 ANTES de construir el modelo** (lección de meses perdidos en CPA/A1): testea I4 directo.
Pregunta falsable: en queries confundibles (gold en top-K del baseline, 1<rank≤K), ¿un feature de
**tipo-de-camino-relacional** `h→v` separa el gold de los hard-negs **mejor** que el estado de nodo
`x_v` que el readout pointwise ya usa? Premisa de RCT: la señal discriminante vive en el espacio de
**relaciones**, no de nodos (I1). El gate la testea sin entrenar nada.

### Harness (`gate_relpath.py`, nuevo)
- **dump** (hook `KNOWFORMER_GATE_DUMP` en `src/model.py`+`lightning.py`, off por defecto = baseline
  intacto): por query con gold rank≤128 vuelca top-K **negativos puros** (top-K por score base sobre
  `~filter_mask`; el gold está en filter_mask ⇒ no contamina), sus estados `x_v`, el `x_v` del gold.
- **feature (i)** = `x_state` d-dim (lo que `mlp_out` puntúa).
- **feature (ii)** = bag-of-relation-paths `h→v` sobre el grafo de MP de test (leak-free, las test
  triples están held-out): `lastrel` = histograma de última-relación por longitud (dim L·|R|);
  `bigram` = composición **ordenada** de long.2 `(r1,r2)` = el operador REAL de RCT (dim |R|², solo
  tratable con |R| chico). Conteo de walks por scatter-add O(E)/hop, log1p.
- **Métrica**: logreg pooled con **GroupKFold por query** (out-of-fold) ⇒ **within-query AUC** (gold
  vs sus negs). Verde si (ii)≫(i) (Δ>0.03 y AUC_ii>0.55).

### Resultado — el estado de nodo GANA y SUBSUME la info de relación, en TODA configuración
Confundibles: FB15k-237 v1 ind n=123 (|R|=360, d=64); WN18RR v2 ind n=159 (|R|=20, d=32).

| dataset | feature (ii) | (i) nodo | (ii) relación | (i)+(ii) | Δ(ii−i) |
|---|---|---|---|---|---|
| FB15k-237 v1 | lastrel L=2 | 0.872 | 0.541 | 0.840 | −0.331 |
| FB15k-237 v1 | lastrel L=3 | 0.872 | 0.569 | 0.779 | −0.303 |
| FB15k-237 v1 | lastrel L=4 | 0.872 | 0.566 | 0.744 | −0.307 |
| WN18RR v2 | lastrel L=2 | 0.862 | 0.675 | 0.852 | −0.187 |
| WN18RR v2 | lastrel L=3 | 0.862 | 0.731 | 0.799 | −0.132 |
| WN18RR v2 | lastrel L=4 | 0.862 | 0.744 | 0.782 | −0.118 |
| WN18RR v2 | **bigram ordenado** | 0.862 | 0.566 | 0.830 | −0.296 |

**Tres lecturas, todas matan a RCT:**
1. **(i)≫(ii) siempre**: el estado de nodo separa el gold de sus hard-negs en AUC 0.86–0.87; la info
   de relación apenas supera el azar (0.54 FB / 0.73 WN; crece con L = alcanzabilidad, no composición).
2. **(i)+(ii) < (i) siempre**: añadir el feature de relación **degrada** ⇒ no aporta señal complementaria;
   lo que codifica YA está en el estado de nodo (subconjunto redundante + ruido dimensional).
3. **La composición ORDENADA (bigram, operador real de RCT) es la PEOR** (0.566): no es que mi
   codificación bag pierda el orden — el orden tampoco discrimina. Cobertura de caminos long-2
   ordenados gold=31% « lastrel.

### Veredicto — kill-criterion §6 cumplido, PIVOTE a paper B
La premisa central de RCT —"la señal discriminante para los hard-negs de KGC vive en el espacio de
**relaciones**, no en el estado de nodo"— está **falsificada con datos, antes de construir el modelo**.
**Causa unificadora de R1–R16**: el V-RMPNN **ya es** una agregación sobre caminos relacionales
(NBFNet = suma sobre paths), así que `x_v` ya contiene la evidencia de composición **mejor codificada**
que cualquier re-encoding explícito (pivotes CPA R12, bias por par A1 R14/R15, o caminos crudos R16).
Mover los tokens a relaciones no añade un sustrato nuevo: re-expone, peor, lo que el MP ya extrajo.
Esto cierra el fork A (RCT). **El gate hizo su trabajo: 1 sesión en vez de meses.** Fork B (paper
diagnóstico+metodología) es el camino: R1–R16 = cadena A* fuerte y ahora con un kill-criterion
front-loaded como pieza final. Decisión A/B = del usuario (pendiente).

Artefactos: `gate_relpath.py` (dump/analyze/both, `--feature {lastrel,bigram,both}`, `--K/--L`),
hook `KNOWFORMER_GATE_DUMP` (`src/model.py` stash `_gate_x_state`; `lightning.py` test_step/epoch_end),
`logs/gate/{fb15k237_v1,wn18rr_v2}.pt`. Rama `rct`.

---

## Resultado 17 — Gate #2 (recall/largo-alcance): el headroom ES alcanzable-pero-lejano en FB (2026-06-18)

**Contexto**: tras R16 (RCT muerto), pregunta del usuario — ¿hay ALGUNA línea donde la atención sea
útil en KGC? Honestidad: R1–R16 prueba que la atención como *re-agregador* sobre estados de nodo MP es
redundante (el V-RMPNN ya es suma de caminos), NO que la atención sea imposible. Apertura #2: atacar
**recall**, no reordenar — el slice donde el MP de L hops no alcanza el gold (`d(h,t)>L`). Gate barato
(`gate_recall.py`, 100% desde `logs/autopsy/*_ranks.pt` + BFS, sin correr el modelo): descompone el
headroom (1−MRR) en near (d≤3) / **far-reach (3<d<∞, ADDRESSABLE)** / unreachable (d=∞, techo del dato).

### Resultado — descomposición del headroom (horizonte=3, baseline num_v_layer=3)
| | FB15k-237 v1 ind | WN18RR v2 ind |
|---|---|---|
| MRR / headroom | 0.444 / 0.556 | 0.687 / 0.313 |
| near (d≤3): %q / %headroom | 70.2% / 55.7% | 78.7% / 34.5% |
| **far-reach (3<d<∞): %q / %headroom** | **26.3% / 39.3%** | 9.3% / **27.4%** |
| unreachable (d=∞): %q / %headroom | 3.4% / **5.0%** | 12.0% / **38.2%** |
| far-reach gold rank>32 / mediana rank | 63% / 78 | 56% / 42 |

**Lecturas:**
1. **FB15k-237 = verde (condición necesaria)**: 39% del headroom es alcanzable-pero-lejano y solo 5%
   inalcanzable. Los far-reach están FUERA del top-K (63% rank>32, mediana 78) ⇒ problema de **recall
   real**, que el reranking NO puede tocar (R13 oracle) ⇒ distinto de la línea A1 muerta. El cliff de
   MRR en d≈3 (= num_v_layer) sostiene la premisa de sub-alcance.
2. **WN18RR = mixto/pared**: domina lo **inalcanzable** (38.2% del headroom, d=∞ MRR≈0.005 = entidades
   de test aisladas) sobre lo addressable (27.4%) ⇒ buena parte del techo de WN es del dato, no del MP.
3. **Complementario al §3, no contradictorio**: el §3 (negativo) midió queries con gold EN top-K
   (confundibles, reranking-shaped); #2 mide las de FUERA del top-K (recall-shaped, mediana rank 78).
   Poblaciones distintas ⇒ el negativo del §3 no condena a #2.

### Veredicto — necesario PASA (FB), falta la SUFICIENCIA
Gate #2 prueba que el headroom **existe, es alcanzable y recall-shaped** (fuerte en FB). NO prueba que
sea **capturable**: la lección del §3/R10 (la señal a distancia puede no discriminar; y deeper MP →
más over-smoothing, R6/R10) está sin testear para este slice. **Próximo gate decisivo barato**:
¿un MP de mayor horizonte (num_v_layer 3→6 / num_layer 3→5) recupera el far-reach (d=4,5)? Si **sí** ⇒
reach es el cuello ⇒ tesis viva = atención para **largo alcance sin over-smoothing** (el diferenciador
vs "más capas NBFNet", que se auto-smoothea). Si **no** ⇒ ni el reach lo arregla ⇒ confirma paper B.
Artefacto: `gate_recall.py`, `--horizon`. Decisión de correr el depth-probe: pendiente del usuario.

---

## Resultado 18 — Depth-probe + sonda far-reach: el headroom NO es de reach pero la relación SÍ complementa al nodo en el slice far (2026-06-18)

**Contexto**: R17 dejó #2 en un filo: el headroom far-reach existe y es alcanzable, pero falta la
**suficiencia** (¿es capturable?). Dos interpretaciones: (a) la señal a distancia no discrimina (⇒ B);
(b) está pero el deep-MP la destruye por over-smoothing (⇒ #2 vivo con atención sin smoothing). Dos
sondas baratas para desambiguar.

### (A) Depth-probe — extender el horizonte del MP NO recupera el far-reach
`run_depthprobe.sh`: re-entrena FB15k-237 v1 (recipe baseline, solo cambia el horizonte), vuelca ranks,
compara MRR por d. 1 seed.
| celda | MRR global | far-reach MRR | d=4 | d=5 | d=6 |
|---|---|---|---|---|---|
| baseline (v3/L3) | 0.444 | 0.1710 | 0.183 | 0.150 | 0.209 |
| deep_v6 (num_v_layer **6**) | 0.442 | 0.1711 | 0.219 | 0.133 | 0.175 |
| deep_L5 (num_layer **5**) | 0.438 | 0.1513 | 0.157 | 0.141 | 0.196 |

**Más reach no compra far-reach**: deep_v6 sube d=4 (+0.036) pero baja d=5/6 ⇒ neto plano (0.1711≈0.1710);
deep_L5 empeora en todo (over-smoothing, R6/R10). El reach *crudo* no es el cuello ⇒ descarta "solo
agregar capas NBFNet". Deja vivas (a) y (b).

### (B) Sonda far-reach (§3-probe sobre la subpoblación d>3) — la relación COMPLEMENTA al nodo
Reusa el dump del §3 + filtro `--min_dist` (d(h,t) por BFS) + relpath largo (L=6) + 20 barajados de fold
(media±std). FB15k-237 v1, K=128 (incluye la cola de recall rank 32–128):
| slice | nodo (i) | relación (ii) | (i)+(ii) | **complemento (i+ii)−(i)** |
|---|---|---|---|---|
| **far-reach (d>3, n=58)** | 0.703±0.020 | 0.562±0.016 | 0.742±0.016 | **+0.039 ± 0.016 (>2σ)** |
| near (d≤3, n=118) **[control]** | 0.865±0.009 | 0.543±0.016 | 0.703±0.019 | **−0.162 ± 0.019** |

**Disociación limpia y específica:**
1. El estado de nodo **se debilita en far-reach** (0.865→0.703): el MP de horizonte-3 es ciego al
   gold lejano, justo como predijo R17.
2. La info de relación de **largo alcance complementa** al nodo **solo** en far-reach (+0.039, >2σ,
   estable sobre 20 reps); en near es **redundante/dañina** (−0.162, replica R16). El control near
   descarta que sea capacidad/dimensión (mismas 2160+64 dims en ambos).
3. **Resuelve (a) vs (b) a favor de (b), parcialmente**: la señal de largo alcance SÍ existe y SÍ es
   no-redundante con el nodo en el slice far — pero el deep-MP no la capturó (A) ⇒ la lee mal por
   smoothing/agregación, no porque no esté.

### Veredicto — primer hilo POSITIVO del proyecto, fino pero real y bien localizado
Las tres piezas convergen: el headroom vive en far-reach recall (R17, 39% en FB); ahí el nodo es ciego
y la **evidencia relacional de largo alcance es complementaria** (R18-B); y el deep-MP crudo no la
extrae (R18-A). El mecanismo indicado NO es RCT (relación-en-vez-de-nodo, muerto R16) ni deep-MP: es
**fusión nodo + evidencia relacional de largo alcance, targeteada al slice far-reach, sin el
over-smoothing del MP profundo**. La atención entra como el mecanismo de largo alcance no-smootheante.

**Caveats honestos (no sobre-vender):** efecto **chico** (+0.039 AUC, NO MRR; la traducción AUC→MRR fue
floja en R15), n=58, **1 dataset** (WN18RR far-reach pendiente, y WN tiene la pared de inalcanzable 38%,
R17). Es un *hilo*, no una tesis probada. Próximo gate barato: replicar la sonda far-reach en WN18RR; si
el complemento aguanta, diseñar la fusión mínima y testear si el +0.039 AUC se traduce en MRR del slice.
Artefactos: `run_depthprobe.sh`, `experiments/depthprobe/`, `logs/depthprobe/`, `gate_relpath.py`
(`--min_dist/--max_dist/--reps`). Decisión perseguir-hilo vs banco-para-paper-B: del usuario.

---

## Síntesis teórica — POR QUÉ NBFNet captura lo que el transformer podría (cierre de la línea arquitectónica, 2026-06-18)

Dos preguntas del usuario que cierran la línea con argumento, no solo con nulls empíricos.

### P1 — Un transformer cuyo V sale de un stream NBFNet, ¿captura lo mismo que el V ya tenía? **SÍ (demostrable).**
1. **NBFNet ya ES la agregación de caminos** (Neural Bellman-Ford, Zhu 2021): `h_v = ⊕_{paths s→v} ⊗_{edges} W_r`.
   `V_v` contiene toda la evidencia de caminos relacionales s→v hasta L hops; no es "features de nodo", es el agregado que la tarea necesita.
2. **La atención solo recombina**: `out_v = Σ_w softmax(q·k)·V_w` ∈ casco convexo de `{V_w}` ⇒ no puede producir
   una dirección **ortogonal** a `span{V_w}`. Si los `V_w` son agregados de L hops, toda mezcla lo es. No crea
   evidencia de caminos que el V no tenga.
3. Lo único no-redundante sería info de **fuera del horizonte de V** (caminos >L, estructura por par) — pero eso
   ya no es "V que viene del stream". Es exactamente R18-B: el +0.039 vino de features L=6 que el V de L=3 no tenía.
   ⇒ **La redundancia es coincidencia de horizonte/sustrato** (explica R1–R16), no debilidad del transformer.

### P2 — Un transformer SOLO (sin NBFNet, sin stream), ¿captura lo mismo que NBFNet? **Depende, y el matiz es el punto.**
- **(a) Transformer de conjunto, ciego a la estructura** (all-pairs sobre embeddings de nodo, sin adyacencia): **NO.**
  Sin aristas no hay caminos; tendría que aprender la adyacencia desde features ⇒ imposible en KGs arbitrarios y
  rompe inductividad (grafo de test disjunto).
- **(b) Graph transformer** (atención enmascarada por la adyacencia relacional, source marcado, L capas): **SÍ, en
  principio.** MP ⊆ atención-sobre-aristas (un GNN = atención con patrón fijo a la adyacencia); el **Edge Transformer**
  (Bergen 2021) prueba que un transformer hace composición relacional e iguala GNNs (CLUTRR). PERO: (i) para igualar a
  NBFNet hay que **darle la estructura** ⇒ está haciendo MP vía atención = **reinventa el MP**, no lo trasciende; (ii)
  **poder ≠ ganar**: el caso expresivo (par/triangular) cuesta O(N²–N³) ⇒ sparse/lineal pierde la expresividad extra; y
  R1–R18 son la evidencia empírica de que esa flexibilidad extra es **inerte** en single-graph WN/FB.

### Cierre
Los 3 argumentos pro-transformer vs GNN —(i) largo alcance 1-capa, (ii) agregación blanda input-dependiente, (iii)
razonamiento por par/orden superior— quedaron, en KGC single-graph: (ii) **inerte** (R1–R16, la suma fija basta), (iii)
**redundante** (CPA/A1/RCT), (i) **único resquicio** (R18) pero chico y sin traducir a MRR. **NBFNet captura todo no
porque el transformer sea más débil, sino porque su libertad extra resuelve flexibilidad que este problema, en estos
datos, no necesita.** La línea arquitectónica "transformer para ganar MRR en WN/FB" queda **cerrada con argumento**.
Valor acumulado = **diagnóstico (paper B)**: la atención es inerte/redundante con el MP porque NBFNet ya es la
agregación óptima de caminos; el único headroom (recall far-reach) no lo captura ni la atención sobre el sustrato ni el
MP profundo. R1–R18 = cadena A* completa. Refs: NBFNet (arXiv:2106.06935), Edge Transformer (Bergen 2021), KnowFormer (arXiv:2409.12865).

---

## Línea previa (descartada): anti-over-smoothing (PairNorm en el QK-RMPNN) — SUPERSEDED (2026-06-05)

**Nota (2026-06-09)**: esta línea quedó **descartada** por el pivote de PLAN_SOLUCION.md. El
barrido PairNorm/JK no se completó; la revisión experta concluyó que des-smoothear con
normalización es *necesario pero no suficiente* (rango ≠ información útil + chicken-and-egg de
query-conditioning), y que el cuello de botella es **qué se le entrega al encoder**, no la
normalización. Se reemplazó por Fases A (Resultado 8) y C (Resultado 9). `PLAN_ANTISMOOTH.md` no
existe como archivo; el plan vivo es **`PLAN_SOLUCION.md`**. Contenido original abajo (histórico).

## Línea nueva: anti-over-smoothing (PairNorm en el QK-RMPNN) — EN CURSO (2026-06-05)

**Pivote**: cerrado el STOP sobre "arreglar el *operador* de atención" (RALA), la pregunta
abierta es atacar la causa raíz medida —el bajo rango está en los **inputs** (RMPNN
over-smoothed)—. Objetivo de primera etapa (def. del usuario): un graph transformer de
**atención lineal** competitivo con KnowFormer/NBFNet en WN18RR/FB15k-237 trans+ind. Plan
falsable completo en **`PLAN_ANTISMOOTH.md`**.

**Hipótesis**: des-smoothear el QK-stream sube `srank_K` ⇒ `kvs` sube de rango ⇒ `q·kvs` se
vuelve informativo por nodo ⇒ la atención pasa a ser *load-bearing* ⇒ sube el MRR.
**Riesgo**: el techo podría ser intrínseco a la tarea (V-RMPNN ya extrae la señal) ⇒ `srank_K`
sube pero el MRR no. Por eso el gate principal es **mecánico** (atención load-bearing vía
ablación `NO_ATTN_TERM`), no el MRR (piso de ruido ~1pt, Fase 0).

**Intervención**: `PairNorm` (Zhao & Akoglu 2020) tras cada capa del QK-RMPNN. Centra sobre
nodos (ataca `DCfrac_K~0.99`); complementario al LayerNorm existente (que centra sobre
features, no sobre nodos). Implementado: clase `PairNorm` en `src/model.py`, flag
`--pairnorm_qk <scale>` (0 = no-op = baseline intacto), cableado lightning/main como `--rala_kv`.
Solo toca el QK-stream, NO el V-RMPNN que carga el MRR. Smoke test OK (1 epoch, entrena, 9GB).

**Fase A en curso** (`run_pairnorm.sh`): wn18rr_v2 ind, seed 42, 20 epochs, barrido
s ∈ {0, 0.5, 1, 2}. Gate: ¿`srank_K` sube de ~1 a >3? Resultados pendientes
(`logs/pairnorm/manifest.csv`). Si PairNorm es inerte ⇒ escalar a Jumping-Knowledge /
initial-residual (GCNII) sobre el QK-stream.

**Nota operacional**: el primer lanzamiento se hizo con `nohup ... &` DENTRO de un tool ya en
background; el hijo nohup sobrevivió y corrió en paralelo con el relanzamiento ⇒ dos instancias
pisándose ckpts/manifest (datos basura). Lección: lanzar el script directo como background del
harness (sin `nohup &` anidado), una sola vez.

---

## Dirección futura: atención a nivel de PAR (donde la atención sí sería central)

Pregunta del usuario: ¿existe un graph transformer para KGC donde la atención tenga
importancia *significativa*? Diagnóstico: en KnowFormer la atención es débil porque opera sobre
el **mismo sustrato que el MP (features de nodo)** haciendo **lo mismo (mezclar nodos)** ⇒
redundante. Para que importe debe operar donde el MP no llega:

- **(A) Atención sobre pares de entidades = composición relacional.** Estilo **Edge Transformer**
  (Bergen et al. NeurIPS'21): representación por par `x_ij`, **atención triangular** (actualiza
  `x_ij` atendiendo sobre `k` vía `x_ik`,`x_kj`) = composición de relaciones i→k→j. Es el álgebra
  del razonamiento KG como operador de atención nativo (análogo al Evoformer de AlphaFold). Aquí
  la atención NO es decorativa, y el costo N² pares hace que la **atención lineal sea *necesaria***
  (unifica con el objetivo de primera etapa). Riesgo: caro, validado en tareas chicas (CLUTRR),
  escalar a FB15k-237 requiere restringir pares a la vecindad/candidatos.
- **(B) Atención sobre caminos/subgrafos como tokens** = ranking diferenciable de caminos
  (interpretable, load-bearing).
- **(C) Régimen donde el MP se queda corto** (cadenas composicionales largas, splits más duros);
  en WN18RR/FB15k-237 el razonamiento es de pocos hops ⇒ el MP basta ⇒ la atención no agrega.

Es una **arquitectura distinta** de KnowFormer, apuesta de etapa más ambiciosa, ortogonal a la
línea PairNorm en curso.

---

## Artefactos en el repo

- **Fase 1 / CPA (Resultado 11, 2026-06-14)**:
  - `lightning.py`: fix del leak val=test (ambos DataModules usan `valid_triplets`+
    `valid_collate_fn`); hook de autopsia `KNOWFORMER_RANK_AUTOPSY` en `test_step`/`test_epoch_end`
    (vuelca per-query `(h,r,t,rank)` a un `.pt`, default off ⇒ baseline intacto).
  - `autopsy.py` — Stage 0. Subcomandos `dump`/`analyze`/`both`: corre test-only desde un ckpt
    con el dump activado y estratifica el reciprocal rank por d(h,t) (BFS no-dirigido en el grafo
    de MP de test), grado(t) y freq(r); genera la Figura 1 (MRR vs d(h,t)).
  - `run_rebaseline.sh` — re-baseline limpio wn18rr_v2 ind (3 seeds). `logs/rebaseline/manifest.csv`,
    `experiments/rebaseline/`.
  - `sbatch_fb237_trans.sh` — template SLURM 2-GPU DDP (FB15k-237 transductivo). Encapsula los 3
    fixes DDP del cluster (srun ntasks=2 + `--cpu-bind=none` + NCCL_IB_DISABLE/NET=Socket). Base
    para futuros runs multi-GPU (paper: FB15k-237 trans, YAGO3-10 escalabilidad).
  - `logs/autopsy/*_ranks.pt` (dumps), `figs/autopsy/*_mrr_vs_d.png` (Figura 1),
    `experiments/autopsy/fb15k237_v1_seed42/` (ckpt baseline FB15k-237 v1, test_mrr=0.444).
- `src/model.py`: hook `KNOWFORMER_RANK_DUMP` (con descomposición de términos + captura
  `q_norm`/`k_prenorm`/`alpha` para el diagnóstico mecánico) y toggles
  `KNOWFORMER_NO_ATTN_TERM`, `KNOWFORMER_NO_VN_TERM`, `KNOWFORMER_ATTN=softmax` (Fase 0).
- `analyze_rank.py` — espectros + rango. `analyze_rank_contrib.py` — contribución + rango centrado.
- `diag_rala_mechanism.py` — diagnóstico mecánico (Resultado 6c): spread de α, colinealidad
  (DCfrac/srank) de K, dominancia DC. Responde POR QUÉ α_j es inerte. **Extendido (Resultado 7)**:
  captura `v` en el hook y mide srank/erank/DCfrac de K **y** V lado a lado.
- `src/model.py`: clase `PairNorm` + flag `--pairnorm_qk <scale>` (anti-over-smoothing del
  QK-RMPNN, línea descartada). Cableado en `lightning.py`/`main.py`. 0 = off.
- `run_pairnorm.sh` — barrido anti-smoothing (s ∈ {0,0.5,1,2}); NO completado (línea descartada).
  `logs/pairnorm/`, `experiments/pairnorm/`.
- **Fase A (Resultado 8)**: `src/model.py` flag `--qk_anchor_head` (one-hot del head al input del
  QK-stream). `run_qkinput.sh`, `logs/qkinput/`, `experiments/qkinput/`. Plan: `PLAN_SOLUCION.md`.
- **Fase C (Resultado 9)**: `src/model.py` `compute_rwpe` + flags `--qk_pe {none,input,post}`,
  `--qk_pe_walk_len` (RWPE desde el head, inyección input/post). `run_qkpe.sh`, `logs/qkpe/`,
  `experiments/qkpe/`. Cableado en `lightning.py`/`main.py`.
- `run_rala.sh` — entreno factorial RALA (6 variantes × 3 seeds, d_head=8). `run_rala_missing.sh` —
  relanza a mano solo los faltantes appendeando al manifest. `analyze_rala.sh` — figs+rango por variante.
- `run_rala_dhead32.sh` + `run_rala_dhead32_missing.sh` — factorial d_head=32 (Resultado 6).
  `analyze_rala_dhead32.sh` — figs+rango apuntando a `logs/rala_dhead32/`.
- `src/model.py`: toggles RALA `--rala_kv`, `--rala_phi {none,x,v}` (mod 1 y mod 2). Cableados en `lightning.py`/`main.py`.
- `figs/rank_wn18rr_v2/` — figuras baseline. `figs/rank_wn18rr_v2_novn/` — figuras sin v·num_node.
  `figs/rank_wn18rr_v2_<variant>_dhead32/` — figuras por variante d_head=32.
- `experiments/rank/` — ckpts baseline y ablación. `experiments/rala_dhead32/` — ckpts d_head=32.
  `logs/rank/`, `logs/rala/`, `logs/rala_dhead32/` — logs/manifests de entrenamiento.
