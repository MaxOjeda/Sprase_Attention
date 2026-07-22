# SESSION_NOTES.md — bitácora de resultados y análisis

Registro cronológico de sesiones: qué se corrió, qué salió, qué se concluyó. Más reciente
arriba. Objetivos estables en `GOALS.md`; briefing operacional en `CLAUDE.md`.

Para cada entrada: **Resultado** (números), **Análisis** (causa mecánica), **Decisión**.

---

## TABLA RESUMEN — 4 modelos × datasets inductivos (best config, seed 42)

Cada modelo con su mejor config: NBFNet (dim 32, drop 0.1, lr 5e-3, pna);
full/sparse/sparse_nbfv (dim 64, drop 0.0, lr 1e-3). Todos L6, 20 ep. Métricas full-filtered.
Negrita = mejor por columna.

### FB15k-237 ind v1

| modelo              |    MRR | Hits@1 | Hits@3 | Hits@10 |
|---------------------|-------:|-------:|-------:|--------:|
| **NBFNet**          | **0.459** | **0.371** | **0.520** | **0.605** |
| Full attention      |  0.375 |  0.320 |  0.402 |   0.471 |
| Sparse attention    |  0.338 |  0.276 |  0.359 |   0.451 |
| Sparse_nbfv (V=NBF) |  0.421 |  0.339 |  0.476 |   0.554 |
| Full + RWSE         |  0.366 |  0.310 |  0.407 |   0.459 |
| Sparse + RWSE       |  0.292 |  0.227 |  0.337 |   0.376 |
| Sparse_nbfv + RWSE  |  0.424 |  0.332 |  0.480 |   0.593 |
| Full + LapPE        |  0.410 |  0.351 |  0.441 |   0.524 |
| Sparse + LapPE      |  0.335 |  0.285 |  0.363 |   0.410 |
| Sparse_nbfv + LapPE |  0.434 |  0.346 |  0.500 |   0.571 |
| Full + source_rw    |  0.313 |  0.266 |  0.332 |   0.395 |

### WN18RR ind v1

| modelo              |    MRR | Hits@1 | Hits@3 | Hits@10 |
|---------------------|-------:|-------:|-------:|--------:|
| **NBFNet**          | **0.740** |  0.689 | **0.774** |   0.822 |
| Full attention      |  0.673 |  0.638 |  0.691 |   0.739 |
| Sparse attention    |  0.738 |  0.686 | **0.774** |   0.819 |
| Sparse_nbfv (V=NBF) | **0.740** | **0.691** |  0.766 | **0.832** |
| Full + RWSE         |  0.670 |  0.638 |  0.684 |   0.734 |
| Sparse + RWSE       |  0.677 |  0.628 |  0.697 |   0.769 |
| Sparse_nbfv + RWSE  |  0.728 |  0.684 |  0.755 |   0.798 |

### FB15k-237 ind v2

| modelo              |    MRR | Hits@1 | Hits@3 | Hits@10 |
|---------------------|-------:|-------:|-------:|--------:|
| NBFNet              |  0.526 |  0.416 | **0.595** | **0.727** |
| Full attention      |  0.491 |  0.389 |  0.544 |   0.686 |
| Sparse attention    |  0.450 |  0.369 |  0.494 |   0.586 |
| **Sparse_nbfv (V=NBF)** | **0.527** | **0.431** |  0.586 |   0.690 |
| Full + RWSE         |  0.477 |  0.382 |  0.535 |   0.652 |
| Sparse + RWSE       |  0.472 |  0.382 |  0.517 |   0.644 |
| Sparse_nbfv + RWSE  |  0.529 |  0.427 |  0.589 |   0.703 |
| Full + LapPE        |  0.496 |  0.397 |  0.556 |   0.681 |
| Sparse + LapPE      |  0.445 |  0.358 |  0.486 |   0.592 |
| Sparse_nbfv + LapPE |  0.524 |  0.424 |  0.585 |   0.687 |
| Full + source_rw    |  0.363 |  0.292 |  0.402 |   0.479 |

**Lectura cruzada**: ninguna variante de atención supera claramente a NBFNet en MRR.
FB15k-237 v1 (composicional): todas pierden, sparse el peor, sparse_nbfv el mejor de atención.
FB15k-237 v2: mismo orden (sparse el peor, full por debajo de NBFNet), pero **sparse_nbfv
empata/roza a NBFNet** (0.527 vs 0.526, mejor Hits@1). WN18RR v1 (local): full el peor,
sparse/sparse_nbfv empatan a NBFNet. El message passing sigue siendo el techo en todos los
regímenes. Detalle por experimento en las entradas de abajo.

**RWSE (filas `+ RWSE`, 2026-06-28)**: no es mejora confiable. Full neutral-negativo en v1/v2
(−0.009, −0.014); sparse INVIERTE signo entre splits (−0.046 v1, +0.022 v2 → no robusto);
sparse_nbfv plano (ruido). Ninguno supera a NBFNet. **WN18RR v1 + RWSE (registrado 2026-07-08):
daña los 3** — full −0.003 (0.673→0.670), sparse −0.061 (0.738→0.677, su firma de overfit otra vez),
sparse_nbfv −0.012 (0.740→0.728). Consistente con FB15k-237: RWSE nunca supera a NBFNet y al sparse
lo perjudica fuerte. RWSE completo en los 3 datasets ⇒ conclusión (lista negra #7) confirmada.

**LapPE (filas `+ LapPE`, 2026-07-08, v1 + v2)**: PE GLOBAL (autovectores del Laplaciano de TODO el
grafo, no local como RWSE). En v1 dio un bump prometedor al full (+0.035, 0.375→0.410), PERO **NO se
replica en v2 (+0.005, 0.491→0.496, dentro de ruido) ⇒ el efecto NO es robusto**, cae al split como
el sparse+RWSE. Δ test por LapPE: Full +0.035(v1)/+0.005(v2); Sparse −0.003/−0.005; Sparse_nbfv
+0.013(v1)/−0.003(v2). Ningún signo consistente y positivo en ambos splits; **ninguno supera a
NBFNet** (v1 full+LapPE 0.410 vs 0.459; v2 full+LapPE 0.496 vs 0.526). Conclusión: LapPE se une a RWSE
como PE no confiable para atención. El bump de v1 fue un artefacto de split, no una mejora estructural.

**source_rw (filas `Full + source_rw`, 2026-07-08)**: labeling trick CONDICIONADO A LA FUENTE
(no node-only como RWSE/LapPE): feature del nodo v = proj de las probabilidades de landing de un
random walk de k=1..8 pasos que ARRANCA en el head de la query (fila head de P^k, P=D^-1 A). En
principio es la señal query-relativa que GOALS deja abierta. Resultado: **el más dañino de los tres
encodings**. A diferencia de RWSE/LapPE (que degradan poco y sólo en test), source_rw **derrumba
valid Y test en ambos splits**: full v1 test 0.375→0.313 (−0.062), v2 test 0.491→0.363 (−0.128);
valid v1 0.429→0.234, v2 0.461→0.279. No es firma de overfit (valid↑/test↓) sino degradación neta
in-distribution incluida. Se une a RWSE/LapPE como fuente de señal NO no-redundante para la atención.

---

## 2026-07-22 — IMPLEMENTADO: atención sin normalizar (`--attn sigmoid`) — opción A, pendiente de correr

**Contexto (hipótesis mecánica)**: diagnóstico de por qué el sparse pierde contra NBFNet pese a ser
estructuralmente el más cercano: (1) el **segment-softmax normaliza a Σα=1** por nodo destino ⇒ la
agregación es un promedio ponderado convexo que **borra el conteo de caminos de evidencia** (10 caminos
de soporte puntúan igual que 1 con la misma composición media) y es ciega al grado; NBFNet agrega por
SUMA y conserva ambos. (2) PNA entrega múltiples estadísticos, el softmax uno. (3) La agregación
aprendida sobre-ajusta el train graph (firma valid↑/test↓ ya documentada). La opción A ataca (1), la
causa candidata dominante: reemplazar el softmax por **gates sigmoides por arista SIN normalizar**
⇒ suma ponderada aprendida que conserva conteo de caminos y sensibilidad al grado, manteniendo la
selectividad por query (el gate puede apagar aristas irrelevantes, cosa que NBFNet no puede). Nota de
techo: sparse+expander interpola entre sparse y full, ambos < NBFNet ⇒ más conectividad no saca del
intervalo; cambiar la agregación sí cambia la familia de funciones.

**Implementación (HECHA, sin correr aún)**: flag `--attn {softmax,sigmoid,degree}` en `train.py`,
aplicable a `--model sparse` y `sparse_exp` (misma capa). `softmax` = comportamiento actual (default,
sin cambio); `sigmoid` = opción A (α=σ(logit), sin scatter-amax ni denom ⇒ además algo más barato);
`degree` = fallback intermedio softmax × log(1+grado_in) del destino (reinyecta el conteo como scaler
estilo PNA) por si la suma cruda desestabiliza el entrenamiento. En `src/model.py::
SparseRelationalAttentionLayer` (branch en forward), threading por `SparseGraphTransformer` y
`SparseExpanderGraphTransformer`. Smoke test CPU sintético OK: forward/backward de los 3 modos en
ambos modelos, gradiente fluye por `rel_bias`, y sigmoid ≠ softmax con los mismos pesos.

**Experimento planificado (predicción falsable)**: FB15k-237 ind v1, best config sparse
(`--model sparse --attn sigmoid --hidden_dim 64 --drop 0.0 --learning_rate 1e-3 --num_layer 6
--batch_size 16 --max_epochs 20 --seed 42`), head-to-head vs sparse softmax (0.338) y NBFNet (0.459).
Si el conteo de caminos es el gap dominante ⇒ salto grande (zona 0.42+); si queda ~0.34 ⇒ la causa
(1) no era la dominante, pasar a regularización hacia agregación fija. Después réplica transductiva
(sparse 0.403 vs NBFNet ~0.415, gap chico). **Resultado: PENDIENTE.**

---

## 2026-07-20 — sparse_exp lanzado en FB15k-237 TRANSDUCTIVO + hallazgo de costo del expander

**Contexto operacional**: se lanzó `--model sparse_exp` en FB15k-237 transductivo reusando la GPU del
job **798757** (sbatch `sparse_trans_fb237`, 24h, dueño de la asignación vía un proceso sparse SIGSTOP'd
= PID 2609016 en estado `T`). Primero se detuvo el NBFNet transductivo que corría como step `srun
--overlap` **una vez terminada su época 1** (checkpoint guardado: `epoch=1-step=34016.ckpt` + `last.ckpt`,
**valid_mrr 0.376**, reanudable con `--resume_from ./experiments/nbfnet_trans_fb237/last.ckpt`), y su
watchdog. sparse_exp se lanzó como nuevo step `srun --overlap` detached. **Se probó exp_degree=4 y se
cambió a exp_degree=3** (a pedido, deg4 demasiado lento). Config: L4, dim 32, batch 8, lr 1e-3, drop 0.0,
20 ep, seed 42. Log `logs/sparse_exp_trans_fb237.log`, ckpts `experiments/sparse_exp_trans_fb237/`.
**Resultado MRR: PENDIENTE** (época 0 en curso; a ~2:55h/época solo caben ~4 ep en el wall restante).

**Hallazgo de rendimiento (por qué el expander casi DUPLICA el tiempo/época pese a agregar solo +15,6%
de aristas)**: comparación it/s directa log-contra-log, misma config L4/dim32/batch8 en FB15k-237 trans:

| run | aristas | it/s real | época |
|-----|--------:|----------:|------:|
| sparse simple (log 797546) | 558.735 | **6,1** | ~1:36h |
| sparse_exp deg3 | 645.765 (+15,6%) | **3,65** | ~2:55h |

Ratio de tiempo **1,67x**. Si el costo fuera plano por arista, +15,6% daría 5,3 it/s, no 3,65. Despejando
`1.67 = 1 + (87030/558735)·(c_exp/c_real)` ⇒ **cada arista expander cuesta ~4,3x una arista real del KG**.

**Causa mecánica**: NO es el cómputo (idéntico) ni el conteo de aristas. Es el **patrón de acceso a
memoria**. La capa (`SparseRelationalAttentionLayer.forward`) está dominada por gather/scatter indexados
por `src`/`dst` (`q[:,dst]`, `k[:,src]`, `v[:,src]`, `scatter_reduce_(amax)`, `index_add_(1,dst,·)` del
segment-softmax), kernels **memory-bandwidth-bound** cuya velocidad depende de la localidad:
- **Aristas reales del KG**: grafo *scale-free* (clusterizado en hubs) → filas `q/k/v` de los hubs quedan
  en cache L2 y se reusan, accesos coalescidos. Baratas.
- **Aristas expander**: permutación aleatoria d-regular sobre los N=14.505 nodos → `src`/`dst` uniformes
  y aleatorios → cache misses, accesos no coalescidos, sin reuso. Cada gather/scatter cae en fila distinta
  e impredecible (3-8x más caro en GPU; medido ~4,3x).

La lentitud **es la aleatoriedad del expander** — justo lo que le da valor (atajos fuera del horizonte);
no se puede "clusterizar" sin destruir el punto. Palancas: bajar degree (lineal sobre la porción cara),
o (requiere tocar `src/model.py`) meter el expander solo en algunas capas / cachear índices ya en GPU.

**Decisión**: dejar corriendo deg3 (~4 ep en el wall). El costo del expander es esperado y entendido
(memoria, no cómputo). Registrar test_mrr cuando avance para comparar vs sparse (0.4028 dim64/L6, 0.3965
dim32/L4) y NBFNet transductivo.

---

## 2026-07-20 — PLAN: Sparse attention + expander graphs (estilo Exphormer) — `--model sparse_exp`

**Contexto**: añadir un modelo más a la comparación transductiva — el `SparseGraphTransformer` actual
pero aumentado con **aristas expander** (grafo aleatorio d-regular) al estilo Exphormer, para dar
atajos estructurales fuera del horizonte de propagación. Decisión de diseño consultada con el código
de `Exphormer/` (`graphgps/transform/expander_edges.py`, `graphgps/encoder/exp_edge_fixer.py`,
`graphgps/layer/Exphormer.py`).

**Cómo lo hace Exphormer (evidencia)**: las aristas expander se generan como grafo aleatorio d-regular
`(sender, receiver)` **sin relación real**. Pero en la atención **sí** reciben un `edge_attr`: un ÚNICO
embedding aprendido compartido por todas (`self.exp_edge_attr = nn.Embedding(1, dim_edge)`), y un
edge_type dedicado (reales=0, expander=1, virtuales/globales=2). O sea: no van peladas ni con relación
KG real, sino con un **tipo de arista aprendido propio**.

**Decisión de diseño (port a la atención relacional del harness)**: las aristas expander llevan **una
relación sintética reservada `R_exp`** (una fila nueva en la tabla de embeddings de relación,
`num_relation += 1`), usada solo por ellas — no una relación KG real (sería inyectar hechos falsos,
las expander son aleatorias) ni nada (Exphormer sí las tipa). `R_exp` entra en los dos canales
relacionales que ya usa el sparse: el bias escalar `b[head, R_exp]` (rol del edge-type de Exphormer,
estilo Graphormer) y la corrección de valor `g[R_exp]` (composición DistMult). Simétricas → ambas
direcciones con `R_exp`. **Inductive-safe**: `R_exp` se comparte train/test como toda relación y las
aristas son estructurales (sin identidad de entidad).

**Advertencias registradas**:
- **Lista negra #2** ("NO expander en inductivo — metían ruido, Exphormer-Max") es específica de
  INDUCTIVO; en transductivo Exphormer-Max iba bien y los experimentos sparse actuales son
  transductivos ⇒ este modelo NO viola la lista negra *en transductivo*. Probarlo inductivo sí sería
  re-pisar terreno refutado.
- **Tensión con GOALS**: las expander son atajos estructurales aleatorios (conectividad fuera del
  horizonte), no evidencia composicional/relacional (caminos). Hay razón teórica para dudar que ayuden;
  es una prueba empírica de si acortar el diámetro efectivo mejora la propagación en transductivo.

**Implementación (HECHA)**: `src/model.py::SparseExpanderGraphTransformer` + helper
`generate_expander_edges` (port del permutation algorithm de Exphormer, simétrico, sin self-loops,
cacheado por num_nodes). Flags nuevos en `train.py`: `--model sparse_exp` y `--exp_degree` (default 4).
Las tablas relacionales de la capa se dimensionan a `num_relation+2` (self-loop en `num_relation`,
expander en `num_relation+1`). Smoke test CPU OK: forward/backward corre y la fila `R_exp` de `rel_bias`
recibe gradiente (se usa). Comando (best config sparse):
`python train.py --data_path ./data/fb15k-237 --model sparse_exp --exp_degree 4 --num_layer 4
--hidden_dim 32 --num_heads 8 --batch_size 8 --learning_rate 1e-3 --drop 0.0 --seed 42 --max_epochs 20
--checkpoint_save_path ./experiments/sparse_exp_trans_fb237`.
**ESTADO**: lanzado el 2026-07-20 (deg3 tras descartar deg4 por lento); ver entrada de arriba
"sparse_exp lanzado en FB15k-237 TRANSDUCTIVO + hallazgo de costo del expander". Resultado MRR pendiente.

---

## 2026-07-20 — Sparse attention (labeling trick, sin nbfv) en FB15k-237 TRANSDUCTIVO: capacidad grande vs chica

**Contexto**: primer experimento del modelo `SparseGraphTransformer` (`--model sparse`: atención
sparse por adyacencia, labeling trick `x⁰_v = emb(r_q)` si `v==head` else `0`, SIN V de NBFNet) en
FB15k-237 **transductivo** (no inductivo). Dos configs para medir sensibilidad a capacidad vs costo:
grande (dim 64, L6) y chica (dim 32, L4). Seed 42, 20 ep, batch 16, full-filtered. Logs
`logs/sparse_trans_fb237_795838.log` (grande) y `logs/sparse_trans_fb237_797546.log` (chica);
ckpts `experiments/sparse_trans_fb237/` y `experiments/sparse_trans_fb237_4_32/`.

**Resultado (FB15k-237 transductivo, full-filtered)**

| config          | params | valid_mrr | test_mrr | H@1 | H@3 | H@10 | MR | mem GPU | t/época | t test |
|-----------------|-------:|----------:|---------:|----:|----:|-----:|---:|--------:|--------:|-------:|
| **dim 64, L6**  | 440 K | 0.407 | **0.4028** | 0.306 | 0.443 | 0.595 | 116.4 | 39.8 GB | ~3:53 h | 9:12 |
| dim 32, L4      | 126 K | 0.402 | 0.3965 | 0.300 | 0.437 | 0.591 | 128.4 | 14.8 GB | ~1:36 h | 3:24 |

(best-valid: grande epoch 12, chica epoch 19.)

**Análisis (causa mecánica)**
- **La capacidad grande casi no compra métrica pero cuesta ~2.5× en todo.** Δ test_mrr solo +0.0063
  (0.4028 vs 0.3965, +1.6% rel.), y H@10 prácticamente empata (0.595 vs 0.591). Pero el grande cuesta
  **2.4× por época** (3:53 h vs 1:36 h), **2.7× en test** (9:12 vs 3:24) y **2.7× memoria** (39.8 vs
  14.8 GB). El único gap real es MR (116 vs 128). ⇒ **dim 32 / L4 da ~98% del rendimiento a ~40% del
  costo**: mejor relación coste/beneficio en transductivo. El modelo satura rápido en capacidad, la
  señal no está limitada por parámetros.
- **Sigue por debajo del message passing**: NBFNet transductivo en FB15k-237 ronda ~0.415 MRR en
  literatura, el sparse (0.403) queda −0.012 debajo. Consistente con el hallazgo inductivo (la atención
  no supera al MP). Baseline propio (mismo eval full-filtered) **EN CURSO** — ver abajo.

**Decisión**
- Config chica (dim 32, L4) preferida para iterar en transductivo por costo. Confirmación (débil, falta
  baseline propio) de que el sparse tampoco supera al MP en régimen transductivo. Baseline NBFNet
  transductivo lanzado para cerrar la comparación (abajo).

### NBFNet transductivo FB15k-237 — baseline EN CURSO (lanzado 2026-07-20)

**Contexto**: cerrar la comparación con el sparse transductivo con un NBFNet propio, mismo eval
full-filtered. Config alineada a `NBFNet/config/knowledge_graph/fb15k237.yaml` + best config validada
del harness: `--model nbfnet --aggregate pna`, L6, dim 32, distmult, short_cut, layer_norm, dependent,
lr 5e-3, drop 0.1, batch 16, 20 ep, seed 42. Log `logs/nbfnet_trans_fb237_manual.log`, ckpts
`experiments/nbfnet_trans_fb237/`, sbatch encadenable de resume `sbatch_nbfnet_trans.sh`.

**Detalle operacional (reuso de GPU sin perder la asignación)**: no había GPUs libres (cola ~2 días).
Se aprovechó el job 798757 (sparse `_3_32`, no necesario) **suspendiéndolo con SIGSTOP** (proceso vivo
→ SLURM mantiene la asignación; retiene ~12.5 GB VRAM, irrelevante) y lanzando NBFNet como job step
superpuesto (`srun --jobid=798757 --overlap`) en la misma GPU. NBFNet es **compute-bound** (~2.5 h/época
en H100 a batch 16; subir batch no acelera) ⇒ en las ~18 h restantes del job de 24 h solo caben **~6-7
épocas**. Se añadió `--eval_only` a `train.py` y un watchdog (`watchdog_nbf_trans.sh`) que ~1.9 h antes
del wall mata el entrenamiento y corre el test sobre el best-valid ckpt → resultado en
`logs/nbfnet_trans_test.log`. `last.ckpt` permite reanudar a 20 ep en futuros jobs
(`sbatch sbatch_nbfnet_trans.sh`; al llegar a época 20 corre `trainer.test` automático) para el número
final comparable.

**Progreso parcial**: valid_mrr 0.365 tras época 0 (NBFNet converge rápido). **PENDIENTE registrar** el
test_mrr del NBFNet (parcial ~6-7 ep, y final a 20 ep cuando haya GPU) para completar la comparación vs
sparse transductivo (dim64/L6 test 0.4028; dim32/L4 test 0.3965).

---

## 2026-07-08 — source_rw (labeling trick condicionado a la fuente) en full attention, FB15k-237 v1+v2

**Contexto**: probar un labeling trick NATIVO del transformer, condicionado a la query, como
alternativa a los PE node-only ya refutados (RWSE local, LapPE global). Feature por nodo
v = proj([P^1[head,v], …, P^8[head,v]]) sumada a x^0, donde P=D^-1 A (adyacencia simetrizada,
relación ignorada) y la fila head de P^k son las probabilidades de landing de un random walk de k
pasos que arranca en el head de la query. La diferencia clave vs RWSE/LapPE: **es query-conditioned**
(depende del source de la query, no sólo de la estructura del nodo) → en principio es la "evidencia
composicional/relacional fuera del horizonte" que GOALS.md deja abierta, no una coordenada de nodo.
Inductivo-safe (sólo estructura + head, sin identidad de entidad). Implementado en `src/model.py`
(`compute_source_rw`/`source_rw_features`, flags `--use_source_rw --source_rw_dim`). Best config del
full (dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42), apples-to-apples vs full sin PE. Script
`run_source_rw.sh`, logs `logs/source_rw_full_v{1,2}.log`, ckpts `experiments/source_rw_full_v{1,2}/`.

**Resultado (FB15k-237 ind v1+v2, full-filtered)**

| split | full baseline (no PE) | **full + source_rw** | Δ test | con-source_rw: H@1 / H@3 / H@10 / MR | NBFNet |
|-------|----------------------:|---------------------:|-------:|-------------------------------------:|-------:|
| v1 valid | 0.429 | **0.234** |        |                                      | 0.492 |
| v1 test  | 0.375 | **0.313** | **−0.062** | 0.266 / 0.332 / 0.395 / 384          | 0.459 |
| v2 valid | 0.461 | **0.279** |        |                                      | 0.483 |
| v2 test  | 0.491 | **0.363** | **−0.128** | 0.292 / 0.402 / 0.479 / 198          | 0.526 |

**Análisis (causa mecánica)**
- **source_rw es el encoding MÁS dañino probado, y el patrón es distinto al de RWSE/LapPE.** RWSE/LapPE
  degradaban poco y sólo en test (firma de overfit: valid se mantiene o sube, test baja). Aquí
  **valid Y test se derrumban juntos en ambos splits**: valid v1 0.429→0.234 (−0.195), v2 0.461→0.279
  (−0.182). No es sobre-ajuste estructural: es **degradación neta, in-distribution incluida**.
- El canal query-conditioned (proj de las landing probs desde el head) no aporta evidencia nueva; **le
  compite/ahoga al x^0 = emb(r_q) la señal útil** y perjudica hasta el ajuste del train graph. train_loss
  se mantuvo bajo (v1 0.147) ⇒ el modelo ajusta algo, pero el feature extra empeora el ranking incluso
  en validación. Que sea condicionado a la fuente (y no node-only) no lo salva: sigue siendo re-codificar
  estructura dentro del horizonte del MP, ahora con más ruido por depender del head.
- **Muy por debajo de NBFNet en ambos** (v1 0.313 vs 0.459; v2 0.363 vs 0.526). Ni de lejos.

**Decisión**
- source_rw se une a RWSE (local) y LapPE (global) como encoding NO confiable para la atención — de
  hecho el peor. Confirma otra vez que la señal útil no viene de re-codificar la estructura del grafo
  como feature por nodo, ni siquiera condicionada a la query. Candidato a lista negra #7 de CLAUDE.md
  (ampliar de "PE por nodo" a "labeling/encoding estructural derivado de random walks, incl.
  condicionado a la fuente"). Artefactos: `run_source_rw.sh`, `experiments/source_rw_full_v{1,2}/`,
  `logs/source_rw_full_v{1,2}.log`.

---

## 2026-07-08 — LapPE en FB15k-237 ind v2 (réplica del v1): el bump del full NO es robusto

**Contexto**: replicar LapPE en FB15k-237 ind v2 con los mismos 3 modelos y la misma best config
(dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42, `--use_lappe --lappe_dim 16`) para testear la
pregunta que dejó abierta v1: **¿es robusto el +0.035 que LapPE le dio al full attention?** Script
`run_lappe_v2.sh`, logs `logs/lappe_*_v2.log`, ckpts `experiments/lappe_*_v2/`.

**Resultado (FB15k-237 ind v2, full-filtered)**

| modelo             | valid sin→con | **test sin→con** | Δ test v2 | *Δ test v1* | con-LapPE: H@1 / H@10 / MR |
|--------------------|--------------:|-----------------:|----------:|------------:|---------------------------:|
| Full attention     | 0.461 → 0.469 | 0.491 → **0.496** | +0.005 | *+0.035* | 0.397 / 0.681 / 98 |
| Sparse adyacencia  | 0.471 → 0.474 | 0.450 → **0.445** | −0.005 | *−0.003* | 0.358 / 0.592 / 112 |
| Sparse_nbfv (V=NBF)| 0.488 → 0.484 | 0.527 → **0.524** | −0.003 | *+0.013* | 0.424 / 0.687 / 57 |
| *NBFNet (ref)*     | *0.483*       | *0.526*          | —      | —      | *0.416 / 0.727 / 49* |

**Análisis (causa mecánica)**
- **El bump del full NO se replica.** v1 daba +0.035 (0.375→0.410); v2 solo +0.005 (0.491→0.496),
  dentro del ruido. La "primera evidencia a favor de PE global" de v1 era un **artefacto de split**,
  no un mecanismo estructural. Mismo fracaso de robustez que ya se vio con sparse+RWSE (que invertía
  signo entre splits): un efecto que no sobrevive al cambio de split no es una mejora real.
- **Sparse**: negativo-plano en ambos (−0.003 v1, −0.005 v2). La coordenada global no le sirve
  porque solo atiende vecinos; consistente con v1.
- **Sparse_nbfv**: +0.013 (v1) → −0.003 (v2), inconsistente y dentro de ruido; su señal es el V de
  NBFNet, no LapPE.
- **Ninguno supera a NBFNet en v2** (0.526): full+LapPE 0.496 (−0.030), sparse_nbfv+LapPE 0.524
  (−0.002, y por debajo de su propio baseline 0.527). El techo sigue siendo el message passing.

**Decisión**
- **LapPE se une a RWSE como PE no confiable para atención.** Lo que parecía la única pista viva tras
  v1 (PE global ayuda al full) NO resiste la réplica en v2. Ni PE local (RWSE) ni PE global (LapPE)
  dan una mejora robusta que supere a NBFNet. Lista negra #7 de CLAUDE.md actualizada: el matiz
  "PE global ayuda" se retira; queda como efecto dependiente de split. Artefactos: `run_lappe_v2.sh`,
  `experiments/lappe_*_v2/`, `logs/lappe_*_v2.log`.

---

## 2026-07-08 — LapPE (Laplacian positional encoding) en los 3 modelos de atención, FB15k-237 ind v1

**Contexto**: probar un PE ESTRUCTURAL GLOBAL (LapPE: los k=16 autovectores no triviales del
Laplaciano normalizado simétrico L = I − D^-1/2 A D^-1/2, autovalores más chicos), sumado a x^0,
como contraste con RWSE (que es local). LapPE depende solo de la estructura (no de identidad de
nodo) → inductivo-safe, NO viola lista negra #3. Ambigüedad de signo de los autovectores tratada
con sign-flip aleatorio en train (canónico). Implementado en `src/model.py` (`compute_lappe`/
`lappe_features`, flags `--use_lappe --lappe_dim`). Best config de cada modelo (dim 64, drop 0.0,
lr 1e-3, L6, 20 ep, seed 42), apples-to-apples vs baselines sin PE. Script `run_lappe_v1.sh`,
logs `logs/lappe_*_v1.log`, ckpts `experiments/lappe_*_v1/`.

**Resultado (FB15k-237 ind v1, full-filtered)**

| modelo             | valid sin→con | **test sin→con** | Δ test | con-LapPE: H@1 / H@3 / H@10 / MR |
|--------------------|--------------:|-----------------:|-------:|---------------------------------:|
| Full attention     | 0.429 → 0.424 | 0.375 → **0.410** | **+0.035** | 0.351 / 0.441 / 0.524 / 227 |
| Sparse adyacencia  | 0.446 → 0.453 | 0.338 → **0.335** | −0.003 | 0.285 / 0.363 / 0.410 / 197 |
| Sparse_nbfv (V=NBF)| 0.472 → 0.477 | 0.421 → **0.434** | +0.013 | 0.346 / 0.500 / 0.571 / 115 |
| *NBFNet (ref)*     | *0.492*       | *0.459*          | —      | *0.371 / 0.520 / 0.605 / 117* |

**Análisis (causa mecánica)**
- **LapPE es el PRIMER encoding estructural que ayuda — y solo al full attention** (+0.035, el
  mejor resultado de full attention hasta ahora: 0.375 base, 0.366 con RWSE, **0.410 con LapPE**).
  Contraste directo con RWSE, que dañaba el full (−0.009).
- **Por qué ayuda al full y no a los sparse**: LapPE es GLOBAL (coordenadas espectrales de todo el
  grafo), RWSE es LOCAL (diag(P^k), dentro del horizonte de k saltos). El full attention es all-pairs
  pero no tenía NINGUNA noción de posición/distancia global; LapPE le da un sistema de coordenadas
  para distinguir nodos lejanos. Eso es señal de FUERA del horizonte de propagación → primera
  evidencia empírica a favor de la única dirección que GOALS.md deja abierta.
- **Sparse**: plano con firma de overfit. valid sube (0.446→0.453) pero test queda igual
  (0.338→0.335): como solo atiende a vecinos, la coordenada global no le sirve para propagar mejor,
  solo le da más con qué sobre-ajustar el train graph. Mismo patrón estructural de siempre.
- **Sparse_nbfv**: +0.013 chico. Su columna vertebral es el V de NBFNet (MR 115 ≈ NBFNet 117);
  LapPE aporta un poco pero no rompe el techo.
- **Ninguno supera a NBFNet (0.459).** Full+LapPE (0.410) sigue −0.049 debajo. El signo positivo es
  real e interpretable (info global espectral), pero no cierra el gap con message passing.

**Decisión**
- LapPE NO es otro RWSE: matiza la conclusión "todo PE es redundante". PE LOCAL (RWSE) sí es
  redundante con el MP; PE GLOBAL (LapPE) aporta señal real de fuera del horizonte al full attention,
  aunque insuficiente para ganarle a NBFNet en v1. Lista negra #7 de CLAUDE.md reformulada (ya no
  prohíbe todo PE; distingue local redundante vs global útil-pero-insuficiente).
- Pista abierta a seguir: (i) ¿es robusto el +0.035 del full en v2 / WN18RR?; (ii) ¿escala el
  efecto global con más lappe_dim o capacidad del full? Artefactos: `run_lappe_v1.sh`,
  `experiments/lappe_*_v1/`, `logs/lappe_*_v1.log`.

---

## 2026-06-28 — RWSE (random-walk structural encoding) en los 3 modelos de atención, FB15k-237 ind v1

**Contexto**: probar si añadir un encoding ESTRUCTURAL por nodo (RWSE: diag(P^k) de la
adyacencia relation-agnostic, k=1..16) a x^0 mejora los modelos de atención. RWSE es
puramente estructural → inductivo-safe (depende de la estructura local, no de identidad de
nodo), NO viola lista negra #3. Implementado en `src/model.py` (`compute_rwse`/`rwse_features`,
flags `--use_rwse --rwse_dim`). Best config de cada modelo (dim 64, drop 0.0, lr 1e-3, L6,
20 ep, seed 42). Apples-to-apples vs los baselines sin RWSE. Scripts `run_rwse_v1.sh`, logs
`logs/rwse_*_v1.log`, ckpts `experiments/rwse_*_v1/`.

**Resultado (FB15k-237 ind v1, full-filtered)**

| modelo            | valid sin→con | **test sin→con** | Δ test | con-RWSE: H@1 / H@3 / H@10 / MR |
|-------------------|--------------:|-----------------:|-------:|--------------------------------:|
| Full attention    | 0.429 → 0.434 | 0.375 → **0.366** | −0.009 | 0.310 / 0.407 / 0.459 / 227 |
| Sparse adyacencia | 0.446 → 0.438 | 0.338 → **0.292** | −0.046 | 0.227 / 0.337 / 0.376 / 211 |
| Sparse_nbfv (V=NBF)| 0.472 → 0.474 | 0.421 → **0.424** | +0.003 | 0.332 / 0.480 / 0.593 / 103 |
| *NBFNet (ref)*    | *0.492*       | *0.459*          | —      | *0.371 / 0.520 / 0.605 / 117* |

**Análisis (causa mecánica)**
- **RWSE no ayuda a ninguno; el techo sigue siendo NBFNet (0.459).**
- **Full**: neutral-negativo. valid sube un pelo (0.429→0.434) pero test baja (0.375→0.366).
  El encoding estructural no aporta evidencia nueva; lo poco que añade no transfiere.
- **Sparse adyacencia**: RWSE lo **empeora claramente** (test −0.046). Agrava su firma de
  overfit estructural: valid se mantiene ~0.44 mientras test cae fuerte (0.338→0.292) → más
  features estructurales para sobre-ajustar el train graph, peor transferencia al grafo
  inductivo disjunto. Mismo patrón que ya tenía, amplificado.
- **Sparse_nbfv**: plano (+0.003, dentro de ruido). Su señal viene del V de NBFNet; sumar
  RWSE encima no mueve nada.

**Decisión**
- RWSE descartado como mejora para atención en FB15k-237 v1: es estructura DENTRO del horizonte
  de propagación que el message passing ya captura → redundante (full/sparse_nbfv) o dañina
  (sparse, abre más margen de overfit). Consistente con GOALS.md: la señal útil debe venir de
  FUERA del horizonte de propagación, no de re-codificar estructura local. Añadido a lista
  negra de CLAUDE.md. Artefactos: `run_rwse_v1.sh`, `experiments/rwse_*_v1/`, `logs/rwse_*_v1.log`.

### Réplica en FB15k-237 ind v2 (misma best config, mismos 3 modelos)

**Resultado (FB15k-237 ind v2, full-filtered)**

| modelo            | valid sin→con | **test sin→con** | Δ test | con-RWSE: H@1 / H@10 / MR |
|-------------------|--------------:|-----------------:|-------:|--------------------------:|
| Full attention    | 0.461 → 0.455 | 0.491 → **0.477** | −0.014 | 0.382 / 0.652 / 154 |
| Sparse adyacencia | 0.471 → 0.465 | 0.450 → **0.472** | +0.022 | 0.382 / 0.644 / 109 |
| Sparse_nbfv (V=NBF)| 0.488 → 0.486 | 0.527 → **0.529** | +0.002 | 0.427 / 0.703 / 60 |
| *NBFNet (ref)*    | *0.483*       | *0.526*          | —      | *0.416 / 0.727 / 49* |

**Lectura cruzada v1 vs v2 (Δ test_mrr por RWSE)**

| modelo      | Δ v1   | Δ v2   |
|-------------|-------:|-------:|
| Full        | −0.009 | −0.014 |
| Sparse      | −0.046 | +0.022 |
| Sparse_nbfv | +0.003 | +0.002 |

**Análisis (causa mecánica)**
- **RWSE es inconsistente y nunca supera a NBFNet en ningún split.**
- **Full**: neutral-negativo en ambos (−0.009, −0.014). El encoding no aporta evidencia nueva.
- **Sparse**: el signo se **invierte** entre splits (daña −0.046 en v1, ayuda +0.022 en v2).
  Efecto NO robusto / dependiente del split, no un mecanismo real: en v2 la estructura local
  de RWSE casualmente transfiere, en v1 sobre-ajusta. Señal poco confiable.
- **Sparse_nbfv**: plano en ambos (+0.003, +0.002, ruido); su señal es el V de NBFNet. En v2
  el mejor (sparse_nbfv+RWSE 0.529) solo EMPATA a NBFNet (0.526), heredando su V — no lo supera.

**Decisión**
- Confirma v1 en v2: RWSE no es una mejora confiable para atención y no rompe el techo del
  message passing. Línea cerrada. Artefactos: `run_rwse_v2.sh`, `experiments/rwse_*_v2/`,
  `logs/rwse_*_v2.log`.

### Réplica en WN18RR ind v1 (corrido 2026-06-29, registrado 2026-07-08)

**Contexto**: los runs terminaron el 29-jun pero quedaron sin registrar (aparecían como
"pendiente/corriendo"). Misma best config (dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42).
Script `run_rwse_wn_v1.sh`, logs `logs/rwse_*_wn_v1.log`, ckpts `experiments/rwse_*_wn_v1/`.

**Resultado (WN18RR ind v1, full-filtered)**

| modelo            | valid (con) | **test sin→con** | Δ test | con-RWSE: H@1 / H@10 / MR |
|-------------------|------------:|-----------------:|-------:|--------------------------:|
| Full attention    | 0.514       | 0.673 → **0.670** | −0.003 | 0.638 / 0.734 / 120 |
| Sparse adyacencia | 0.535       | 0.738 → **0.677** | −0.061 | 0.628 / 0.769 / 56 |
| Sparse_nbfv (V=NBF)| 0.557      | 0.740 → **0.728** | −0.012 | 0.684 / 0.798 / 44 |
| *NBFNet (ref)*    | *0.578*     | *0.740*          | —      | *0.689 / 0.822 / 30* |

**Análisis (causa mecánica)**
- **RWSE daña los 3 en WN18RR también; ninguno supera a NBFNet (0.740).**
- **Sparse: el más dañado otra vez (−0.061, 0.738→0.677)** — su firma de overfit estructural
  (más features estructurales → sobre-ajusta el train graph, transfiere peor). Mismo patrón que
  FB15k-237 v1 (−0.046). El sparse pasa de EMPATAR a NBFNet sin PE (0.738) a quedar claramente
  debajo con RWSE.
- **Full (−0.003) y sparse_nbfv (−0.012)**: neutral-negativos, el encoding local no aporta.

**Decisión**
- Cierra RWSE en los 3 datasets (FB15k-237 v1+v2, WN18RR v1). Conclusión uniforme: RWSE nunca
  supera a NBFNet y perjudica al sparse en todos los splits salvo v2 (donde el signo casualmente
  se invierte). Confirma lista negra #7. Artefactos: `run_rwse_wn_v1.sh`, `experiments/rwse_*_wn_v1/`,
  `logs/rwse_*_wn_v1.log`.

---

## 2026-06-23 — Los 4 modelos en FB15k-237 ind v2 (best config de cada uno)

**Contexto**: a pedido del usuario, replicar el head-to-head de v1 ahora en FB15k-237 ind
v2. Pregunta explícita: ¿corre el full attention en v2? **Sí** — grafo chico para O(N²):
train graph N≈2608 nodos, test graph (`_ind`) N≈1660. Cada modelo con su mejor config:
NBFNet dim 32 / drop 0.1 / lr 5e-3 / pna; full/sparse/sparse_nbfv dim 64 / drop 0.0 / lr
1e-3. Todos L6, batch 16, 20 ep, seed 42. Script `run_v2_all.sh`, logs `experiments/v2_*.log`,
ckpts `experiments/v2_*/`.

**Resultado (FB15k-237 ind v2, full-filtered)**

| test     | NBFNet | full attn | sparse adj | sparse_nbfv (V=NBF) |
|----------|-------:|----------:|-----------:|--------------------:|
| valid_mrr| 0.483  | 0.461     | 0.471      | **0.488**           |
| test_mrr | 0.526  | 0.491     | 0.450      | **0.527**           |
| Hits@1   | 0.416  | 0.389     | 0.369      | **0.431**           |
| Hits@3   | **0.595** | 0.544  | 0.494      | 0.586               |
| Hits@10  | **0.727** | 0.686  | 0.586      | 0.690               |
| MR       | **49.3** | 76.6    | 112.6      | 54.5                |

Uso de memoria GPU (H100): full attn **31 GB**, sparse_nbfv 7.1 GB, sparse 3.6 GB, NBFNet
2.0 GB. El full attn cabe de sobra; solo es el más lento (~3.5 it/s, 3.5 min/época).

**Análisis (causa mecánica)**
- **Se repite el patrón de v1**: ni full attn ni sparse-adyacencia superan a NBFNet.
  Orden test: sparse_nbfv (0.527) ≈ NBFNet (0.526) > full (0.491) > sparse adj (0.450).
- **Sparse-adyacencia sigue siendo el peor** (0.450) y reproduce la firma de sobre-ajuste
  estructural: valid (0.471) > su propio test (0.450) → ajusta mejor el train graph y
  transfiere peor al grafo inductivo disjunto. Mismo eco que v1 y que los expander.
- **sparse_nbfv empata/roza a NBFNet** (0.527 vs 0.526, mejor Hits@1 0.431): como en
  WN18RR, su V es el de NBFNet y la reponderación por adyacencia no degrada aquí. Sigue sin
  **superar** de forma significativa al MP (Δ test_mrr +0.001, dentro de ruido).
- NBFNet sube a 0.526 (de 0.459 en v1): v2 es un split más grande/fácil.

**Decisión**
- Confirma GOALS.md también en v2: ninguna atención supera de forma clara a NBFNet. El único
  candidato que empata (sparse_nbfv) hereda el V del message passing por construcción, no
  aporta señal nueva. Artefactos: `experiments/v2_{nbfnet,rfat,sparse,sparse_nbfv}/`,
  `experiments/v2_*.log`, `run_v2_all.sh`.

---

## 2026-06-22 — Revisión: confirmación de logs + recuperación del historial CPA/A1

**Contexto**: lectura de CLAUDE.md y los logs/documentos del proyecto. Sin correr nada
nuevo; verificación cruzada de números y rescate de resultados previos relevantes.

**Resultado**
- Los `.log` de `experiments/` (gt/nbf/sweep_*/sparse_*) **coinciden exactamente** con la
  tabla head-to-head de CLAUDE.md. Implementación de NBFNet validada (test_mrr 0.459, en
  rango de literatura ~0.42–0.46).
- Se recuperó del `SESSION_LOG.md` (proyecto previo, legacy) que la idea **CPA
  (Compositional Pivot Attention)** y su sucesora **A1 (candidate-set attention)** YA
  fueron probadas y **refutadas** (R12–R15). Ver entrada histórica abajo.

**Análisis**
- La pregunta "¿alimentar Q/K/V de la atención desde NBFNet?" se mapea a tres casos:
  (V = NBFNet → KnowFormer, lista negra #6, decorativa); (Q/K = NBFNet → línea Structural
  Query-PE, load-bearing↑ pero MRR plano); (valores composicionales de NBFNet fuera del
  horizonte → **eso ES CPA, ya refutado**).

**Decisión**
- Crear `GOALS.md` y este `SESSION_NOTES.md`.
- Pendiente (decisión usuario): añadir CPA/A1 a la lista negra de CLAUDE.md y/o guardar
  memoria con la causa mecánica de CPA (corridas NBFNet ancladas colapsan a cos 0.992).

### Experimento nuevo — sparse attention con V desde stream NBFNet (lista negra #6, a pedido)

**Contexto**: a pedido del usuario, para completar la tabla. `SparseNBFValueTransformer`
(`src/model.py`, `--model sparse_nbfv`): atención sparse por adyacencia donde Q,K salen del
stream de atención (labeling trick) pero **V = representaciones de nodo de un NBFNet corrido
aparte** (`NBFNet.encode()`, h^L_{u->v}). Ambos streams entrenan end-to-end. Recrea a
propósito el patrón refutado de KnowFormer (V-stream alimentando la atención). Config B
EXACTA del mejor sparse adyacencia (dim 64, drop 0.0, L6, lr 1e-3, 20 ep, seed 42) para
comparación apples-to-apples vs la fila 0.338.

**Resultado (FB15k-237 ind v1, best ckpt epoch 7)**

| test     | NBFNet | full attn | sparse adj | **sparse_nbfv (V=NBF)** |
|----------|-------:|----------:|-----------:|------------------------:|
| valid_mrr| 0.492  | 0.429     | 0.446      | **0.472**               |
| test_mrr | **0.459** | 0.375  | 0.338      | **0.421**               |
| Hits@1   | 0.371  | —         | —          | 0.339                   |
| Hits@3   | 0.520  | —         | —          | 0.476                   |
| Hits@10  | 0.605  | 0.471     | 0.451      | 0.554                   |
| MR       | 117    | —         | —          | 118                     |

**Análisis (causa mecánica)**
- Es el **mejor de todas las variantes de atención** (0.421 > full 0.375 > sparse 0.338):
  hereda casi toda la señal de NBFNet a través de V. MR ~118 ≈ NBFNet (117) confirma que la
  columna vertebral es el V del message passing.
- Pero **sigue −0.038 MRR bajo NBFNet puro (0.459)**: la atención encima del V de NBFNet es
  **net-ligeramente-destructiva**. Reponderar por adyacencia un V que ya resuelve la query
  no añade nada y cuesta un poco. Confirma lista negra #6 / diagnóstico KnowFormer (atención
  sobre V-stream de MP = redundante). No supera al MP, lo degrada.

**Decisión**
- Línea "V desde NBFNet en la atención" cerrada: redundante por construcción, como se
  predijo. Queda en la tabla como evidencia. Artefactos: `SparseNBFValueTransformer` +
  `--model sparse_nbfv`, `experiments/sparse_nbfv_B/`, `logs/sparse_nbfv_B.log`.

---

### Experimento nuevo — los 4 modelos en WN18RR ind v1 (best config de cada uno)

**Contexto**: a pedido del usuario, replicar el head-to-head en WN18RR ind v1. Cada modelo
con su mejor config: NBFNet dim 32 / drop 0.1 / lr 5e-3 / pna (= config NBFNet repo
wn18rr); full/sparse/sparse_nbfv dim 64 / drop 0.0 / lr 1e-3. Todos L6, batch 16, 20 ep,
seed 42. Script `run_wn18rr_v1.sh`, logs `logs/wn_*_v1.log`.

**Resultado (WN18RR ind v1, full-filtered)**

| test     | NBFNet | full attn | sparse adj | sparse_nbfv (V=NBF) |
|----------|-------:|----------:|-----------:|--------------------:|
| valid_mrr| 0.578  | 0.521     | 0.567      | 0.578               |
| test_mrr | **0.740** | 0.673  | 0.738      | **0.740**           |
| Hits@1   | 0.689  | 0.638     | 0.686      | 0.691               |
| Hits@3   | 0.774  | 0.691     | 0.774      | 0.766               |
| Hits@10  | 0.822  | 0.739     | 0.819      | 0.832               |
| MR       | 29.6   | 97.4      | 26.7       | 29.5                |

**Análisis**
- NBFNet 0.740 ≈ literatura (paper NBFNet ~0.741 wn18rr v1) → baseline validado.
- **Cuadro INVERTIDO respecto a FB15k-237**: aquí la **full attention es la PEOR** (0.673,
  −0.067) — el mecanismo global daña en WN18RR (grafo local/jerárquico; eco del log previo
  "WN18RR selecciona en contra de cualquier mecanismo global").
- **Sparse adyacencia (0.738) ≈ NBFNet** (−0.002): restringir a vecinos cierra el gap, al
  revés que en FB15k-237 (donde sparse era el peor). En WN18RR lo local es lo que importa.
- **sparse_nbfv (0.740) EMPATA NBFNet** (test_mrr idéntico, H@10 0.832 > 0.822): su V es
  NBFNet y la reponderación por adyacencia no degrada en este régimen.

**Síntesis cruzada (FB15k-237 + WN18RR)**: ninguna variante de atención **supera** a NBFNet
en ningún dataset. FB15k-237 (composicional): todas pierden, sparse el peor. WN18RR
(local): full pierde, sparse empata. El techo es el message passing en ambos regímenes.

**Decisión**
- Confirma GOALS.md: la dirección "atención supera a NBFNet" sigue sin evidencia a favor en
  ningún régimen. Artefactos: `experiments/wn_{nbf,full,sparse,sparse_nbfv}_v1/`,
  `logs/wn_*_v1.log`, `run_wn18rr_v1.sh`.

---

## 2026-06-21 — Reinicio desde cero: RFAT (full) vs NBFNet vs Sparse en FB15k-237 ind v1

**Contexto**: arranque del proyecto nuevo (RFAT, `src/model.py` escrito desde cero, NO
KnowFormer). Sanity check de `GOALS.md`: ¿atención full supera a NBFNet?

**Resultado (FB15k-237 ind v1, full-filtered, seed 42, 6 capas, dim 32, lr 5e-3, 20 ep)**

| test     | NBFNet (pna) | RFAT full (base) | RFAT full (best) | Sparse adyacencia |
|----------|-------------:|-----------------:|-----------------:|------------------:|
| valid_mrr| 0.492        | —                | 0.429            | 0.446             |
| test_mrr | **0.459**    | 0.318            | 0.375            | 0.338             |
| Hits@1   | **0.371**    | 0.271            | —                | —                 |
| Hits@3   | **0.520**    | 0.341            | —                | —                 |
| Hits@10  | **0.605**    | 0.405            | 0.471            | 0.451             |
| MR       | **117**      | 273              | —                | —                 |

Sweep RFAT (descarta subentrenamiento): base(d32,dp.1,lr5e-3)=0.318 → A(d64,L6)=0.353 →
C(d64,L3)=0.363 → **B(d64,L6,lr1e-3)=0.375** (mejor). Sparse sweep: best test_mrr 0.338.

**Orden en test: NBFNet (0.459) > full attention (0.375) > sparse adyacencia (0.338).**

**Análisis (causa mecánica)**
- El gap es **estructural, no de entrenamiento**: el tuning solo movió RFAT 0.318→0.375;
  sigue −0.084 MRR bajo NBFNet (−22% rel.), no cierra con hiperparámetros razonables. Lo
  que más ayudó fue estabilidad de optimización (lr 1e-3), no capacidad.
- El **sparse generaliza PEOR que el denso** pese a parecerse más a NBFNet: valid_mrr más
  alto (0.446 > 0.429) pero test_mrr más bajo (0.338 < 0.375) → ajusta mejor el train
  graph y **transfiere peor** al grafo inductivo disjunto. Eco del fracaso de los expander
  en inductivo: agregación blanda aprendida sobre-ajusta la estructura del train.
- Confirma `transformer_vs_nbfnet.tex`: la atención reimplementa peor el message passing.

**Decisión**
- Dirección "graph transformer (full o sparse) como reemplazo de NBFNet" **descartada**.
- Único margen abierto: inyectar evidencia fuera del horizonte de propagación, no
  recombinar/reponderar lo que el MP ya entrega.
- Config RFAT recomendada si se reusa: `--hidden_dim 64 --drop 0.0 --learning_rate 1e-3`.

---

## HISTÓRICO (proyecto previo, KnowFormer — de `SESSION_LOG.md`, contexto, NO repetir)

Estos resultados son del proyecto anterior (ya descartado por construir sobre KnowFormer),
pero **refutan ideas que podrían re-proponerse**. Conservados como advertencia.

### R12 (2026-06-14) — CPA (Compositional Pivot Attention): NEGATIVO LIMPIO
- **Idea**: Edge Transformer restringido a fila h × k pivotes. 1ª corrida V-RMPNN anclada
  en h → x_{h,v}; top-k pivotes u; 2ª corrida anclada en cada u → x_{u,t}; atención
  out(t)=Σ_u softmax(β(x_{h,u}))·g(x_{h,u}⊙x_{u,t}).
- **Resultado**: TODAS las celdas < baseline (0.4626). cpa_k8=0.4313 (−0.031). Control
  invertido: pivotes **aleatorios** (0.4538) > selección dirigida (0.4313).
- **Causa mecánica decisiva**: las corridas NBFNet ancladas en pivotes distintos son casi
  idénticas entre sí (**cos = 0.992**); el ancla one-hot se lava tras 3 capas de MP. Solo
  el **7.4%** de la energía composicional varía entre pivotes → CPA = un canal promediado,
  no k evidencias distintas. La premisa "NBFNet es la fila h del Edge Transformer" es solo
  débilmente cierta.
- Fix `--cpa_center` (R12c): valid↑ sobre baseline (señal real in-distribution) pero NO
  transfiere a test y dirigido≈aleatorio sigue fallando. **Kill-criterion cumplido.**

### R14 (2026-06-17) — A1-v0 candidate-set attention (sin bias par): NEGATIVO
- Reranker listwise sobre top-K. El reordenamiento **daña −0.08**; empeora 99 queries,
  mejora 2. El readout pointwise ya extrae la mejor señal de las features de nodo.

### R15 (2026-06-17) — A1 con bias por par estructural: señal real pero techo ~baseline
- Bias por par (adyacencia + vecinos comunes) **pasa gate de capacidad** (pair > shuffle):
  señal no-redundante REAL. Pero gate MRR falla: cuello de botella = pick accuracy 44%
  (no calibración). Techo calculado ~baseline aun con reranking perfecto-no-dañino.

**Tema recurrente R9/R12/R14/R15**: toda señal dada a la atención en KGC es redundante con
lo que el readout/MP ya extrae, o no transfiere al grafo inductivo. Mismo hallazgo que el
reinicio 2026-06-21.
