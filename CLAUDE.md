# CLAUDE.md — Attention (Graph Transformer para KGC, desde cero)

Briefing operacional para sesiones de Claude Code. Conciso a propósito.

---

## Qué es este proyecto (reinicio 2026-06-21)

Proyecto nuevo, **desde cero**. Dos proyectos previos quedaron botados y NO se
construye sobre ellos:

- **Exphormer-Max**: adaptar Exphormer (expander graphs) a KGC. Bien en transductivo,
  **pésimo en inductivo** (los expander graphs metían ruido). Línea descartada.
- **Knowformer-Expander**: construir sobre KnowFormer. Tras meses se concluyó que su
  atención es **decorativa** (todo el peso está en el V-RMPNN stream ≈ NBFNet).
  Confuso e improductivo. Descartado.

**Pregunta de esta etapa (sanity check antes de invertir en sparse/lineal):**

> Un **Graph Transformer con atención FULL (densa, all-pairs O(N²))** —el techo de
> expresividad de cualquier atención sparse— ¿supera a **NBFNet** en **FB15k-237
> inductivo v1**? Si la versión densa **no** gana, una versión sparse (que es una
> restricción de la densa) no tiene caso, y habría que repensar todo el enfoque de
> "Graph Transformer para KGC".

Marco teórico que justifica el diseño: `transformer_vs_nbfnet.tex`. Resumen:
- **Caso (a)** transformer ciego a la estructura (atención sobre embeddings de nodo sin
  aristas): **rompe la inductividad** (no hay embeddings de entidad que transferir al
  grafo de test disjunto). Inviable → NO hacer esto.
- **Caso (b)** *graph* transformer con labeling trick + adyacencia relacional: **puede
  igualar a NBFNet en principio**, pero al usar aristas+relaciones+fuente está
  "reimplementando message passing con agregación aprendida". Este proyecto mide
  empíricamente si esa flexibilidad extra **mejora** o no.

---

## Arquitectura actual: RFAT (Relational Full-Attention Graph Transformer)

`src/model.py` (NUEVO, escrito desde cero — NO es KnowFormer). Una sola torre de
atención. **Sin V-RMPNN, sin QK-RMPNN, sin RSPMM kernel.**

- **Inductivo puro**: sin embeddings de entidad, sin positional encoding de nodo. Lo
  único compartido train/test son embeddings de **relación** (las relaciones sí se ven).
- **Labeling trick** (NBFNet): `x⁰_v = emb(r_q)` si `v == head`, si no `0`.
- **Cada capa** (multi-head, pre-LN), atención **densa all-pairs**
  `softmax(QKᵀ/√d + b_rel)` con:
  - (i) **bias escalar relacional** `b[head, rel]` en pares conectados por arista
    (estilo Graphormer: marca quién es vecino y por qué relación).
  - (ii) **corrección de valor relacional** `Σ_edges α·(V_w ⊙ g[rel])` (composición
    estilo DistMult; aporta el razonamiento por caminos).
  - residual + FFN.
- **Readout** puntual `MLP(x^L_v) → score (B, N)`. **Loss = CE de grafo completo**.
- La arista de la query (y su reversa) se quita del grafo en train (`graph_mask`).

El nodo destino `dst` agrega de `src` (arista `src --rel--> dst`), igual que el message
passing de NBFNet para predecir cola.

---

## Repositorio

```
Attention/
  src/
    model.py     # NUEVO. GraphTransformer + RelationalAttentionLayer. Editar aquí.
    data.py      # CONSERVADO de KnowFormer. Loaders KG transductivo + inductivo.
    metric.py    # CONSERVADO. MR / MRR / Hits.
    rspmm/       # kernel CUDA de KnowFormer. NO se usa en este proyecto.
  data/          # datasets: wn18rr, fb15k-237, nell-995, yago3-10, inductive/
  train.py       # NUEVO. Entry point limpio (argparse + PL module + datamodule).
  NBFNet/        # repo ORIGINAL de NBFNet (referencia/baseline). No modificar.
  Exphormer/     # repo ORIGINAL de Exphormer (referencia). No se usa.
  transformer_vs_nbfnet.tex   # análisis teórico que justifica el diseño. LEER.
  manuscrito_candidatura.md   # propuesta de tesis (referencia).
  CLAUDE.md
```

**Legacy (de KnowFormer, NO usar)**: `main.py`, `lightning.py` importan el viejo
`Knowformer` que ya no existe en `model.py` → quedan rotos a propósito. El entry point
nuevo es `train.py`. Los `.md`/`.sh` heredados (SESSION_LOG.md, PLAN_*.md, run_*.sh,
sbatch_*.sh) son del proyecto previo y NO son guía para este.

---

## Entorno

- **Python**: `/nfs_ssd/mojeda_imfd/miniconda3/envs/knowformer/bin/python` (env reusado;
  torch 2.1.0+cu121, PyG 2.4.0, pytorch_lightning 1.9.1, torchmetrics 0.11.4).
- **Setup obligatorio**: `source env.sh` (PYTHONNOUSERSITE=1, gnu12, cuda/12.6, PATH al env).
- **NO usar `conda run`** — la ruta directa al binario funciona.
- El modelo nuevo **no compila kernels** (no usa rspmm). Pure PyTorch.
- Cluster: **H100 NVL 95GB** en UAI. Atención densa O(N²) con N≈1600 es trivial.
- **No se habilita `torch.use_deterministic_algorithms(True)`** en `train.py`: el modelo
  usa `index_add_` (no determinista en CUDA), aceptable para baseline.

## Comandos típicos

```bash
source env.sh && PY=$(which python)

# Run principal: RFAT en FB15k-237 inductivo v1 (recipe alineado a NBFNet config).
$PY train.py --data_path ./data/inductive/fb15k-237_v1 \
  --num_layer 6 --hidden_dim 32 --num_heads 8 \
  --batch_size 16 --test_batch_size 16 --max_epochs 20 \
  --learning_rate 5e-3 --weight_decay 1e-4 --drop 0.1 --seed 42 \
  --checkpoint_save_path ./experiments/gt_fb15k237_v1

# Smoke test: --num_layer 2 --batch_size 8 --max_epochs 1
```

`is_inductive(data_path)` detecta inductivo por el sufijo `_vN` del nombre del dataset.
El split de test usa el grafo **disjunto** `<dataset>_ind`; val usa el train graph.

---

## Baseline de comparación: NBFNet (implementado en este harness)

`src/model.py::NBFNet` — reimplementación fiel (message passing DistMult dependiente de
query + PNA + short-cut + layer-norm), MISMA data y MISMO eval full-filtered que el RFAT.
Correr con `--model nbfnet --aggregate pna`. Alineado a `NBFNet/config/inductive/fb15k237.yaml`.

### Resultado head-to-head (FB15k-237 ind v1, full-filtered, seed 42, 6 capas, dim 32, lr 5e-3, 20 ep)

| test           | RFAT (full attention) | NBFNet (pna) |
|----------------|----------------------:|-------------:|
| MRR            | 0.318                 | **0.459**    |
| Hits@1         | 0.271                 | **0.371**    |
| Hits@3         | 0.341                 | **0.520**    |
| Hits@10        | 0.405                 | **0.605**    |
| MR             | 273                   | **117**      |

**Hallazgo (2026-06-21):** el Graph Transformer full attention —el techo de expresividad
de cualquier atención sparse— pierde **decisivamente** contra NBFNet (+0.14 MRR, +44%
relativo). Confirma empíricamente `transformer_vs_nbfnet.tex`: la atención reimplementa
peor el message passing y no lo supera. **=> Una versión sparse no tiene caso a menos que
cambie algo fundamental** (p.ej. inyectar evidencia fuera del horizonte de propagación).
Nuestro NBFNet (0.459) cae en el rango de literatura (~0.42–0.46), implementación validada.

### Sweep RFAT (20 ep, confirma que el gap es estructural)

Probadas 3 configs para descartar subentrenamiento (dim 64, sin dropout):

| test    | base(d32,dp.1,lr5e-3) | A(d64,L6,lr5e-3) | C(d64,L3,lr5e-3) | **B(d64,L6,lr1e-3)** | NBFNet |
|---------|----------------------:|-----------------:|-----------------:|---------------------:|-------:|
| MRR     | 0.318                 | 0.353            | 0.363            | **0.375**            | **0.459** |
| Hits@10 | 0.405                 | 0.429            | 0.441            | **0.471**            | **0.605** |

El tuning subió el RFAT 0.318 → **0.375** (mejor: dim 64, drop 0.0, lr **1e-3** — la
estabilidad de optimización fue lo que más ayudó), pero **sigue −0.084 MRR bajo NBFNet
(0.459)**, un −22% relativo que NO cierra con tuning razonable. **El gap es estructural,
no un artefacto de entrenamiento.** Conclusión confirmada: la atención full no supera al
message passing aquí ⇒ atención sparse descartada para esta dirección.

Config RFAT recomendada si se vuelve a usar: `--hidden_dim 64 --drop 0.0 --learning_rate 1e-3`.

### Sparse Graph Transformer (atención por adyacencia) — `--model sparse`

`src/model.py::SparseGraphTransformer`: cada nodo atiende SOLO a sus vecinos del grafo
(segment-softmax sobre aristas entrantes + self-loop), no a los N. Es "NBFNet con
agregación aprendida por atención". Mismo sweep A/B/C (dim 64, drop 0.0).

| test (best config)  | NBFNet | RFAT denso (full) | **Sparse adyacencia** |
|---------------------|-------:|------------------:|----------------------:|
| valid_mrr           | 0.492  | 0.429             | 0.446                 |
| **test_mrr**        | **0.459** | 0.375          | **0.338**             |
| Hits@10             | 0.605  | 0.471             | 0.451                 |

**Hallazgo (2026-06-21):** el sparse es el **PEOR** de los tres en test, aunque es
estructuralmente el más cercano a NBFNet. Clave: el sparse tiene **valid_mrr MÁS alto que
el denso (0.446 > 0.429) pero test_mrr MÁS bajo (0.338 < 0.375)** ⇒ ajusta mejor el train
graph y **transfiere peor al grafo inductivo disjunto**. Eco del fracaso de los expander
en inductivo (proyecto Exphormer-Max): la agregación blanda aprendida por atención
sobre-ajusta la estructura del train y generaliza peor que la agregación FIJA de NBFNet.

**Orden en test inductivo: NBFNet (0.459) > full attention (0.375) > sparse adyacencia
(0.338).** Ni la atención full ni la sparse superan a NBFNet; la sparse encima generaliza
peor. La dirección "graph transformer (full o sparse) para KGC" queda descartada como
reemplazo de NBFNet; el único margen es inyectar evidencia fuera del horizonte de
propagación, no recombinar/reponderar lo que el message passing ya entrega.

---

## Convenciones

- Reportar siempre **valid_mrr** y **test_mrr** (+ hits@1/3/10, mr). Model selection por
  `valid_mrr` (val = train graph); test sobre el grafo inductivo disjunto.
- **Una sola arquitectura** debería servir transductivo e inductivo; solo cambian
  hiperparámetros (requisito metodológico del manuscrito).
- Si train↑ y test↓ con magnitudes grandes → overfit estructural (el mal del proyecto
  previo). Parar y analizar.

## Lista negra (refutado / NO repetir)

1. **NO construir sobre KnowFormer** (V-RMPNN/QK-RMPNN/RSPMM). Atención decorativa, meses perdidos.
2. **NO expander graphs** en inductivo (metían ruido — proyecto Exphormer-Max).
3. **NO embeddings de entidad ni PE de nodo** → rompen inductividad (caso (a) del análisis).
4. **NO atención ciega a la estructura** (sin aristas/relaciones) → caso (a), inviable.
5. **NO BCE-k negative sampling** como loss principal → usar CE de grafo completo.
6. **NO leer `h` acumulada en K/V de un stream separado** (causa del overfit previo).
7. **NO positional/structural encoding por nodo (RWSE ni LapPE) en la atención** (FB15k-237 v1+v2,
   2026-06-28 / 2026-07-08): ninguno da mejora robusta ni supera a NBFNet.
   - **RWSE (PE LOCAL, diag(P^k))**: full neutral-negativo (v1 −0.009, v2 −0.014); sparse INVIERTE
     signo entre splits (v1 −0.046, v2 +0.022 → no robusto); sparse_nbfv plano (ruido). Estructura
     DENTRO del horizonte que el MP ya captura → redundante.
   - **LapPE (PE GLOBAL, autovectores del Laplaciano)**: en v1 parecía ayudar al full (+0.035,
     0.375→0.410) pero **NO se replica en v2 (+0.005, dentro de ruido) → artefacto de split, no
     mejora estructural**. Sparse −0.003/−0.005; sparse_nbfv +0.013/−0.003 (inconsistente). Ninguno
     supera a NBFNet (v1 0.410 vs 0.459; v2 0.496 vs 0.526). Ver SESSION_NOTES 2026-07-08.
   => Ni PE local ni global mueven la aguja de forma confiable. La señal útil debe venir de FUERA del
   horizonte pero **como evidencia composicional/relacional** (caminos, no coordenadas de nodo), no
   de re-codificar la estructura del grafo como feature por nodo.
