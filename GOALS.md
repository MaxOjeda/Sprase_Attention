# GOALS.md — Attention (Graph Transformer para KGC)

Objetivos del proyecto. Estable; cambia poco. Para resultados de sesión ver
`SESSION_NOTES.md`; para el briefing operacional ver `CLAUDE.md`.

---

## Objetivo general

Determinar si una arquitectura basada en **atención (Graph Transformer)** puede
**superar a NBFNet** en knowledge graph completion (KGC), con énfasis en el régimen
**inductivo** (grafo de test con entidades disjuntas). Una sola arquitectura debe servir
transductivo e inductivo; solo cambian hiperparámetros (requisito del manuscrito de tesis).

---

## Pregunta central de la etapa actual (reinicio 2026-06-21)

> Un **Graph Transformer con atención FULL (densa, all-pairs O(N²))** —el techo de
> expresividad de cualquier atención sparse— ¿supera a **NBFNet** en **FB15k-237
> inductivo v1**?

Lógica del sanity check: si la versión densa (el techo) **no** gana, una versión sparse
(que es una restricción de la densa) no tiene caso. Decide si la dirección "Graph
Transformer para KGC" sigue viva antes de invertir en variantes sparse/lineales.

**Estado: RESPONDIDA — NO. Ver `SESSION_NOTES.md` (sesión 2026-06-21).**

---

## Restricciones de diseño (no negociables)

- **Inductivo puro**: sin embeddings de entidad, sin positional encoding de nodo. Lo
  único compartido train/test son embeddings de **relación**.
- **Labeling trick** (NBFNet): anclar la propagación a la fuente y condicionar a la
  relación de la query.
- **Loss = CE de grafo completo** (no BCE-k negative sampling).
- **Model selection por `valid_mrr`** (val = train graph); test sobre el grafo inductivo
  disjunto. Reportar siempre valid_mrr y test_mrr + hits@1/3/10, mr.
- La señal de no-redundancia, si la hay, debe venir de **composición/evidencia fuera del
  horizonte de propagación del message passing**, no de recombinar/reponderar lo que el
  MP ya entrega.

---

## Criterio de éxito

- **Éxito**: una arquitectura de atención que iguale o supere el test_mrr de NBFNet
  (~0.459 en FB15k-237 ind v1) **sin** romper inductividad y **transfiriendo** al grafo
  disjunto (no solo subir valid).
- **Fracaso de una línea**: test_mrr < NBFNet de forma estructural (no cierra con tuning
  razonable), o valid↑ / test↓ (overfit estructural). En ese caso: parar, diagnosticar
  causa mecánica, registrar en `SESSION_NOTES.md` y en la lista negra de `CLAUDE.md`.

---

## Marco teórico

`transformer_vs_nbfnet.tex`:
- **Caso (a)** transformer ciego a la estructura (atención sobre embeddings de nodo sin
  aristas): rompe la inductividad. Inviable → NO hacer.
- **Caso (b)** graph transformer con labeling trick + adyacencia relacional: puede igualar
  a NBFNet en principio, pero "reimplementa message passing con agregación aprendida".
  Este proyecto mide empíricamente si esa flexibilidad extra mejora o no.

---

## Fuera de alcance (refutado — ver lista negra en `CLAUDE.md`)

1. Construir sobre KnowFormer (V-RMPNN/QK-RMPNN/RSPMM) — atención decorativa.
2. Expander graphs en inductivo — metían ruido (Exphormer-Max).
3. Embeddings de entidad / PE de nodo — rompen inductividad.
4. Atención ciega a la estructura — caso (a), inviable.
5. BCE-k negative sampling como loss principal.
6. Leer `h` acumulada en K/V de un stream separado — causa del overfit previo.
