# Bitácora de experimentos — corrección de rasters y ajuste del modelo ConvLSTM

*Continuación de `convLSTM_3.ipynb` (versión "final" original). Este documento resume,
en orden cronológico, los cambios probados sobre esa base y su resultado.*

---

## 1. Regeneración de rasters (`crear_raster.ipynb` → `input3/`)

Se detectaron 3 defectos en los rasters originales (`input2/`):

- El área geográfica cubierta cambiaba ligeramente de un mes a otro (el extent no era
  fijo), por lo que la misma celda de la cuadrícula no representaba el mismo punto real
  de la ciudad entre distintos meses.
- Los píxeles no eran cuadrados (~4.66 × 3.32 m).
- El archivo no declaraba su sistema de coordenadas (CRS).

Se regeneraron los 72 rasters mensuales (2018-2023) corrigiendo los tres puntos: extent
fijo calculado una sola vez sobre todo el periodo, celda cuadrada de 5×5 m, y CRS
explícito (EPSG:32614).

---

## 2. Etapa 3 — rasters corregidos + fix de orden cronológico

Se detectó además que el código original ordenaba los archivos de raster
**alfabéticamente como texto** (`sorted(glob.glob(...))`) en vez de cronológicamente. Como
los meses no llevan cero a la izquierda en el nombre del archivo, el orden real que
recibía el modelo dentro de cada año era: enero, octubre, noviembre, diciembre, febrero,
marzo... en vez de enero a diciembre. Se corrigió ordenando explícitamente por
`(año, mes)` extraído del nombre del archivo.

**Resultado (una sola corrida, sin control de semilla todavía):**

| Métrica | Original (`input2/`) | Etapa 3 (`input3/` + orden corregido) |
|---|---|---|
| MAE | 0.0918 | 0.0846 |
| RMSE | 0.1147 | 0.1055 |

---

## 3. Etapa 4 — pérdida ponderada + menos regularización + log-transform (descartado)

**Hipótesis:** en la Etapa 3, la predicción autorregresiva a 12 meses colapsaba a un mapa
casi plano (sin distinguir zona caliente de zona fría), confirmado visualmente contra el
mapa real. Se planteó que el error cuadrático medio (MSE) no distingue entre acertar el
punto caliente y acertar el fondo, y que demasiada regularización (`dropout`/`recurrent_
dropout=0.3`, `l2=1e-4`) con pocos datos de entrenamiento empujaba al modelo hacia esa
solución "segura". Se probaron 3 cambios juntos:

1. Pérdida MSE ponderada por la densidad real de cada celda (más peso al error en zonas
   de alto crimen real).
2. Menos regularización (`dropout`/`recurrent_dropout` de 0.3 a 0.1, `l2` de 1e-4 a 1e-5).
3. Transformación `log1p` de los datos antes de normalizar, para no aplastar el contraste
   entre fondo y pico.

**Primeras corridas sueltas** (sin semilla fija) dieron resultados muy distintos entre sí
en IoU/PAI (0.0000 a 0.33 según la corrida) — evidencia de que el test set de solo 2
secuencias mensuales genera mucha varianza run-a-run, independiente de la configuración.

**Prueba rigurosa** (`comparacion_semillas_etapa3_vs_etapa4.ipynb`): se repitió el
entrenamiento de Etapa 3 y Etapa 4, cada una con 3 semillas distintas, y se comparó el
promedio ± desviación estándar:

| Configuración | MAE | RMSE | IoU (12 meses) | PAI (12 meses) |
|---|---|---|---|---|
| Etapa 3 (MSE plano, dropout 0.3, sin log) | 0.0774 ± 0.0080 | 0.0991 ± 0.0111 | 0.266 ± 0.053 | 4.17 ± 0.67 |
| Etapa 4 (pérdida ponderada, dropout 0.1, log) | 0.0874 ± 0.0103 | 0.1107 ± 0.0141 | 0.173 ± 0.033 | 2.94 ± 0.47 |

**Conclusión: la Etapa 4 tuvo peor desempeño que la Etapa 3 en las 4 métricas, de forma
consistente entre semillas (las distribuciones casi no se solapan). Se descarta esta
combinación de cambios** — la hipótesis tenía sentido en teoría, pero no se sostuvo con
evidencia. La causa exacta de por qué empeoró no se investigó a fondo (candidato para
trabajo futuro: aislar cada uno de los 3 cambios por separado en vez de probarlos juntos).

---

## 4. Limitación de fondo, sin resolver todavía

El test set son solo 2 secuencias mensuales (14 meses de test ÷ ventana de 12 = 2
secuencias posibles). Esto por sí solo genera bastante varianza entre corridas y limita
qué tan concluyente puede ser cualquier resultado — incluyendo los de esta misma
bitácora — hasta que se consiga más historia de datos de prueba o se cambie la estrategia
de validación (por ejemplo, validación walk-forward en vez de un solo corte 80/20).

---

## Archivos de esta ronda de experimentos

| Archivo | Qué hace |
|---|---|
| `crear_raster.ipynb` | Genera los 72 `.tif` de `input3/` con extent fijo, CRS y celda cuadrada |
| `convLSTM_etapa3_rasters_alineados.ipynb` | ConvLSTM original + fix de orden cronológico, corrido sobre `input3/` |
| `convLSTM_etapa4_perdida_ponderada.ipynb` | Experimento descartado: pérdida ponderada + menos regularización + log-transform |
| `comparacion_semillas_etapa3_vs_etapa4.ipynb` | Prueba con múltiples semillas que muestra que Etapa 4 no mejora sobre Etapa 3 |
