# Registro de versiones experimentales

Este archivo define cómo guardar y comparar incrementalmente las versiones del pipeline de predicción espacio-temporal de incidencia delictiva. El objetivo es conservar trazabilidad científica sin romper los notebooks y rutas existentes.

## Configuración base obligatoria

Toda versión nueva debe declarar explícitamente:

| Parámetro | Valor base actual |
|---|---:|
| Resolución raster | `TARGET_SHAPE = (90, 124)` |
| Ventana de entrada | `INPUT_FRAMES = 7` |
| Horizonte autoregresivo | `FORECAST_STEPS = 12` |
| Evaluación temporal | `t+1, ..., t+12` |
| Métricas continuas | MAE, RMSE |
| Métricas espaciales | IoU, PAI |

La comparación entre modelos solo es válida si se mantienen constantes los datos, el split temporal, el horizonte de predicción y las métricas.

## Versiones registradas

| Versión | Modelo | Rol científico | Estado | Resultado principal |
|---|---|---|---|---|
| V1 | Historical Average | Baseline estadístico | Referencia | Controla si Deep Learning supera persistencia histórica |
| V2 | XGBoost optimizado | Baseline clásico con variables agregadas | Referencia | No debe entrenarse como modelo pixel-wise masivo |
| V3 | ConvLSTM inicial | Red profunda base | Superado | Tendencia a predicciones homogéneas |
| V4 | ConvLSTM V3 final | Modelo seleccionado | Activo | Mejor balance MAE/RMSE/IoU/PAI |
| V5 | ConvLSTM multi-scale | Ablación de complejidad espacial | No seleccionado | IoU/PAI similares, MAE/RMSE peores; posible sobreajuste |

## ConvLSTM V3 seleccionado

La versión ConvLSTM V3 se conserva como línea base profunda final porque combina mejoras incrementales directamente relacionadas con los errores observados:

- pérdida híbrida con MAE ponderado y Soft IoU para penalizar errores en hotspots;
- optimizador Adam con `learning_rate = 1e-4` para estabilizar el entrenamiento;
- Batch Normalization y regularización L2 para reducir sobreajuste;
- cuatro capas ConvLSTM con 36 filtros y kernels 3x3;
- forecasting autoregresivo residual para conservar memoria espacial;
- suavizado gaussiano controlado para reducir artefactos sin eliminar hotspots.

Resultados actuales reportados:

| Modelo | MAE | RMSE | IoU | PAI |
|---|---:|---:|---:|---:|
| ConvLSTM V3 | 0.079377 | 0.092125 | 0.559545 | 7.092934 |
| ConvLSTM V4 multi-scale | 0.340113 | 0.350291 | 0.559913 | 7.106021 |

Interpretación: V4 multi-scale no justifica su complejidad adicional porque mantiene IoU/PAI prácticamente iguales pero degrada MAE/RMSE. La hipótesis operativa es que el cuello de botella actual es temporal/autoregresivo, no únicamente espacial.

## Plantilla obligatoria para nuevas versiones

Cada versión futura debe documentarse con esta estructura:

```markdown
## V{n} - {nombre}

### Problema que resuelve

### Justificación matemática

### Referencia científica

### Cambios de implementación

### Impacto esperado

| Métrica | Impacto esperado | Riesgo |
|---|---|---|
| MAE | | |
| RMSE | | |
| IoU | | |
| PAI | | |

### Archivos generados

- métricas por horizonte: `output/{version}/metrics_by_horizon.csv`
- métricas agregadas: `output/{version}/metrics_summary.csv`
- predicciones: `output/{version}/predictions/`
- figuras: `output/{version}/figures/`
- modelo: `output/{version}/models/`
```

## Criterios de aceptación

Una versión solo puede considerarse comparable si:

1. evalúa los 12 horizontes autoregresivos por separado;
2. reporta MAE, RMSE, IoU y PAI;
3. guarda CSV con métricas y figuras reproducibles;
4. conserva el split temporal;
5. explica limitaciones técnicas y éticas;
6. no reemplaza ConvLSTM V3 sin una mejora clara en IoU, PAI y estabilidad de t+12.
