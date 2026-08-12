# Hoja de ruta científica

## Objetivo del artículo

El artículo debe demostrar que la predicción de hotspots delictivos requiere evaluar simultáneamente error continuo, estructura espacial, eficiencia operacional y estabilidad temporal. El argumento central no es que el modelo más complejo gane siempre, sino que ConvLSTM V3 ofrece la mejor justificación empírica dentro del pipeline actual.

## Preguntas de investigación

1. ¿Superan los modelos espacio-temporales profundos a baselines estadísticos y clásicos bajo evaluación multi-horizonte?
2. ¿Las métricas espaciales IoU y PAI revelan diferencias que MAE/RMSE ocultan?
3. ¿La predicción autoregresiva conserva hotspots hasta t+12 o se homogeniza con el horizonte?
4. ¿La complejidad multi-scale mejora realmente la localización de hotspots?
5. ¿Qué limitaciones éticas aparecen al usar datos históricos de delitos denunciados?

## Línea experimental recomendada

1. Consolidar resultados de Historical Average, XGBoost agregado, ConvLSTM inicial, ConvLSTM V3 y ConvLSTM V4 multi-scale.
2. Guardar métricas por horizonte para `t+1` hasta `t+12`.
3. Priorizar comparaciones por IoU y PAI cuando MAE/RMSE contradigan la calidad espacial visual.
4. Analizar homogenización temporal con curvas de métricas por horizonte y mapas comparativos.
5. Incorporar atención sobre ConvLSTM como siguiente evolución incremental, sin reemplazar todavía la línea base ConvLSTM V3.
6. Dejar Transformers espacio-temporales como comparación posterior contra la línea base ya estabilizada.

## Interpretación metodológica

MAE y RMSE miden discrepancia pixel-wise, pero pueden favorecer mapas promedio. IoU evalúa coincidencia de áreas hotspot y PAI mide eficiencia operacional al relacionar eventos capturados con proporción de área marcada. Por eso, una reducción de MAE no debe aceptarse como mejora científica si reduce localización espacial o degrada t+12.

## Riesgos y controles

| Riesgo | Control experimental |
|---|---|
| Homogenización temporal | Métricas y visualizaciones por horizonte |
| Sobreajuste espacial | Comparar V3 contra V4 multi-scale y revisar validación temporal |
| Drift administrativo o de denuncia | Discutir cambios institucionales y no inferir causalidad criminal directa |
| Sesgo por datos históricos | Incluir sección de fairness, uso responsable y retroalimentación policial |
| Comparación injusta | Mantener split, horizonte, resolución y métricas constantes |

## Próxima versión sugerida: ConvLSTM con atención

### Problema que resuelve

La atención busca reducir pérdida de memoria espacio-temporal en horizontes largos, especialmente entre `t+7` y `t+12`.

### Justificación matemática

Un módulo de atención aprende pesos normalizados sobre estados o mapas de características. Si `H_t` representa características ConvLSTM, la atención produce una combinación ponderada `sum(alpha_t * H_t)` donde `alpha_t` aumenta la contribución de regiones o instantes relevantes. Esto puede preservar hotspots persistentes o emergentes que el autoregresivo residual tiende a suavizar.

### Referencias científicas sugeridas

- ConvLSTM para precipitación y dinámica espacio-temporal: Shi et al. (2015).
- Attention como ponderación adaptativa de contexto: Bahdanau et al. (2015).
- Self-attention/Transformers como base futura: Vaswani et al. (2017).

### Impacto esperado

| Métrica | Impacto esperado | Riesgo |
|---|---|---|
| MAE | Similar o leve aumento | La atención puede concentrar picos y aumentar error pixel-wise |
| RMSE | Similar o leve aumento | Penalización por picos mal localizados |
| IoU | Mejora esperada | Si la atención se sobreajusta puede desplazar hotspots |
| PAI | Mejora esperada | Puede aumentar falsas alarmas si predice expansión excesiva |

## Salidas mínimas por experimento

Cada ejecución debe guardar:

- `metrics_by_horizon.csv` con una fila por horizonte;
- `metrics_summary.csv` con agregados globales;
- mapas reales y predichos por horizonte;
- curvas MAE/RMSE/IoU/PAI por horizonte;
- configuración del experimento en JSON o Markdown;
- modelo entrenado en formato compatible con Google Colab.
