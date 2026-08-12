# V6 - ConvLSTM con atención espacial

## Problema que resuelve

Busca reducir la homogenización temporal observada entre `t+7` y `t+12` agregando una máscara de atención espacial sobre las características ConvLSTM antes de estimar el residual autoregresivo.

## Justificación matemática

Sea `H` el tensor de características producido por la pila ConvLSTM. La atención espacial aprende `A = sigmoid(Conv2D_1x1(H))` y genera `H_att = A ⊙ H`. El operador `⊙` pondera regiones relevantes antes de calcular el residual `Δ_t`, de modo que la predicción queda `Y_hat_t = ReLU(X_t + Δ_t)`. Esto mantiene el supuesto residual de V3 y añade una puerta espacial diferenciable para enfatizar hotspots.

## Referencias científicas

- Shi et al. (2015): ConvLSTM para predicción espacio-temporal.
- Bahdanau et al. (2015): atención como ponderación adaptativa de contexto.
- Vaswani et al. (2017): self-attention como antecedente de futuras comparaciones Transformer.

## Cambios de implementación

- Nuevo módulo `src/crime_prediction/attention_convlstm.py`.
- Métricas reutilizables por horizonte en `src/crime_prediction/metrics.py`.
- Conserva `TARGET_SHAPE = (90, 124)`, `INPUT_FRAMES = 7` y `FORECAST_STEPS = 12`.
- Mantiene entrenamiento de un paso y evaluación multi-step autoregresiva para comparación justa contra ConvLSTM V3.

## Impacto esperado

| Métrica | Impacto esperado | Riesgo |
|---|---|---|
| MAE | Similar o levemente mayor | Puede penalizar picos más intensos |
| RMSE | Similar o levemente mayor | Sensible a picos desplazados |
| IoU | Mejora esperada | Sobreajuste de la máscara espacial |
| PAI | Mejora esperada | Posibles falsas alarmas por expansión anticipada |

## Salidas reproducibles

- `output/attention_convlstm_v6/metrics_by_horizon.csv`
- `output/attention_convlstm_v6/metrics_summary.csv`
- `output/attention_convlstm_v6/models/attention_convlstm.keras`
- `output/attention_convlstm_v6/figures/`
- `output/attention_convlstm_v6/predictions/`
