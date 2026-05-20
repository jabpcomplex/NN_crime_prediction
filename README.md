# Redes Neuronales para Predicción Criminal

Este repositorio agrupa la implementación de modelos basados en redes neuronales profundas orientados a la predicción espacio–temporal de patrones delictivos a partir de datos geoespaciales. La idea central es transformar secuencias de mapas de calor (rasters) de incidencia criminal mensual en predicciones sobre patrones futuros mediante arquitecturas convolucionales y recurrentes (especialmente ConvLSTM).

Se presentan scripts, notebooks y funciones auxiliares para:

1. Cargar y preprocesar datos raster de incidencia criminal mensual.
2. Generar secuencias espacio–temporales para modelos de Deep Learning.
3. Entrenar, validar y evaluar modelos para predicción multi-step.
4. Evaluar predicciones con métricas espaciales relevantes para hotspots.

# 📂 Estructura del repositorio

<img src="https://raw.githubusercontent.com/jabpcomplex/NN_crime_prediction/refs/heads/main/estructura_repo.png" alt="logo_complex">

# 📌 Versión final del modelo

## 📍 El archivo

RNN LSTMConv/convLSTM_3.ipynb

es la versión final hasta el momento del modelo de predicción.
Este notebook contiene:

- La carga de datos raster (incidencia delictiva mensual).
- El preprocesamiento (normalización, redimensionamiento).
- La definición y entrenamiento del modelo ConvLSTM.
- La evaluación con métricas globales (MAE, RMSE) y espaciales (IoU, PAI).
- La predicción autoregresiva de hasta 12 meses futuros.

Este notebook es el principal punto de partida para reproducir los resultados actuales y para continuar con extensiones del modelo.

# 🧠 Arquitectura del modelo

En convLSTM_3.ipynb se emplea una arquitectura ConvLSTM para capturar dependencias espacio–temporales de secuencias de mapas de calor.

- Entrada: secuencias de rasters normalizados (shape: time × height × width × channels).
- Salida: predicción del siguiente paso temporal, luego extendida a 12 meses en predicción autoregresiva.
- Entrenamiento: pérdida MAE (y variantes con ponderación espacial si se activa).
- Callbacks: EarlyStopping, ReduceLROnPlateau y ModelCheckpoint.
- Métricas de evaluación

Además de métricas globales tradicionales:


| Métrica | Qué mide |
|--- | --- |
| MAE |Error absoluto medio de predicción pixel a pixel|
| RMSE|Raíz de error cuadrático medio|
| IoU (Intersection over Union)|Coincidencia espacial entre hotspots reales y predichos|
| PAI (Prediction Accuracy Index) | Relación entre crímenes dentro de hotspots y área cubierta|


Estas evaluaciones permiten comparar no solo el ajuste numérico de la predicción, sino también la calidad espacial en cuanto a detección de zonas calientes de delito.


# 📌 Buenas prácticas y recomendaciones

Mantén las rutas de entrada y salida parametrizadas (variables).
Loguea métricas y modelos con frameworks como TensorBoard o MLflow.
Divide el conjunto de datos temporalmente (train/valid/test) respetando secuencia cronológica.
Usa transformaciones como log1p para realzar picos en mapas de calor.
Explora versiones mejoradas de pérdida que consideren IoU directamente.

# 📄 Licencia

Este proyecto está licenciado bajo GPL v2

# 📌 Notas finales

Este repositorio es un espacio de investigación y desarrollo con enfoque en:

- Modelado espacio–temporal de fenómenos urbanos.
- Aplicación de Deep Learning a datos geoespaciales.
- Evaluación cuantitativa y visual de predicciones.

La versión actual más madura se encuentra en **convLSTM_3.ipynb**, y se espera que futuras versiones incorporen variantes de arquitectura, pérdida compuesta y análisis comparativo más robusto.
