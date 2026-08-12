"""ConvLSTM with spatial attention for the next experimental delivery.

This module keeps ConvLSTM V3 as the baseline and adds a lightweight attention
head intended to reduce long-horizon homogenization without changing the input
resolution, temporal window, metrics or autoregressive evaluation protocol.
"""

from __future__ import annotations

from pathlib import Path


TARGET_SHAPE = (90, 124)
INPUT_FRAMES = 7
FORECAST_STEPS = 12
DEFAULT_FILTERS = 36
DEFAULT_LEARNING_RATE = 1e-4


def build_attention_convlstm(
    input_frames: int = INPUT_FRAMES,
    target_shape: tuple[int, int] = TARGET_SHAPE,
    filters: int = DEFAULT_FILTERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_strength: float = 1e-5,
):
    """Build a ConvLSTM V3-compatible model with spatial attention.

    The model predicts one raster frame. Multi-step forecasts should be produced
    autoregressively with :func:`autoregressive_forecast` to keep the comparison
    against ConvLSTM V3 fair.
    """

    import tensorflow as tf
    from tensorflow.keras import Model, regularizers
    from tensorflow.keras.layers import (
        Add,
        BatchNormalization,
        Conv2D,
        ConvLSTM2D,
        Input,
        Multiply,
    )

    rows, cols = target_shape
    inputs = Input(shape=(input_frames, rows, cols, 1), name="raster_sequence")
    regularizer = regularizers.l2(l2_strength)

    x = ConvLSTM2D(filters, (3, 3), padding="same", return_sequences=True,
                   kernel_regularizer=regularizer, name="convlstm_1")(inputs)
    x = BatchNormalization(name="bn_1")(
        x
    )
    x = ConvLSTM2D(filters, (3, 3), padding="same", return_sequences=True,
                   kernel_regularizer=regularizer, name="convlstm_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = ConvLSTM2D(filters, (3, 3), padding="same", return_sequences=True,
                   kernel_regularizer=regularizer, name="convlstm_3")(x)
    x = BatchNormalization(name="bn_3")(x)
    x = ConvLSTM2D(filters, (3, 3), padding="same", return_sequences=False,
                   kernel_regularizer=regularizer, name="convlstm_4")(x)
    x = BatchNormalization(name="bn_4")(x)

    attention = Conv2D(1, (1, 1), activation="sigmoid", padding="same", name="spatial_attention")(x)
    attended = Multiply(name="attention_weighted_features")([x, attention])
    residual = Conv2D(1, (3, 3), activation="linear", padding="same", name="residual_delta")(attended)
    last_frame = inputs[:, -1, :, :, :]
    outputs = Add(name="residual_forecast")([last_frame, residual])
    outputs = tf.keras.layers.Activation("relu", name="non_negative_forecast")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="ConvLSTM_V6_attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=hybrid_hotspot_loss,
        metrics=["mae"],
    )
    return model


def hybrid_hotspot_loss(y_true, y_pred):
    """Weighted MAE plus Soft IoU loss for hotspot-preserving training."""

    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    hotspot_weights = 1.0 + 4.0 * y_true
    weighted_mae = tf.reduce_mean(hotspot_weights * tf.abs(y_true - y_pred))

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=[1, 2, 3])
    soft_iou = (intersection + 1e-6) / (union + 1e-6)
    return weighted_mae + (1.0 - tf.reduce_mean(soft_iou))


def autoregressive_forecast(model, seed_sequence, steps: int = FORECAST_STEPS):
    """Generate ``steps`` future rasters by feeding predictions back as input."""

    import numpy as np

    window = np.array(seed_sequence, dtype=np.float32, copy=True)
    predictions = []
    for _ in range(steps):
        next_frame = model.predict(window, verbose=0)
        predictions.append(next_frame)
        window = np.concatenate([window[:, 1:], next_frame[:, None, ...]], axis=1)
    return np.stack(predictions, axis=1)


def save_model_artifacts(model, output_dir: str | Path) -> None:
    """Save the attention model in Keras format for Colab reuse."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(output_path / "attention_convlstm.keras")
