import numpy as np

from src.crime_prediction.metrics import evaluate_by_horizon, hotspot_mask, predictive_accuracy_index


def test_hotspot_mask_keeps_shape_and_boolean_dtype():
    frame = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = hotspot_mask(frame, quantile=0.75)
    assert mask.shape == frame.shape
    assert mask.dtype == bool
    assert mask.sum() == 4


def test_evaluate_by_horizon_returns_required_metrics():
    y_true = np.ones((2, 3, 4, 4), dtype=np.float32)
    y_pred = np.ones((2, 3, 4, 4), dtype=np.float32)
    metrics = evaluate_by_horizon(y_true, y_pred, quantile=0.75)
    assert list(metrics.columns) == ["horizon", "mae", "rmse", "iou", "pai"]
    assert metrics["horizon"].tolist() == [1, 2, 3]
    assert np.allclose(metrics["mae"], 0.0)
    assert np.allclose(metrics["rmse"], 0.0)
    assert np.allclose(metrics["iou"], 1.0)


def test_predictive_accuracy_index_rewards_concentrated_capture():
    y_true = np.zeros((4, 4), dtype=np.float32)
    y_true[0, 0] = 10.0
    y_pred = np.zeros((4, 4), dtype=np.float32)
    y_pred[0, 0] = 1.0
    pai = predictive_accuracy_index(y_true, y_pred, quantile=0.9)
    assert pai > 1.0
