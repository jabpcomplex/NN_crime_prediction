"""Spatial and temporal evaluation metrics for raster hotspot forecasting.

The functions in this module are NumPy-only so they can be reused from Google
Colab notebooks without requiring TensorFlow during post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonMetrics:
    """Metrics for one autoregressive forecast horizon."""

    horizon: int
    mae: float
    rmse: float
    iou: float
    pai: float


def hotspot_mask(frame: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """Return a binary hotspot mask using an upper-tail quantile threshold.

    Parameters
    ----------
    frame:
        Two-dimensional raster of normalized crime intensity.
    quantile:
        Upper quantile used as hotspot cutoff. The default marks the top 10% of
        raster cells, preserving a fixed-area comparison for IoU and PAI.
    """

    if frame.ndim != 2:
        raise ValueError("hotspot_mask expects a 2D raster frame")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    threshold = np.quantile(frame, quantile)
    return frame >= threshold


def intersection_over_union(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.9,
) -> float:
    """Compute hotspot IoU between real and predicted rasters."""

    true_mask = hotspot_mask(y_true, quantile=quantile)
    pred_mask = hotspot_mask(y_pred, quantile=quantile)
    union = np.logical_or(true_mask, pred_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(true_mask, pred_mask).sum()
    return float(intersection / union)


def predictive_accuracy_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.9,
) -> float:
    """Compute PAI using predicted hotspot area and observed raster intensity.

    PAI = (crime captured inside predicted hotspots / total crime) /
          (predicted hotspot cells / total cells)
    """

    pred_mask = hotspot_mask(y_pred, quantile=quantile)
    total_intensity = float(np.sum(y_true))
    area_fraction = float(np.mean(pred_mask))
    if total_intensity <= 0 or area_fraction <= 0:
        return 0.0
    captured_fraction = float(np.sum(y_true[pred_mask]) / total_intensity)
    return captured_fraction / area_fraction


def evaluate_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.9,
) -> pd.DataFrame:
    """Evaluate MAE, RMSE, IoU and PAI for each forecast horizon.

    Expected shape is ``(samples, horizons, rows, cols)`` or
    ``(samples, horizons, rows, cols, channels)``. If channels are present, the
    first channel is used because current rasters are single-band KDE maps.
    """

    true = _squeeze_channel(y_true)
    pred = _squeeze_channel(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.ndim != 4:
        raise ValueError("expected arrays with shape (samples, horizons, rows, cols)")

    rows: list[HorizonMetrics] = []
    for horizon_index in range(true.shape[1]):
        true_h = true[:, horizon_index]
        pred_h = pred[:, horizon_index]
        mae = float(np.mean(np.abs(true_h - pred_h)))
        rmse = float(np.sqrt(np.mean(np.square(true_h - pred_h))))
        iou = float(np.mean([
            intersection_over_union(t, p, quantile=quantile)
            for t, p in zip(true_h, pred_h)
        ]))
        pai = float(np.mean([
            predictive_accuracy_index(t, p, quantile=quantile)
            for t, p in zip(true_h, pred_h)
        ]))
        rows.append(HorizonMetrics(horizon_index + 1, mae, rmse, iou, pai))

    return pd.DataFrame([row.__dict__ for row in rows])


def save_metrics(
    metrics_by_horizon: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save horizon-level and aggregate metric CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_by_horizon.to_csv(output_path / "metrics_by_horizon.csv", index=False)
    summary = metrics_by_horizon.drop(columns=["horizon"]).agg(["mean", "std", "min", "max"])
    summary.to_csv(output_path / "metrics_summary.csv")


def _squeeze_channel(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 5:
        if arr.shape[-1] != 1:
            raise ValueError("only single-channel raster forecasts are supported")
        arr = arr[..., 0]
    return arr
