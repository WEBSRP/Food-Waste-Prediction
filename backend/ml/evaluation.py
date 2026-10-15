"""Chronological forecasting evaluation metrics and time-series validation splits."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, safe WAPE (%), and Forecast Bias on held-out observations."""
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(actual) == 0:
        return {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "bias": 0.0}

    errors = pred - actual
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    total_actual = float(np.sum(np.abs(actual)))
    if total_actual <= 1e-9:
        # Safe handling when actual demand sums to zero
        wape = 0.0 if float(np.sum(abs_errors)) <= 1e-9 else 100.0
    else:
        wape = float((np.sum(abs_errors) / total_actual) * 100.0)

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape": round(wape, 3),
        "bias": round(bias, 3),
    }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a chronologically ordered dataset into strictly non-overlapping train, validation, and test sets."""
    if not (0.4 <= train_ratio <= 0.9) or not (0.0 <= val_ratio <= 0.4):
        raise ValueError("Invalid chronological split ratios.")
    n = len(df)
    if n < 15:
        raise ValueError("At least 15 chronological records are required for a 3-way split.")
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df
