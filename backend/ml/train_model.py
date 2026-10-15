"""Training pipeline comparing a Seasonal-Naive Moving Average baseline with a Tree-Ensemble ML model.

Uses scikit-learn's HistGradientBoostingRegressor when installed, and provides a genuine
CPU-native Gradient Boosted Regression Tree Ensemble in pure NumPy/SciPy when running in
minimal environments without scikit-learn.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from backend.ml.evaluation import calculate_forecast_metrics, chronological_split
from backend.ml.feature_engineering import FEATURE_COLUMNS, build_supervised_dataset
from backend.ml.model_registry import save_model_artifact

try:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class SeasonalMovingAverageBaseline:
    """Baseline model combining same-weekday 4-week moving average and 7-day rolling mean."""

    def __init__(self) -> None:
        self.holiday_ratio_by_meal: Dict[int, float] = {0: 0.65, 1: 0.65, 2: 0.65}
        self.mean_by_meal: Dict[int, float] = {0: 300.0, 1: 520.0, 2: 450.0}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeasonalMovingAverageBaseline":
        df = X.copy()
        df["_y"] = y
        for m_code in (0, 1, 2):
            sub = df[df["meal_type_code"] == m_code]
            if not sub.empty:
                self.mean_by_meal[m_code] = float(sub["_y"].mean())
                non_hol = sub[sub["is_holiday"] == 0]["_y"]
                hol = sub[sub["is_holiday"] == 1]["_y"]
                if len(non_hol) > 0 and len(hol) > 0:
                    ratio = float(hol.mean() / max(1.0, non_hol.mean()))
                    self.holiday_ratio_by_meal[m_code] = float(np.clip(ratio, 0.4, 1.0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X), dtype=float)
        for i, (_, row) in enumerate(X.iterrows()):
            m_code = int(row.get("meal_type_code", 1))
            same_dow = float(row.get("same_weekday_4w_mean", self.mean_by_meal.get(m_code, 400.0)))
            roll_7 = float(row.get("rolling_7d_mean", same_dow))
            base_pred = 0.65 * same_dow + 0.35 * roll_7
            if float(row.get("is_holiday", 0.0)) >= 0.5:
                base_pred *= self.holiday_ratio_by_meal.get(m_code, 0.65)
            preds[i] = max(20.0, base_pred)
        return preds


class NumpyGradientBoostedTreesRegressor:
    """CPU-native Gradient Boosted Regression Trees learner.

    Used automatically when scikit-learn is not installed in the host environment.
    Initializes predictions using meal-type baseline priors and fits stagewise regression
    trees of depth 3 on the residuals without unbounded linear extrapolation on calendar features.
    """

    def __init__(
        self,
        n_estimators: int = 85,
        learning_rate: float = 0.12,
        max_depth: int = 3,
        min_samples_leaf: int = 6,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.base_estimator_ = SeasonalMovingAverageBaseline()
        self.trees_: List[Dict[str, Any]] = []

    def _fit_tree(self, X: np.ndarray, residuals: np.ndarray, depth: int) -> Dict[str, Any]:
        n, p = X.shape
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.std(residuals) < 1e-4:
            return {"is_leaf": True, "value": float(np.mean(residuals))}

        best_feat = -1
        best_thresh = 0.0
        best_gain = 0.0
        parent_var = float(np.sum((residuals - np.mean(residuals)) ** 2))

        for feat_idx in range(p):
            col = X[:, feat_idx]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            thresholds = (
                np.quantile(uniq, np.linspace(0.15, 0.85, min(7, len(uniq))))
                if len(uniq) > 8
                else (uniq[:-1] + uniq[1:]) / 2.0
            )
            for thresh in np.unique(thresholds):
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                r_left = residuals[left_mask]
                r_right = residuals[~left_mask]
                sse = float(
                    np.sum((r_left - np.mean(r_left)) ** 2) + np.sum((r_right - np.mean(r_right)) ** 2)
                )
                gain = parent_var - sse
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = float(thresh)

        if best_feat == -1:
            return {"is_leaf": True, "value": float(np.mean(residuals))}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "is_leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._fit_tree(X[left_mask], residuals[left_mask], depth + 1),
            "right": self._fit_tree(X[~left_mask], residuals[~left_mask], depth + 1),
        }

    def _predict_tree_row(self, node: Dict[str, Any], row: np.ndarray) -> float:
        while not node["is_leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["value"])

    def _predict_tree(self, node: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_tree_row(node, r) for r in X], dtype=float)

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> "NumpyGradientBoostedTreesRegressor":
        self.base_estimator_.fit(X_df, y)
        current_pred = self.base_estimator_.predict(X_df)
        X_raw = X_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        self.trees_ = []

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = self._fit_tree(X_raw, residuals, depth=0)
            update = self._predict_tree(tree, X_raw)
            current_pred += self.learning_rate * update
            self.trees_.append(tree)

        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        pred = self.base_estimator_.predict(X_df)
        X_raw = X_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * self._predict_tree(tree, X_raw)
        return np.maximum(15.0, pred)


def _compute_permutation_importances(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute normalized permutation feature importances on held-out evaluation data."""
    rng = np.random.default_rng(seed)
    base_preds = model.predict(X_eval[FEATURE_COLUMNS])
    base_mae = float(np.mean(np.abs(y_eval - base_preds)))
    importances: Dict[str, float] = {}

    for col in FEATURE_COLUMNS:
        X_perm = X_eval[FEATURE_COLUMNS].copy()
        X_perm[col] = rng.permutation(X_perm[col].to_numpy())
        perm_preds = model.predict(X_perm)
        perm_mae = float(np.mean(np.abs(y_eval - perm_preds)))
        importances[col] = max(0.0, perm_mae - base_mae)

    total = sum(importances.values())
    if total > 1e-9:
        importances = {k: round(v / total, 4) for k, v in importances.items()}
    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))


def train_and_evaluate_pipeline(raw_records_df: pd.DataFrame) -> Dict[str, Any]:
    """Execute full chronological preprocessing, feature engineering, model training, and evaluation."""
    if len(raw_records_df) < 30:
        raise ValueError("Insufficient historical records to train ML model (minimum 30 records required).")

    sup_df = build_supervised_dataset(raw_records_df)
    warmup = min(42, max(6, len(sup_df) // 10))
    modeling_df = sup_df.iloc[warmup:].copy().reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(modeling_df, train_ratio=0.70, val_ratio=0.15)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    X_fit = fit_df[FEATURE_COLUMNS]
    y_fit = fit_df["target_demand"].to_numpy(dtype=float)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_demand"].to_numpy(dtype=float)

    baseline_model = SeasonalMovingAverageBaseline().fit(X_fit, y_fit)
    baseline_test_preds = baseline_model.predict(X_test)
    baseline_metrics = calculate_forecast_metrics(y_test, baseline_test_preds)

    if _HAS_SKLEARN:
        ml_model_name = "HistGradientBoostingRegressor (scikit-learn)"
        ml_model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_depth=5,
            min_samples_leaf=10,
            random_state=202604,
        )
        ml_model.fit(X_fit, y_fit)
    else:
        ml_model_name = "GradientBoostedRegressionTrees (CPU-Ensemble)"
        ml_model = NumpyGradientBoostedTreesRegressor(
            n_estimators=85,
            learning_rate=0.12,
            max_depth=3,
            min_samples_leaf=6,
        )
        ml_model.fit(X_fit, y_fit)

    ml_test_preds = np.maximum(15.0, np.asarray(ml_model.predict(X_test), dtype=float))
    ml_metrics = calculate_forecast_metrics(y_test, ml_test_preds)

    residuals = y_test - ml_test_preds
    residual_std = round(float(np.std(residuals)), 2)
    residual_q85 = round(float(np.quantile(np.maximum(0.0, residuals), 0.85)), 2)

    residual_stats_by_meal: Dict[str, Dict[str, float]] = {}
    for m_name, m_code in [("breakfast", 0), ("lunch", 1), ("dinner", 2)]:
        mask = (X_test["meal_type_code"] == m_code).to_numpy()
        if np.any(mask):
            m_res = residuals[mask]
            residual_stats_by_meal[m_name] = {
                "std": round(float(np.std(m_res)), 2),
                "q85": round(float(np.quantile(np.maximum(0.0, m_res), 0.85)), 2),
                "q90": round(float(np.quantile(np.maximum(0.0, m_res), 0.90)), 2),
            }
        else:
            residual_stats_by_meal[m_name] = {"std": residual_std, "q85": residual_q85, "q90": residual_q85 * 1.15}

    importances = _compute_permutation_importances(ml_model, test_df, y_test)
    selected_model = "ml_model" if ml_metrics["mae"] <= baseline_metrics["mae"] else "baseline_model"

    version_ts = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    comparison_series: List[Dict[str, Any]] = []
    for i in range(min(90, len(test_df))):
        row_idx = len(test_df) - min(90, len(test_df)) + i
        r = test_df.iloc[row_idx]
        comparison_series.append(
            {
                "service_date": str(r["service_date"]),
                "meal_type": str(r["meal_type"]),
                "menu_category": str(r["menu_category"]),
                "actual_demand": int(round(float(r["target_demand"]))),
                "ml_predicted": round(float(ml_test_preds[row_idx]), 1),
                "baseline_predicted": round(float(baseline_test_preds[row_idx]), 1),
                "meals_prepared": int(round(float(r["meals_prepared"]))),
            }
        )

    artifact_bundle = {
        "model_version": version_ts,
        "model_name": ml_model_name,
        "baseline_name": "SeasonalWeekdayMovingAverage",
        "selected_model": selected_model,
        "ml_model": ml_model,
        "baseline_model": baseline_model,
        "feature_columns": FEATURE_COLUMNS,
        "residual_std": residual_std,
        "residual_q85": residual_q85,
        "residual_stats_by_meal": residual_stats_by_meal,
        "ml_metrics": ml_metrics,
        "baseline_metrics": baseline_metrics,
        "feature_importances": importances,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    artifact_path = save_model_artifact(artifact_bundle, version_ts)

    return {
        "model_version": version_ts,
        "model_name": ml_model_name,
        "baseline_name": "SeasonalWeekdayMovingAverage",
        "train_start_date": str(fit_df["service_date"].iloc[0]),
        "train_end_date": str(fit_df["service_date"].iloc[-1]),
        "eval_start_date": str(test_df["service_date"].iloc[0]),
        "eval_end_date": str(test_df["service_date"].iloc[-1]),
        "train_samples": int(len(fit_df)),
        "eval_samples": int(len(test_df)),
        "ml_mae": ml_metrics["mae"],
        "ml_rmse": ml_metrics["rmse"],
        "ml_wape": ml_metrics["wape"],
        "ml_bias": ml_metrics["bias"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_wape": baseline_metrics["wape"],
        "baseline_bias": baseline_metrics["bias"],
        "selected_model": selected_model,
        "residual_std": residual_std,
        "residual_q85": residual_q85,
        "feature_importances": importances,
        "comparison_series": comparison_series,
        "artifact_path": artifact_path,
    }
