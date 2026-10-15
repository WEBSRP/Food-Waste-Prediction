"""Demand prediction inference with cold-start fallback, planning buffers, and factor explanations."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.ml.feature_engineering import FEATURE_COLUMNS, build_single_forecast_features
from backend.ml.model_registry import load_model_artifact

DEFAULT_COLD_START_MEANS = {
    "breakfast": 300.0,
    "lunch": 520.0,
    "dinner": 450.0,
}


def predict_meal_demand(
    historical_df: pd.DataFrame,
    forecast_date: str,
    meal_type: str,
    menu_category: str = "north_indian_standard",
    is_holiday: int = 0,
    special_event: str = "none",
    weather_condition: str = "clear",
    temperature_c: float = 28.0,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a meal demand forecast with uncertainty interval and explainable factor breakdown.

    Automatically falls back to a documented cold-start baseline if fewer than 21 historical
    records exist or if no trained model artifact is available.
    """
    m_type = meal_type.lower().strip()
    feat_dict = build_single_forecast_features(
        historical_df=historical_df,
        forecast_date=forecast_date,
        meal_type=m_type,
        menu_category=menu_category,
        is_holiday=is_holiday,
        special_event=special_event,
        weather_condition=weather_condition,
        temperature_c=temperature_c,
    )
    X_row = pd.DataFrame([feat_dict])[FEATURE_COLUMNS]

    artifact = load_model_artifact(model_version)
    has_sufficient_history = (
        historical_df is not None
        and len(historical_df[historical_df["service_date"] < forecast_date]) >= 21
    )

    if artifact is None or not has_sufficient_history:
        # Cold-start documented baseline fallback
        base_val = float(feat_dict.get("same_weekday_4w_mean", DEFAULT_COLD_START_MEANS.get(m_type, 400.0)))
        if not has_sufficient_history and (historical_df is None or historical_df.empty):
            base_val = DEFAULT_COLD_START_MEANS.get(m_type, 400.0)
        if int(is_holiday) == 1:
            base_val *= 0.65
        rounded = int(math.ceil(base_val))
        buffer_portions = int(math.ceil(base_val * 0.12))
        return {
            "forecast_date": forecast_date,
            "meal_type": m_type,
            "menu_category": menu_category,
            "is_holiday": int(is_holiday),
            "special_event": special_event,
            "weather_condition": weather_condition,
            "temperature_c": float(temperature_c),
            "predicted_demand": round(base_val, 1),
            "rounded_operational_demand": rounded,
            "baseline_demand": round(base_val, 1),
            "planning_buffer_portions": buffer_portions,
            "upper_bound_demand": rounded + buffer_portions,
            "lower_bound_demand": max(0, rounded - buffer_portions),
            "model_version": "cold_start_baseline_v1",
            "is_fallback_baseline": True,
            "contributing_factors": [
                {
                    "factor": "Cold-Start Baseline Fallback",
                    "impact": "Preliminary estimate",
                    "detail": "Historical records (<21 services) or trained artifact unavailable; using weekday moving average baseline.",
                }
            ],
        }

    baseline_model = artifact["baseline_model"]
    ml_model = artifact["ml_model"]
    selected = artifact.get("selected_model", "ml_model")

    base_pred = float(baseline_model.predict(X_row)[0])
    ml_pred = float(ml_model.predict(X_row)[0])
    primary_pred = ml_pred if selected == "ml_model" else base_pred
    rounded_demand = int(math.ceil(primary_pred))

    meal_stats = artifact.get("residual_stats_by_meal", {}).get(m_type, {})
    q85_buffer = float(meal_stats.get("q85", artifact.get("residual_q85", 22.0)))
    std_buffer = float(meal_stats.get("std", artifact.get("residual_std", 18.0)))
    planning_buffer = int(math.ceil(max(q85_buffer, std_buffer)))

    # Build human-readable factor explanation grounded in the input features
    factors: List[Dict[str, str]] = [
        {
            "factor": "Same-Weekday 4-Week Demand Baseline",
            "impact": f"{feat_dict['same_weekday_4w_mean']:.0f} portions",
            "detail": f"Historical average for {m_type} on this day of the week over the prior 4 weeks.",
        },
        {
            "factor": "Recent 7-Day Rolling Trend",
            "impact": f"{feat_dict['rolling_7d_mean']:.0f} portions (±{feat_dict['rolling_7d_std']:.0f})",
            "detail": "Recent 7-day consumption mean and volatility for this meal slot.",
        },
    ]

    if int(is_holiday) == 1:
        factors.append(
            {
                "factor": "Scheduled Holiday Adjustment",
                "impact": "-35% to -42% attendance shift",
                "detail": "Campus holiday reduces on-campus hostel dining attendance significantly.",
            }
        )
    elif int(feat_dict["is_weekend"]) == 1:
        factors.append(
            {
                "factor": "Weekend Attendance Pattern",
                "impact": "-20% to -25% vs mid-week",
                "detail": "Saturday/Sunday hostel attendance is systematically lower than Monday–Thursday.",
            }
        )

    if menu_category in ("festive_thali", "south_indian_special", "continental_fusion"):
        factors.append(
            {
                "factor": f"Menu Category ({menu_category.replace('_', ' ').title()})",
                "impact": "+8% to +22% popularity lift",
                "detail": "Special and festive menus historically attract higher student turnout.",
            }
        )
    elif menu_category == "healthy_millet_bowl":
        factors.append(
            {
                "factor": "Menu Category (Healthy Millet Bowl)",
                "impact": "-8% to -10% turnout adjustment",
                "detail": "Standardized down-scaling prevents habitual overproduction on niche menu days.",
            }
        )

    if special_event != "none":
        factors.append(
            {
                "factor": f"Campus Event ({special_event.replace('_', ' ').title()})",
                "impact": "+8% to +20% demand surge",
                "detail": "Special campus schedule increases on-campus meal participation.",
            }
        )

    return {
        "forecast_date": forecast_date,
        "meal_type": m_type,
        "menu_category": menu_category,
        "is_holiday": int(is_holiday),
        "special_event": special_event,
        "weather_condition": weather_condition,
        "temperature_c": float(temperature_c),
        "predicted_demand": round(primary_pred, 1),
        "rounded_operational_demand": rounded_demand,
        "baseline_demand": round(base_pred, 1),
        "planning_buffer_portions": planning_buffer,
        "upper_bound_demand": rounded_demand + planning_buffer,
        "lower_bound_demand": max(0, rounded_demand - planning_buffer),
        "model_version": str(artifact.get("model_version", "v1")),
        "is_fallback_baseline": False,
        "contributing_factors": factors,
    }
