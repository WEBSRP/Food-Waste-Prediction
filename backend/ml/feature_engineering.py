"""Chronological feature engineering with strict prevention of target leakage.

All historical lag and rolling statistics for a target date `t` are computed strictly
from observations with `service_date < t`. Future or same-day target values are never
accessible during feature construction.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MEAL_TYPE_MAP = {"breakfast": 0, "lunch": 1, "dinner": 2}
MENU_CATEGORY_MAP = {
    "north_indian_standard": 0,
    "south_indian_special": 1,
    "continental_fusion": 2,
    "festive_thali": 3,
    "healthy_millet_bowl": 4,
}
SPECIAL_EVENT_MAP = {
    "none": 0,
    "exam_week": 1,
    "sports_meet": 2,
    "campus_hackathon_fest": 3,
}
WEATHER_MAP = {
    "clear": 0,
    "rainy": 1,
    "extreme_heat": 2,
    "cold_fog": 3,
}

FEATURE_COLUMNS = [
    "meal_type_code",
    "menu_category_code",
    "day_of_week",
    "is_weekend",
    "is_friday",
    "is_holiday",
    "special_event_code",
    "weather_code",
    "temperature_c",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "lag_1d_demand",
    "lag_7d_demand",
    "rolling_7d_mean",
    "rolling_14d_mean",
    "rolling_7d_std",
    "same_weekday_4w_mean",
]


def _compute_past_lags_for_meal(history_demands: List[float], history_dows: List[int], target_dow: int) -> Dict[str, float]:
    """Compute lag and rolling features strictly from prior records of the same meal_type."""
    if not history_demands:
        return {
            "lag_1d_demand": 400.0,
            "lag_7d_demand": 400.0,
            "rolling_7d_mean": 400.0,
            "rolling_14d_mean": 400.0,
            "rolling_7d_std": 25.0,
            "same_weekday_4w_mean": 400.0,
        }
    arr = np.asarray(history_demands, dtype=float)
    lag_1 = float(arr[-1])
    lag_7 = float(arr[-7]) if len(arr) >= 7 else float(np.mean(arr))
    win_7 = arr[-7:]
    win_14 = arr[-14:]
    r7_mean = float(np.mean(win_7))
    r14_mean = float(np.mean(win_14))
    r7_std = float(np.std(win_7)) if len(win_7) >= 2 else 15.0

    # Last up to 4 occurrences of the same day_of_week in past history
    same_dow_vals = [
        val for val, d in zip(history_demands, history_dows) if d == target_dow
    ]
    if same_dow_vals:
        same_dow_mean = float(np.mean(same_dow_vals[-4:]))
    else:
        same_dow_mean = r7_mean

    return {
        "lag_1d_demand": lag_1,
        "lag_7d_demand": lag_7,
        "rolling_7d_mean": r7_mean,
        "rolling_14d_mean": r14_mean,
        "rolling_7d_std": r7_std,
        "same_weekday_4w_mean": same_dow_mean,
    }


def build_supervised_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transform chronological meal records into a leakage-free supervised feature matrix."""
    work = df.copy()
    if "effective_demand" not in work.columns:
        unmet = work["unmet_demand_portions"] if "unmet_demand_portions" in work.columns else 0
        work["effective_demand"] = work["meals_served"] + unmet

    work["service_date_dt"] = pd.to_datetime(work["service_date"], format="%Y-%m-%d")
    work["_meal_order"] = work["meal_type"].map(MEAL_TYPE_MAP).fillna(0)
    work = work.sort_values(["service_date_dt", "_meal_order"]).reset_index(drop=True)

    # Track strictly past history per meal_type
    past_demands: Dict[str, List[float]] = {"breakfast": [], "lunch": [], "dinner": []}
    past_dows: Dict[str, List[int]] = {"breakfast": [], "lunch": [], "dinner": []}

    feature_rows: List[Dict[str, Any]] = []
    for idx, row in work.iterrows():
        m_type = str(row["meal_type"]).lower()
        dt = row["service_date_dt"]
        dow = int(dt.weekday())
        month = int(dt.month)

        lags = _compute_past_lags_for_meal(
            past_demands.get(m_type, []),
            past_dows.get(m_type, []),
            dow,
        )

        feat = {
            "service_date": str(row["service_date"]),
            "meal_type": m_type,
            "menu_category": str(row["menu_category"]),
            "meal_type_code": float(MEAL_TYPE_MAP.get(m_type, 0)),
            "menu_category_code": float(MENU_CATEGORY_MAP.get(str(row["menu_category"]), 0)),
            "day_of_week": float(dow),
            "is_weekend": 1.0 if dow in (5, 6) else 0.0,
            "is_friday": 1.0 if dow == 4 else 0.0,
            "is_holiday": float(int(row.get("is_holiday", 0))),
            "special_event_code": float(SPECIAL_EVENT_MAP.get(str(row.get("special_event", "none")), 0)),
            "weather_code": float(WEATHER_MAP.get(str(row.get("weather_condition", "clear")), 0)),
            "temperature_c": float(row.get("temperature_c", 28.0)),
            "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
            "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
            "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
            "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
            **lags,
            "target_demand": float(row["effective_demand"]),
            "meals_served": float(row["meals_served"]),
            "meals_prepared": float(row["meals_prepared"]),
        }
        feature_rows.append(feat)

        # Append current observation to history ONLY AFTER building features for row `idx`
        past_demands.setdefault(m_type, []).append(float(row["effective_demand"]))
        past_dows.setdefault(m_type, []).append(dow)

    return pd.DataFrame(feature_rows)


def build_single_forecast_features(
    historical_df: pd.DataFrame,
    forecast_date: str,
    meal_type: str,
    menu_category: str = "north_indian_standard",
    is_holiday: int = 0,
    special_event: str = "none",
    weather_condition: str = "clear",
    temperature_c: float = 28.0,
) -> Dict[str, float]:
    """Construct forecast-time features for a single upcoming meal strictly from records < forecast_date."""
    m_type = meal_type.lower().strip()
    dt = pd.to_datetime(forecast_date, format="%Y-%m-%d")
    dow = int(dt.weekday())
    month = int(dt.month)

    past_demands: List[float] = []
    past_dows: List[int] = []

    if historical_df is not None and not historical_df.empty:
        sub = historical_df[
            (historical_df["meal_type"].str.lower() == m_type)
            & (historical_df["service_date"] < forecast_date)
        ].sort_values("service_date")
        if not sub.empty:
            unmet = sub["unmet_demand_portions"] if "unmet_demand_portions" in sub.columns else 0
            eff = sub["meals_served"] + unmet
            past_demands = [float(x) for x in eff.tolist()]
            past_dows = [int(pd.to_datetime(d).weekday()) for d in sub["service_date"].tolist()]

    lags = _compute_past_lags_for_meal(past_demands, past_dows, dow)

    return {
        "meal_type_code": float(MEAL_TYPE_MAP.get(m_type, 0)),
        "menu_category_code": float(MENU_CATEGORY_MAP.get(menu_category, 0)),
        "day_of_week": float(dow),
        "is_weekend": 1.0 if dow in (5, 6) else 0.0,
        "is_friday": 1.0 if dow == 4 else 0.0,
        "is_holiday": float(int(is_holiday)),
        "special_event_code": float(SPECIAL_EVENT_MAP.get(special_event, 0)),
        "weather_code": float(WEATHER_MAP.get(weather_condition, 0)),
        "temperature_c": float(temperature_c),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        **lags,
    }
