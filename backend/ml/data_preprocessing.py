"""Data validation, preprocessing, outlier inspection, and true-demand estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import io

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "service_date",
    "meal_type",
    "menu_category",
    "meals_prepared",
    "meals_served",
    "leftover_portions",
    "food_discarded_portions",
}

VALID_MEAL_TYPES = {"breakfast", "lunch", "dinner"}
MAX_CSV_BYTES = 10 * 1024 * 1024  # 10 MB upload limit


@dataclass
class ValidationReport:
    is_valid: bool
    total_rows: int
    valid_rows: int
    duplicate_count: int = 0
    missing_value_count: int = 0
    impossible_value_count: int = 0
    outlier_count: int = 0
    censored_demand_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "duplicate_count": self.duplicate_count,
            "missing_value_count": self.missing_value_count,
            "impossible_value_count": self.impossible_value_count,
            "outlier_count": self.outlier_count,
            "censored_demand_count": self.censored_demand_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary_stats": self.summary_stats,
        }


def parse_and_validate_csv_bytes(content: bytes, filename: str = "upload.csv") -> Tuple[Optional[pd.DataFrame], ValidationReport]:
    """Safely parse and validate uploaded CSV bytes without executing embedded formulas or code."""
    if not filename.lower().endswith(".csv"):
        return None, ValidationReport(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=["Unsupported file extension. Only .csv files are permitted."],
        )
    if len(content) == 0:
        return None, ValidationReport(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=["Uploaded CSV file is empty."],
        )
    if len(content) > MAX_CSV_BYTES:
        return None, ValidationReport(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=[f"File size exceeds maximum limit of {MAX_CSV_BYTES // (1024 * 1024)} MB."],
        )
    try:
        text = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        return None, ValidationReport(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=[f"Failed to parse CSV: {exc}"],
        )
    return validate_meal_dataframe(df)


def validate_meal_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], ValidationReport]:
    """Validate schema, types, dates, duplicates, impossible values, and outliers."""
    errors: List[str] = []
    warnings: List[str] = []
    total_rows = int(len(df))

    if total_rows == 0:
        return None, ValidationReport(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=["Dataset contains 0 rows."],
        )

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return None, ValidationReport(
            is_valid=False,
            total_rows=total_rows,
            valid_rows=0,
            errors=[f"Missing required columns: {sorted(missing_cols)}"],
        )

    work = df.copy()

    # Check missing values in required columns
    missing_mask = work[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        warnings.append(f"Detected {missing_count} row(s) with missing required values.")

    # Normalize strings & sanitize against CSV injection characters
    for str_col in ["meal_type", "menu_category", "menu_items", "special_event", "weather_condition"]:
        if str_col in work.columns:
            work[str_col] = (
                work[str_col]
                .fillna("none" if str_col != "menu_items" else "")
                .astype(str)
                .str.strip()
                .str.lstrip("=+-@")
            )

    work["meal_type"] = work["meal_type"].str.lower()
    invalid_meals = set(work["meal_type"].dropna().unique()) - VALID_MEAL_TYPES
    if invalid_meals:
        errors.append(f"Invalid meal_type values found: {sorted(invalid_meals)}. Allowed: breakfast, lunch, dinner.")

    # Parse dates
    parsed_dates = pd.to_datetime(work["service_date"], format="%Y-%m-%d", errors="coerce")
    bad_dates = int(parsed_dates.isna().sum())
    if bad_dates > 0:
        errors.append(f"Found {bad_dates} row(s) with invalid service_date (expected YYYY-MM-DD).")

    # Coerce numeric columns
    num_cols = ["meals_prepared", "meals_served", "leftover_portions", "food_discarded_portions"]
    if "unmet_demand_portions" not in work.columns:
        work["unmet_demand_portions"] = 0
    num_cols.append("unmet_demand_portions")

    for col in num_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    numeric_na = int(work[num_cols].isna().any(axis=1).sum())
    if numeric_na > missing_count:
        errors.append(f"Found {numeric_na} row(s) with non-numeric quantity values.")

    # Duplicate check on (service_date, meal_type)
    dup_mask = work.duplicated(subset=["service_date", "meal_type"], keep="first")
    duplicate_count = int(dup_mask.sum())
    if duplicate_count > 0:
        warnings.append(
            f"Found {duplicate_count} duplicate (service_date, meal_type) record(s); keeping first occurrence."
        )

    # Impossible values check: negative numbers, served > prepared, discarded > leftover
    impossible_mask = (
        (work["meals_prepared"] < 0)
        | (work["meals_served"] < 0)
        | (work["leftover_portions"] < 0)
        | (work["food_discarded_portions"] < 0)
        | (work["unmet_demand_portions"] < 0)
        | (work["meals_served"] > work["meals_prepared"])
        | (work["food_discarded_portions"] > work["leftover_portions"])
    )
    impossible_count = int(impossible_mask.sum())
    if impossible_count > 0:
        errors.append(
            f"Found {impossible_count} row(s) with impossible physical quantities "
            "(e.g., negative values, meals_served > meals_prepared, or food_discarded > leftover_portions)."
        )

    if errors:
        return None, ValidationReport(
            is_valid=False,
            total_rows=total_rows,
            valid_rows=0,
            duplicate_count=duplicate_count,
            missing_value_count=missing_count,
            impossible_value_count=impossible_count,
            errors=errors,
            warnings=warnings,
        )

    # Drop missing and duplicates if only warnings
    clean = work.loc[~missing_mask & ~dup_mask].copy()
    clean["service_date_dt"] = pd.to_datetime(clean["service_date"], format="%Y-%m-%d")
    clean["day_of_week"] = clean["service_date_dt"].dt.weekday
    clean["is_holiday"] = pd.to_numeric(clean.get("is_holiday", 0), errors="coerce").fillna(0).astype(int)
    clean["special_event"] = clean.get("special_event", "none")
    clean["weather_condition"] = clean.get("weather_condition", "clear")
    clean["temperature_c"] = pd.to_numeric(clean.get("temperature_c", 28.0), errors="coerce").fillna(28.0)
    clean["is_synthetic"] = pd.to_numeric(clean.get("is_synthetic", 0), errors="coerce").fillna(0).astype(int)

    for col in num_cols:
        clean[col] = clean[col].astype(int)

    # Chronological sorting
    meal_order = {"breakfast": 0, "lunch": 1, "dinner": 2}
    clean["_meal_order"] = clean["meal_type"].map(meal_order)
    clean = clean.sort_values(["service_date_dt", "_meal_order"]).drop(columns=["_meal_order"]).reset_index(drop=True)

    # True demand vs observed consumption distinction
    clean["effective_demand"] = clean["meals_served"] + clean["unmet_demand_portions"]
    censored_mask = (clean["meals_served"] >= clean["meals_prepared"]) & (clean["unmet_demand_portions"] == 0)
    censored_count = int(censored_mask.sum())
    if censored_count > 0:
        warnings.append(
            f"{censored_count} service record(s) had meals_served == meals_prepared with 0 recorded unmet demand. "
            "When food runs out and unmet demand is unrecorded, meals_served is right-censored and may underestimate true demand."
        )

    # Outlier inspection per meal_type using IQR (1.5x)
    outlier_count = 0
    for m_type, grp in clean.groupby("meal_type"):
        if len(grp) >= 10:
            q25 = float(grp["effective_demand"].quantile(0.25))
            q75 = float(grp["effective_demand"].quantile(0.75))
            iqr = max(1.0, q75 - q25)
            outliers = (grp["effective_demand"] < (q25 - 2.0 * iqr)) | (grp["effective_demand"] > (q75 + 2.0 * iqr))
            outlier_count += int(outliers.sum())

    if outlier_count > 0:
        warnings.append(
            f"Inspected demand distribution: {outlier_count} extreme demand observation(s) flagged outside 2.0x IQR."
        )

    summary_stats = {
        "date_min": str(clean["service_date"].min()),
        "date_max": str(clean["service_date"].max()),
        "mean_prepared": round(float(clean["meals_prepared"].mean()), 1),
        "mean_served": round(float(clean["meals_served"].mean()), 1),
        "mean_effective_demand": round(float(clean["effective_demand"].mean()), 1),
        "mean_leftover": round(float(clean["leftover_portions"].mean()), 1),
        "mean_discarded": round(float(clean["food_discarded_portions"].mean()), 1),
        "by_meal_type": {
            m: round(float(g["effective_demand"].mean()), 1)
            for m, g in clean.groupby("meal_type")
        },
    }

    report = ValidationReport(
        is_valid=True,
        total_rows=total_rows,
        valid_rows=len(clean),
        duplicate_count=duplicate_count,
        missing_value_count=missing_count,
        impossible_value_count=impossible_count,
        outlier_count=outlier_count,
        censored_demand_count=censored_count,
        errors=errors,
        warnings=warnings,
        summary_stats=summary_stats,
    )
    return clean.drop(columns=["service_date_dt"]), report
