"""Relational database models and SQL schema definitions for all 12 core entities.

Supports SQLAlchemy ORM when installed and provides a built-in parameterized SQLite
ORM compatibility layer so the application runs both with SQLAlchemy/PostgreSQL and
in minimal CPU environments using Python's built-in sqlite3 engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SCHEMA_DDL: List[str] = [
    "PRAGMA foreign_keys = ON;",
    """
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL,
        permissions_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS recipients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        organization_name TEXT NOT NULL UNIQUE,
        recipient_type TEXT NOT NULL,
        contact_person TEXT NOT NULL,
        contact_email TEXT NOT NULL,
        contact_phone TEXT NOT NULL,
        pickup_address TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        accepted_categories_json TEXT NOT NULL,
        max_capacity_portions INTEGER NOT NULL CHECK (max_capacity_portions > 0),
        operating_hours TEXT NOT NULL,
        pickup_window_start TEXT NOT NULL,
        pickup_window_end TEXT NOT NULL,
        is_verified INTEGER NOT NULL DEFAULT 1,
        is_active INTEGER NOT NULL DEFAULT 1,
        priority_tier INTEGER NOT NULL DEFAULT 2,
        is_synthetic INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        recipient_id INTEGER NULL REFERENCES recipients(id) ON DELETE SET NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        is_demo_account INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS meal_service_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service_date TEXT NOT NULL,
        meal_type TEXT NOT NULL CHECK (meal_type IN ('breakfast', 'lunch', 'dinner')),
        menu_category TEXT NOT NULL,
        menu_items TEXT NOT NULL DEFAULT '',
        day_of_week INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
        is_holiday INTEGER NOT NULL DEFAULT 0,
        special_event TEXT NOT NULL DEFAULT 'none',
        weather_condition TEXT NOT NULL DEFAULT 'clear',
        temperature_c REAL NOT NULL DEFAULT 28.0,
        meals_prepared INTEGER NOT NULL CHECK (meals_prepared >= 0),
        meals_served INTEGER NOT NULL CHECK (meals_served >= 0),
        leftover_portions INTEGER NOT NULL CHECK (leftover_portions >= 0),
        food_discarded_portions INTEGER NOT NULL CHECK (food_discarded_portions >= 0),
        unmet_demand_portions INTEGER NOT NULL DEFAULT 0 CHECK (unmet_demand_portions >= 0),
        is_synthetic INTEGER NOT NULL DEFAULT 1,
        recorded_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        UNIQUE(service_date, meal_type)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_meal_records_date_type ON meal_service_records(service_date, meal_type);",
    """
    CREATE TABLE IF NOT EXISTS model_evaluation_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL UNIQUE,
        model_name TEXT NOT NULL,
        baseline_name TEXT NOT NULL,
        train_start_date TEXT NOT NULL,
        train_end_date TEXT NOT NULL,
        eval_start_date TEXT NOT NULL,
        eval_end_date TEXT NOT NULL,
        train_samples INTEGER NOT NULL,
        eval_samples INTEGER NOT NULL,
        ml_mae REAL NOT NULL,
        ml_rmse REAL NOT NULL,
        ml_wape REAL NOT NULL,
        ml_bias REAL NOT NULL,
        baseline_mae REAL NOT NULL,
        baseline_rmse REAL NOT NULL,
        baseline_wape REAL NOT NULL,
        baseline_bias REAL NOT NULL,
        selected_model TEXT NOT NULL,
        residual_std REAL NOT NULL,
        residual_q85 REAL NOT NULL,
        feature_importances_json TEXT NOT NULL,
        comparison_series_json TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS demand_forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_date TEXT NOT NULL,
        meal_type TEXT NOT NULL CHECK (meal_type IN ('breakfast', 'lunch', 'dinner')),
        menu_category TEXT NOT NULL,
        is_holiday INTEGER NOT NULL DEFAULT 0,
        special_event TEXT NOT NULL DEFAULT 'none',
        weather_condition TEXT NOT NULL DEFAULT 'clear',
        temperature_c REAL NOT NULL DEFAULT 28.0,
        predicted_demand REAL NOT NULL CHECK (predicted_demand >= 0),
        rounded_operational_demand INTEGER NOT NULL CHECK (rounded_operational_demand >= 0),
        baseline_demand REAL NOT NULL CHECK (baseline_demand >= 0),
        planning_buffer_portions INTEGER NOT NULL CHECK (planning_buffer_portions >= 0),
        upper_bound_demand INTEGER NOT NULL CHECK (upper_bound_demand >= 0),
        lower_bound_demand INTEGER NOT NULL CHECK (lower_bound_demand >= 0),
        model_version TEXT NOT NULL,
        is_fallback_baseline INTEGER NOT NULL DEFAULT 0,
        contributing_factors_json TEXT NOT NULL,
        created_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_forecasts_date_meal ON demand_forecasts(forecast_date, meal_type);",
    """
    CREATE TABLE IF NOT EXISTS production_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service_date TEXT NOT NULL,
        meal_type TEXT NOT NULL CHECK (meal_type IN ('breakfast', 'lunch', 'dinner')),
        forecast_id INTEGER NULL REFERENCES demand_forecasts(id) ON DELETE SET NULL,
        predicted_demand REAL NOT NULL,
        planned_production INTEGER NOT NULL CHECK (planned_production >= 0),
        recommended_production INTEGER NOT NULL CHECK (recommended_production >= 0),
        production_capacity INTEGER NOT NULL CHECK (production_capacity > 0),
        shortage_risk_tolerance REAL NOT NULL,
        safety_buffer_portions INTEGER NOT NULL,
        expected_surplus_existing_plan INTEGER NOT NULL,
        expected_surplus_recommended_plan INTEGER NOT NULL,
        expected_shortage_risk_existing REAL NOT NULL,
        expected_shortage_risk_recommended REAL NOT NULL,
        actual_production INTEGER NULL,
        actual_served INTEGER NULL,
        actual_leftover INTEGER NULL,
        actual_discarded INTEGER NULL,
        unit_cost_inr REAL NOT NULL DEFAULT 65.0,
        shortage_penalty_inr REAL NOT NULL DEFAULT 110.0,
        explanation TEXT NOT NULL,
        created_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_plans_date_meal ON production_plans(service_date, meal_type);",
    """
    CREATE TABLE IF NOT EXISTS surplus_listings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        listing_code TEXT NOT NULL UNIQUE,
        service_date TEXT NOT NULL,
        meal_type TEXT NOT NULL,
        food_description TEXT NOT NULL,
        food_category TEXT NOT NULL,
        initial_quantity_portions INTEGER NOT NULL CHECK (initial_quantity_portions > 0),
        remaining_quantity_portions INTEGER NOT NULL CHECK (remaining_quantity_portions >= 0),
        reserved_quantity_portions INTEGER NOT NULL DEFAULT 0 CHECK (reserved_quantity_portions >= 0),
        collected_quantity_portions INTEGER NOT NULL DEFAULT 0 CHECK (collected_quantity_portions >= 0),
        unit TEXT NOT NULL DEFAULT 'portions',
        preparation_time TEXT NOT NULL,
        safe_use_deadline TEXT NOT NULL,
        available_from_time TEXT NOT NULL,
        pickup_deadline TEXT NOT NULL,
        storage_condition TEXT NOT NULL,
        handling_notes TEXT NOT NULL,
        pickup_location TEXT NOT NULL,
        latitude REAL NOT NULL DEFAULT 28.3670,
        longitude REAL NOT NULL DEFAULT 77.5410,
        safety_approved INTEGER NOT NULL DEFAULT 0,
        approved_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        approved_at TEXT NULL,
        status TEXT NOT NULL CHECK (status IN ('DRAFT', 'AVAILABLE', 'RESERVED', 'PARTIALLY_RESERVED', 'COLLECTED', 'CANCELLED', 'EXPIRED')),
        is_synthetic INTEGER NOT NULL DEFAULT 1,
        created_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_surplus_status_deadline ON surplus_listings(status, pickup_deadline);",
    """
    CREATE TABLE IF NOT EXISTS redistribution_offers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        offer_code TEXT NOT NULL UNIQUE,
        listing_id INTEGER NOT NULL REFERENCES surplus_listings(id) ON DELETE CASCADE,
        recipient_id INTEGER NOT NULL REFERENCES recipients(id) ON DELETE CASCADE,
        offered_quantity_portions INTEGER NOT NULL CHECK (offered_quantity_portions > 0),
        accepted_quantity_portions INTEGER NOT NULL DEFAULT 0 CHECK (accepted_quantity_portions >= 0),
        collected_quantity_portions INTEGER NOT NULL DEFAULT 0 CHECK (collected_quantity_portions >= 0),
        uncollected_quantity_portions INTEGER NOT NULL DEFAULT 0 CHECK (uncollected_quantity_portions >= 0),
        match_score REAL NOT NULL DEFAULT 0.0,
        match_reasons_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'ACCEPTED', 'DECLINED', 'COLLECTED', 'CANCELLED', 'EXPIRED')),
        notes TEXT NOT NULL DEFAULT '',
        created_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        responded_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        responded_at TEXT NULL,
        collected_at TEXT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_offers_listing_recipient ON redistribution_offers(listing_id, recipient_id, status);",
    """
    CREATE TABLE IF NOT EXISTS redistribution_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        offer_id INTEGER NOT NULL REFERENCES redistribution_offers(id) ON DELETE CASCADE,
        listing_id INTEGER NOT NULL REFERENCES surplus_listings(id) ON DELETE CASCADE,
        recipient_id INTEGER NOT NULL REFERENCES recipients(id) ON DELETE CASCADE,
        action_type TEXT NOT NULL CHECK (action_type IN ('OFFER_CREATED', 'OFFER_ACCEPTED', 'OFFER_DECLINED', 'COLLECTION_CONFIRMED', 'OFFER_CANCELLED', 'LISTING_EXPIRED')),
        quantity_portions INTEGER NOT NULL CHECK (quantity_portions >= 0),
        actor_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        verification_notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        actor_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        actor_role TEXT NOT NULL,
        details_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS application_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_key TEXT NOT NULL UNIQUE,
        setting_value TEXT NOT NULL,
        description TEXT NOT NULL,
        updated_by_user_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
        updated_at TEXT NOT NULL
    );
    """,
]


@dataclass
class UserRecord:
    id: Optional[int]
    username: str
    full_name: str
    email: str
    password_hash: str
    role: str
    recipient_id: Optional[int] = None
    is_active: int = 1
    is_demo_account: int = 1
    created_at: str = field(default_factory=utc_now_iso)

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("password_hash", None)
        d["is_active"] = bool(self.is_active)
        d["is_demo_account"] = bool(self.is_demo_account)
        return d
