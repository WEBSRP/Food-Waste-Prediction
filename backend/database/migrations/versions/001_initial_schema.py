"""Initial schema migration for AI-Powered Food Waste Prediction and Redistribution Intelligence.

Revision ID: 001_initial_schema
Create Date: 2026-10-15
"""
from __future__ import annotations

from backend.database.models import SCHEMA_DDL
from backend.database.session import DatabaseSession

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade(db_path: str | None = None) -> None:
    """Apply initial schema DDL creating all 12 tables and indexes."""
    session = DatabaseSession(db_path=db_path)
    try:
        with session.transaction():
            for statement in SCHEMA_DDL:
                session.execute(statement)
    finally:
        session.close()


def downgrade(db_path: str | None = None) -> None:
    """Drop all tables in reverse dependency order."""
    tables = [
        "application_settings",
        "audit_events",
        "redistribution_transactions",
        "redistribution_offers",
        "surplus_listings",
        "production_plans",
        "demand_forecasts",
        "model_evaluation_runs",
        "meal_service_records",
        "users",
        "recipients",
        "roles",
    ]
    session = DatabaseSession(db_path=db_path)
    try:
        with session.transaction():
            for tbl in tables:
                session.execute(f"DROP TABLE IF EXISTS {tbl};")
    finally:
        session.close()
