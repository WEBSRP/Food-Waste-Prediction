"""Core configuration management loaded from environment variables and .env files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"

for _d in (DATA_DIR, SYNTHETIC_DIR, MODELS_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings:
    """Application settings with safe defaults for local CPU execution."""

    APP_NAME: str = os.getenv(
        "APP_NAME", "AI-Powered Food Waste Prediction and Redistribution Intelligence"
    )
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    ORGANIZATION_NAME: str = os.getenv(
        "ORGANIZATION_NAME",
        "Galgotias University — ORIGIN MINI 2026 (Problem Statement 04)",
    )

    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "origin-mini-2026-demo-secret-key-change-in-production-32b"
    )
    TOKEN_EXPIRE_MINUTES: int = int(os.getenv("TOKEN_EXPIRE_MINUTES", "480"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "240"))

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{DATA_DIR / 'food_waste_intelligence.db'}"
    )

    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    # Default Cafeteria Operational Parameters
    DEFAULT_CAPACITIES: Dict[str, int] = {
        "breakfast": int(os.getenv("DEFAULT_BREAKFAST_CAPACITY", "450")),
        "lunch": int(os.getenv("DEFAULT_LUNCH_CAPACITY", "750")),
        "dinner": int(os.getenv("DEFAULT_DINNER_CAPACITY", "650")),
    }
    DEFAULT_SHORTAGE_RISK_TOLERANCE: float = float(
        os.getenv("DEFAULT_SHORTAGE_RISK_TOLERANCE", "0.15")
    )
    DEFAULT_MEAL_UNIT_COST_INR: float = float(
        os.getenv("DEFAULT_MEAL_UNIT_COST_INR", "65.0")
    )
    DEFAULT_SHORTAGE_PENALTY_INR: float = float(
        os.getenv("DEFAULT_SHORTAGE_PENALTY_INR", "110.0")
    )
    REQUIRE_MANUAL_SAFETY_APPROVAL: bool = (
        os.getenv("REQUIRE_MANUAL_SAFETY_APPROVAL", "true").lower() == "true"
    )
    CAFETERIA_LAT: float = float(os.getenv("CAFETERIA_LAT", "28.3670"))
    CAFETERIA_LON: float = float(os.getenv("CAFETERIA_LON", "77.5410"))
    CAFETERIA_LOCATION_NAME: str = os.getenv(
        "CAFETERIA_LOCATION_NAME", "Galgotias University Central Hostel Mess Block A"
    )

    @property
    def sqlite_path(self) -> str:
        """Resolve filesystem path when using SQLite."""
        if self.DATABASE_URL.startswith("sqlite:///"):
            raw = self.DATABASE_URL.replace("sqlite:///", "", 1)
            p = Path(raw)
            if not p.is_absolute():
                p = BASE_DIR / p
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p)
        return str(DATA_DIR / "food_waste_intelligence.db")


settings = Settings()
