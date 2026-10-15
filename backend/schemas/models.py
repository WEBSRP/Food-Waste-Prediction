"""Pydantic schemas for request validation and response serialization."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=4, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class MealRecordCreate(BaseModel):
    service_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    meal_type: str
    menu_category: str = "north_indian_standard"
    menu_items: str = ""
    is_holiday: int = Field(0, ge=0, le=1)
    special_event: str = "none"
    weather_condition: str = "clear"
    temperature_c: float = Field(28.0, ge=-10.0, le=55.0)
    meals_prepared: int = Field(..., ge=0, le=10000)
    meals_served: int = Field(..., ge=0, le=10000)
    leftover_portions: int = Field(..., ge=0, le=10000)
    food_discarded_portions: int = Field(..., ge=0, le=10000)
    unmet_demand_portions: int = Field(0, ge=0, le=10000)
    is_synthetic: int = Field(0, ge=0, le=1)

    @validator("meal_type")
    def validate_meal_type(cls, v: str) -> str:
        val = v.strip().lower()
        if val not in ("breakfast", "lunch", "dinner"):
            raise ValueError("meal_type must be breakfast, lunch, or dinner")
        return val

    @validator("meals_served")
    def validate_served_le_prepared(cls, v: int, values: Dict[str, Any]) -> int:
        prep = values.get("meals_prepared")
        if prep is not None and v > prep:
            raise ValueError("meals_served cannot exceed meals_prepared")
        return v

    @validator("food_discarded_portions")
    def validate_discarded_le_leftover(cls, v: int, values: Dict[str, Any]) -> int:
        leftover = values.get("leftover_portions")
        if leftover is not None and v > leftover:
            raise ValueError("food_discarded_portions cannot exceed leftover_portions")
        return v


class MealRecordUpdate(BaseModel):
    meals_prepared: Optional[int] = Field(None, ge=0, le=10000)
    meals_served: Optional[int] = Field(None, ge=0, le=10000)
    leftover_portions: Optional[int] = Field(None, ge=0, le=10000)
    food_discarded_portions: Optional[int] = Field(None, ge=0, le=10000)
    unmet_demand_portions: Optional[int] = Field(None, ge=0, le=10000)


class ForecastRequest(BaseModel):
    forecast_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    meal_type: str
    menu_category: str = "north_indian_standard"
    is_holiday: int = Field(0, ge=0, le=1)
    special_event: str = "none"
    weather_condition: str = "clear"
    temperature_c: float = Field(28.0, ge=-10.0, le=55.0)
    save_forecast: bool = True

    @validator("meal_type")
    def validate_meal_type(cls, v: str) -> str:
        val = v.strip().lower()
        if val not in ("breakfast", "lunch", "dinner"):
            raise ValueError("meal_type must be breakfast, lunch, or dinner")
        return val


class ProductionPlanRequest(BaseModel):
    service_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    meal_type: str
    forecast_id: Optional[int] = None
    predicted_demand: float = Field(..., ge=0.0, le=10000.0)
    planned_production: int = Field(..., ge=0, le=10000)
    production_capacity: int = Field(..., gt=0, le=10000)
    uncertainty_std: float = Field(20.0, ge=0.0, le=2000.0)
    shortage_risk_tolerance: float = Field(0.15, ge=0.01, le=0.50)
    unit_cost_inr: float = Field(65.0, ge=0.0, le=5000.0)
    shortage_penalty_inr: float = Field(110.0, ge=0.0, le=10000.0)
    custom_buffer_override: Optional[int] = Field(None, ge=0, le=2000)
    save_plan: bool = True


class ProductionPlanActualsUpdate(BaseModel):
    actual_production: int = Field(..., ge=0)
    actual_served: int = Field(..., ge=0)
    actual_leftover: int = Field(..., ge=0)
    actual_discarded: int = Field(..., ge=0)


class RecipientCreate(BaseModel):
    organization_name: str = Field(..., min_length=3, max_length=150)
    recipient_type: str = Field(..., min_length=3, max_length=64)
    contact_person: str = Field(..., min_length=2, max_length=100)
    contact_email: str = Field(..., min_length=5, max_length=120)
    contact_phone: str = Field(..., min_length=6, max_length=32)
    pickup_address: str = Field(..., min_length=5, max_length=255)
    latitude: float = Field(28.4500, ge=-90.0, le=90.0)
    longitude: float = Field(77.5000, ge=-180.0, le=180.0)
    accepted_categories: List[str]
    max_capacity_portions: int = Field(..., gt=0, le=5000)
    operating_hours: str = "08:00-21:00"
    pickup_window_start: str = Field("08:00", regex=r"^\d{2}:\d{2}$")
    pickup_window_end: str = Field("21:00", regex=r"^\d{2}:\d{2}$")
    is_verified: int = Field(1, ge=0, le=1)
    is_active: int = Field(1, ge=0, le=1)
    priority_tier: int = Field(2, ge=1, le=3)
    is_synthetic: int = Field(0, ge=0, le=1)


class SurplusListingCreate(BaseModel):
    service_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    meal_type: str
    food_description: str = Field(..., min_length=3, max_length=255)
    food_category: str
    initial_quantity_portions: int = Field(..., gt=0, le=5000)
    preparation_time: str
    available_from_time: str
    pickup_deadline: str
    safe_use_deadline: str
    storage_condition: str = Field(..., min_length=3, max_length=200)
    handling_notes: str = Field("", max_length=500)
    pickup_location: Optional[str] = None
    safety_approved: bool = False
    is_synthetic: int = Field(0, ge=0, le=1)


class SurplusApprovalRequest(BaseModel):
    handling_verification_notes: str = Field(
        "Temperature and hygienic storage verified by mess supervisor.",
        min_length=3,
        max_length=400,
    )


class OfferCreateRequest(BaseModel):
    listing_id: int = Field(..., gt=0)
    recipient_id: int = Field(..., gt=0)
    offered_quantity_portions: int = Field(..., gt=0, le=5000)
    match_score: float = 0.0
    match_reasons: List[str] = []
    notes: str = ""


class OfferRespondRequest(BaseModel):
    action: str = Field(..., description="ACCEPT, DECLINE, or CANCEL")
    accepted_quantity_portions: Optional[int] = Field(None, gt=0, le=5000)
    notes: str = ""


class CollectionConfirmRequest(BaseModel):
    collected_quantity_portions: int = Field(..., gt=0, le=5000)
    verification_notes: str = Field(
        "Handover verified and signed at mess dispatch gate.",
        max_length=400,
    )
