"""REST API routes for AI-Powered Food Waste Prediction and Redistribution Intelligence."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
import pandas as pd

from backend.core.config import settings
from backend.core.security import create_access_token, verify_password
from backend.database.models import utc_now_iso
from backend.database.session import DatabaseSession, get_db
from backend.dependencies.auth import check_rate_limit, get_current_user, require_permission
from backend.domain.impact_calculator import compute_system_impact_report
from backend.domain.production_planner import compute_production_recommendation
from backend.domain.recipient_matcher import match_and_rank_recipients
from backend.domain.surplus_detector import (
    approve_surplus_listing_for_redistribution,
    confirm_collection,
    create_redistribution_offer,
    create_surplus_listing,
    respond_to_offer,
)
from backend.ml.data_preprocessing import parse_and_validate_csv_bytes, validate_meal_dataframe
from backend.ml.predict_demand import predict_meal_demand
from backend.ml.train_model import train_and_evaluate_pipeline
from backend.schemas.models import (
    CollectionConfirmRequest,
    ForecastRequest,
    LoginRequest,
    MealRecordCreate,
    MealRecordUpdate,
    OfferCreateRequest,
    OfferRespondRequest,
    ProductionPlanActualsUpdate,
    ProductionPlanRequest,
    RecipientCreate,
    SurplusApprovalRequest,
    SurplusListingCreate,
    TokenResponse,
)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(check_rate_limit)])


@router.get("/health", tags=["System"])
def health_check(db: DatabaseSession = Depends(get_db)) -> Dict[str, Any]:
    """System health check verifying database connectivity and model availability."""
    meal_count = db.fetch_one("SELECT COUNT(*) AS cnt FROM meal_service_records")
    latest_eval = db.fetch_one("SELECT model_version, selected_model, ml_mae FROM model_evaluation_runs ORDER BY id DESC LIMIT 1")
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "organization": settings.ORGANIZATION_NAME,
        "database_connected": True,
        "meal_records_count": int(meal_count["cnt"]) if meal_count else 0,
        "latest_model": latest_eval,
        "timestamp": utc_now_iso(),
    }


@router.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login(payload: LoginRequest, db: DatabaseSession = Depends(get_db)) -> Dict[str, Any]:
    """Authenticate a user and issue a signed Bearer access token."""
    user = db.fetch_one("SELECT * FROM users WHERE username = ?", (payload.username.strip(),))
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    if int(user.get("is_active", 0)) != 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )
    token = create_access_token(
        user_id=int(user["id"]),
        username=str(user["username"]),
        role=str(user["role"]),
        recipient_id=user.get("recipient_id"),
    )
    pub_user = {
        "id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"],
        "recipient_id": user.get("recipient_id"),
        "is_demo_account": bool(user.get("is_demo_account", 1)),
    }
    return {"access_token": token, "token_type": "bearer", "user": pub_user}


@router.get("/auth/me", tags=["Authentication"])
def read_current_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return currently authenticated user profile."""
    return current_user


@router.get("/auth/demo-accounts", tags=["Authentication"])
def list_demo_accounts(db: DatabaseSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List available synthetic demo accounts (without password hashes) for college evaluation."""
    rows = db.fetch_all(
        """
        SELECT u.id, u.username, u.full_name, u.email, u.role, u.recipient_id, r.organization_name
        FROM users u
        LEFT JOIN recipients r ON u.recipient_id = r.id
        WHERE u.is_demo_account = 1 AND u.is_active = 1
        ORDER BY u.id ASC
        """
    )
    return rows


# ---------------------------------------------------------------------------
# Meal Service Records & CSV Validation/Import
# ---------------------------------------------------------------------------
@router.get("/meals", tags=["Meal Records"])
def list_meal_records(
    meal_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("meals:read")),
) -> Dict[str, Any]:
    """List historical meal-service records with optional filtering and pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if meal_type:
        clauses.append("meal_type = ?")
        params.append(meal_type.lower().strip())
    if start_date:
        clauses.append("service_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("service_date <= ?")
        params.append(end_date)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    total_row = db.fetch_one(f"SELECT COUNT(*) AS cnt FROM meal_service_records {where_sql}", params)
    records = db.fetch_all(
        f"""
        SELECT * FROM meal_service_records
        {where_sql}
        ORDER BY service_date DESC, id DESC
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )
    return {
        "total": int(total_row["cnt"]) if total_row else 0,
        "limit": limit,
        "offset": offset,
        "items": records,
    }


@router.post("/meals", status_code=status.HTTP_201_CREATED, tags=["Meal Records"])
def create_meal_record(
    payload: MealRecordCreate,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("meals:write")),
) -> Dict[str, Any]:
    """Create or upsert a single meal-service record."""
    dt = pd.to_datetime(payload.service_date, format="%Y-%m-%d")
    dow = int(dt.weekday())
    now_iso = utc_now_iso()

    existing = db.fetch_one(
        "SELECT id FROM meal_service_records WHERE service_date = ? AND meal_type = ?",
        (payload.service_date, payload.meal_type),
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Meal record already exists for {payload.service_date} ({payload.meal_type}). Use PUT to update.",
        )

    with db.transaction():
        rec_id = db.insert(
            "meal_service_records",
            {
                "service_date": payload.service_date,
                "meal_type": payload.meal_type,
                "menu_category": payload.menu_category,
                "menu_items": payload.menu_items,
                "day_of_week": dow,
                "is_holiday": payload.is_holiday,
                "special_event": payload.special_event,
                "weather_condition": payload.weather_condition,
                "temperature_c": payload.temperature_c,
                "meals_prepared": payload.meals_prepared,
                "meals_served": payload.meals_served,
                "leftover_portions": payload.leftover_portions,
                "food_discarded_portions": payload.food_discarded_portions,
                "unmet_demand_portions": payload.unmet_demand_portions,
                "is_synthetic": payload.is_synthetic,
                "recorded_by_user_id": current_user["id"],
                "created_at": now_iso,
            },
        )
        db.log_audit(
            event_type="MEAL_RECORD_CREATED",
            entity_type="meal_service_record",
            entity_id=rec_id,
            actor_user_id=current_user["id"],
            actor_role=current_user["role"],
            details={"service_date": payload.service_date, "meal_type": payload.meal_type},
        )
        created = db.fetch_one("SELECT * FROM meal_service_records WHERE id = ?", (rec_id,))
    return created or {}


@router.put("/meals/{record_id}", tags=["Meal Records"])
def update_meal_record(
    record_id: int,
    payload: MealRecordUpdate,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("meals:write")),
) -> Dict[str, Any]:
    """Update quantities on an existing meal-service record."""
    existing = db.fetch_one("SELECT * FROM meal_service_records WHERE id = ?", (record_id,))
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meal record not found.")

    prep = payload.meals_prepared if payload.meals_prepared is not None else int(existing["meals_prepared"])
    served = payload.meals_served if payload.meals_served is not None else int(existing["meals_served"])
    leftover = payload.leftover_portions if payload.leftover_portions is not None else int(existing["leftover_portions"])
    discarded = (
        payload.food_discarded_portions
        if payload.food_discarded_portions is not None
        else int(existing["food_discarded_portions"])
    )
    unmet = (
        payload.unmet_demand_portions
        if payload.unmet_demand_portions is not None
        else int(existing["unmet_demand_portions"])
    )

    if served > prep:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="meals_served cannot exceed meals_prepared.")
    if discarded > leftover:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="food_discarded_portions cannot exceed leftover_portions.",
        )

    with db.transaction():
        db.update_by_id(
            "meal_service_records",
            record_id,
            {
                "meals_prepared": prep,
                "meals_served": served,
                "leftover_portions": leftover,
                "food_discarded_portions": discarded,
                "unmet_demand_portions": unmet,
            },
        )
        updated = db.fetch_one("SELECT * FROM meal_service_records WHERE id = ?", (record_id,))
    return updated or {}


@router.post("/meals/validate-csv", tags=["Meal Records"])
async def validate_csv_upload(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_permission("meals:write")),
) -> Dict[str, Any]:
    """Validate an uploaded CSV file without persisting rows."""
    content = await file.read()
    _, report = parse_and_validate_csv_bytes(content, filename=file.filename or "upload.csv")
    return report.to_dict()


@router.post("/meals/import-csv", tags=["Meal Records"])
async def import_csv_upload(
    file: UploadFile = File(...),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("meals:write")),
) -> Dict[str, Any]:
    """Validate and import historical meal records from a CSV file."""
    content = await file.read()
    clean_df, report = parse_and_validate_csv_bytes(content, filename=file.filename or "upload.csv")
    if not report.is_valid or clean_df is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "CSV validation failed.", "validation_report": report.to_dict()},
        )

    inserted = 0
    updated = 0
    now_iso = utc_now_iso()
    with db.transaction():
        for _, row in clean_df.iterrows():
            existing = db.fetch_one(
                "SELECT id FROM meal_service_records WHERE service_date = ? AND meal_type = ?",
                (str(row["service_date"]), str(row["meal_type"])),
            )
            record_data = {
                "service_date": str(row["service_date"]),
                "meal_type": str(row["meal_type"]),
                "menu_category": str(row["menu_category"]),
                "menu_items": str(row.get("menu_items", "")),
                "day_of_week": int(row["day_of_week"]),
                "is_holiday": int(row["is_holiday"]),
                "special_event": str(row["special_event"]),
                "weather_condition": str(row["weather_condition"]),
                "temperature_c": float(row["temperature_c"]),
                "meals_prepared": int(row["meals_prepared"]),
                "meals_served": int(row["meals_served"]),
                "leftover_portions": int(row["leftover_portions"]),
                "food_discarded_portions": int(row["food_discarded_portions"]),
                "unmet_demand_portions": int(row["unmet_demand_portions"]),
                "is_synthetic": int(row.get("is_synthetic", 0)),
                "recorded_by_user_id": current_user["id"],
            }
            if existing:
                db.update_by_id("meal_service_records", int(existing["id"]), record_data)
                updated += 1
            else:
                record_data["created_at"] = now_iso
                db.insert("meal_service_records", record_data)
                inserted += 1

        db.log_audit(
            event_type="CSV_MEAL_RECORDS_IMPORTED",
            entity_type="meal_service_records",
            entity_id=file.filename or "upload.csv",
            actor_user_id=current_user["id"],
            actor_role=current_user["role"],
            details={"inserted": inserted, "updated": updated, "valid_rows": report.valid_rows},
        )

    return {
        "inserted_records": inserted,
        "updated_records": updated,
        "validation_report": report.to_dict(),
    }


# ---------------------------------------------------------------------------
# Forecasting & Model Training / Evaluation
# ---------------------------------------------------------------------------
@router.post("/models/train", tags=["Forecasting & ML"])
def train_and_evaluate_models(
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("models:train")),
) -> Dict[str, Any]:
    """Train and evaluate Baseline and ML forecasting models on chronological historical records."""
    rows = db.fetch_all("SELECT * FROM meal_service_records ORDER BY service_date ASC, id ASC")
    if len(rows) < 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least 30 historical meal records are required to train models (found {len(rows)}).",
        )
    df = pd.DataFrame(rows)
    clean_df, report = validate_meal_dataframe(df)
    if clean_df is None or not report.is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=report.errors)

    eval_result = train_and_evaluate_pipeline(clean_df)
    now_iso = utc_now_iso()

    with db.transaction():
        run_id = db.insert(
            "model_evaluation_runs",
            {
                "model_version": eval_result["model_version"],
                "model_name": eval_result["model_name"],
                "baseline_name": eval_result["baseline_name"],
                "train_start_date": eval_result["train_start_date"],
                "train_end_date": eval_result["train_end_date"],
                "eval_start_date": eval_result["eval_start_date"],
                "eval_end_date": eval_result["eval_end_date"],
                "train_samples": eval_result["train_samples"],
                "eval_samples": eval_result["eval_samples"],
                "ml_mae": eval_result["ml_mae"],
                "ml_rmse": eval_result["ml_rmse"],
                "ml_wape": eval_result["ml_wape"],
                "ml_bias": eval_result["ml_bias"],
                "baseline_mae": eval_result["baseline_mae"],
                "baseline_rmse": eval_result["baseline_rmse"],
                "baseline_wape": eval_result["baseline_wape"],
                "baseline_bias": eval_result["baseline_bias"],
                "selected_model": eval_result["selected_model"],
                "residual_std": eval_result["residual_std"],
                "residual_q85": eval_result["residual_q85"],
                "feature_importances_json": json.dumps(eval_result["feature_importances"]),
                "comparison_series_json": json.dumps(eval_result["comparison_series"]),
                "artifact_path": eval_result["artifact_path"],
                "created_at": now_iso,
            },
        )
        db.log_audit(
            event_type="MODEL_TRAINED_AND_EVALUATED",
            entity_type="model_evaluation_run",
            entity_id=run_id,
            actor_user_id=current_user["id"],
            actor_role=current_user["role"],
            details={
                "model_version": eval_result["model_version"],
                "ml_mae": eval_result["ml_mae"],
                "baseline_mae": eval_result["baseline_mae"],
                "selected_model": eval_result["selected_model"],
            },
        )
    return eval_result


@router.get("/models/evaluations", tags=["Forecasting & ML"])
def list_model_evaluations(
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("forecasts:read")),
) -> List[Dict[str, Any]]:
    """Return historical model training and chronological evaluation runs."""
    rows = db.fetch_all("SELECT * FROM model_evaluation_runs ORDER BY id DESC LIMIT 10")
    for r in rows:
        r["feature_importances"] = json.loads(r.pop("feature_importances_json", "{}"))
        r["comparison_series"] = json.loads(r.pop("comparison_series_json", "[]"))
    return rows


@router.post("/forecasts/predict", tags=["Forecasting & ML"])
def generate_demand_forecast(
    payload: ForecastRequest,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("forecasts:write")),
) -> Dict[str, Any]:
    """Generate a meal demand forecast using forecast-time features and persist to forecast history."""
    rows = db.fetch_all(
        "SELECT * FROM meal_service_records WHERE service_date < ? ORDER BY service_date ASC",
        (payload.forecast_date,),
    )
    hist_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    pred = predict_meal_demand(
        historical_df=hist_df,
        forecast_date=payload.forecast_date,
        meal_type=payload.meal_type,
        menu_category=payload.menu_category,
        is_holiday=payload.is_holiday,
        special_event=payload.special_event,
        weather_condition=payload.weather_condition,
        temperature_c=payload.temperature_c,
    )
    if payload.save_forecast:
        with db.transaction():
            f_id = db.insert(
                "demand_forecasts",
                {
                    "forecast_date": pred["forecast_date"],
                    "meal_type": pred["meal_type"],
                    "menu_category": pred["menu_category"],
                    "is_holiday": pred["is_holiday"],
                    "special_event": pred["special_event"],
                    "weather_condition": pred["weather_condition"],
                    "temperature_c": pred["temperature_c"],
                    "predicted_demand": pred["predicted_demand"],
                    "rounded_operational_demand": pred["rounded_operational_demand"],
                    "baseline_demand": pred["baseline_demand"],
                    "planning_buffer_portions": pred["planning_buffer_portions"],
                    "upper_bound_demand": pred["upper_bound_demand"],
                    "lower_bound_demand": pred["lower_bound_demand"],
                    "model_version": pred["model_version"],
                    "is_fallback_baseline": 1 if pred["is_fallback_baseline"] else 0,
                    "contributing_factors_json": json.dumps(pred["contributing_factors"]),
                    "created_by_user_id": current_user["id"],
                    "created_at": utc_now_iso(),
                },
            )
            pred["id"] = f_id
    return pred


@router.get("/forecasts/history", tags=["Forecasting & ML"])
def get_forecast_history(
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("forecasts:read")),
) -> List[Dict[str, Any]]:
    """Retrieve generated demand forecasts joined with actual observed demand where available."""
    rows = db.fetch_all(
        """
        SELECT
            f.*,
            m.meals_prepared AS actual_prepared,
            m.meals_served AS actual_served,
            m.unmet_demand_portions AS actual_unmet,
            m.leftover_portions AS actual_leftover
        FROM demand_forecasts f
        LEFT JOIN meal_service_records m
            ON f.forecast_date = m.service_date AND f.meal_type = m.meal_type
        ORDER BY f.forecast_date DESC, f.id DESC
        LIMIT ?
        """,
        (limit,),
    )
    for r in rows:
        r["contributing_factors"] = json.loads(r.pop("contributing_factors_json", "[]"))
    return rows


# ---------------------------------------------------------------------------
# Production Planning
# ---------------------------------------------------------------------------
@router.post("/production-plans", status_code=status.HTTP_201_CREATED, tags=["Production Planning"])
def create_production_plan(
    payload: ProductionPlanRequest,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("plans:write")),
) -> Dict[str, Any]:
    """Generate and optionally save an intelligent production plan recommendation."""
    rec = compute_production_recommendation(
        predicted_demand=payload.predicted_demand,
        planned_production=payload.planned_production,
        production_capacity=payload.production_capacity,
        uncertainty_std=payload.uncertainty_std,
        shortage_risk_tolerance=payload.shortage_risk_tolerance,
        unit_cost_inr=payload.unit_cost_inr,
        shortage_penalty_inr=payload.shortage_penalty_inr,
        custom_buffer_override=payload.custom_buffer_override,
    )
    rec["service_date"] = payload.service_date
    rec["meal_type"] = payload.meal_type.lower().strip()

    if payload.save_plan:
        with db.transaction():
            plan_id = db.insert(
                "production_plans",
                {
                    "service_date": rec["service_date"],
                    "meal_type": rec["meal_type"],
                    "forecast_id": payload.forecast_id,
                    "predicted_demand": rec["predicted_demand"],
                    "planned_production": rec["planned_production"],
                    "recommended_production": rec["recommended_production"],
                    "production_capacity": rec["production_capacity"],
                    "shortage_risk_tolerance": rec["shortage_risk_tolerance"],
                    "safety_buffer_portions": rec["safety_buffer_portions"],
                    "expected_surplus_existing_plan": rec["expected_surplus_existing_plan"],
                    "expected_surplus_recommended_plan": rec["expected_surplus_recommended_plan"],
                    "expected_shortage_risk_existing": rec["expected_shortage_risk_existing"],
                    "expected_shortage_risk_recommended": rec["expected_shortage_risk_recommended"],
                    "actual_production": None,
                    "actual_served": None,
                    "actual_leftover": None,
                    "actual_discarded": None,
                    "unit_cost_inr": payload.unit_cost_inr,
                    "shortage_penalty_inr": payload.shortage_penalty_inr,
                    "explanation": rec["explanation"],
                    "created_by_user_id": current_user["id"],
                    "created_at": utc_now_iso(),
                },
            )
            rec["id"] = plan_id
    return rec


@router.get("/production-plans", tags=["Production Planning"])
def list_production_plans(
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("plans:read")),
) -> List[Dict[str, Any]]:
    """List saved production plans and post-service actuals."""
    return db.fetch_all(
        "SELECT * FROM production_plans ORDER BY service_date DESC, id DESC LIMIT ?",
        (limit,),
    )


@router.put("/production-plans/{plan_id}/actuals", tags=["Production Planning"])
def record_production_plan_actuals(
    plan_id: int,
    payload: ProductionPlanActualsUpdate,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("plans:write")),
) -> Dict[str, Any]:
    """Record actual post-service production, consumption, leftover, and discard quantities."""
    existing = db.fetch_one("SELECT * FROM production_plans WHERE id = ?", (plan_id,))
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Production plan not found.")
    if payload.actual_served > payload.actual_production:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="actual_served cannot exceed actual_production.")

    with db.transaction():
        db.update_by_id(
            "production_plans",
            plan_id,
            {
                "actual_production": payload.actual_production,
                "actual_served": payload.actual_served,
                "actual_leftover": payload.actual_leftover,
                "actual_discarded": payload.actual_discarded,
            },
        )
        updated = db.fetch_one("SELECT * FROM production_plans WHERE id = ?", (plan_id,))
    return updated or {}


# ---------------------------------------------------------------------------
# Recipients Directory
# ---------------------------------------------------------------------------
@router.get("/recipients", tags=["Recipients & Redistribution"])
def list_recipients(
    active_only: bool = Query(False),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("recipients:read")),
) -> List[Dict[str, Any]]:
    """List recipient organizations in the redistribution network."""
    sql = "SELECT * FROM recipients WHERE is_active = 1 ORDER BY priority_tier ASC, id ASC" if active_only else "SELECT * FROM recipients ORDER BY priority_tier ASC, id ASC"
    rows = db.fetch_all(sql)
    for r in rows:
        r["accepted_categories"] = json.loads(r.pop("accepted_categories_json", "[]"))
    return rows


@router.post("/recipients", status_code=status.HTTP_201_CREATED, tags=["Recipients & Redistribution"])
def create_recipient(
    payload: RecipientCreate,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("recipients:write")),
) -> Dict[str, Any]:
    """Create a new recipient organization record (Project Admin only)."""
    now_iso = utc_now_iso()
    with db.transaction():
        rec_id = db.insert(
            "recipients",
            {
                "organization_name": payload.organization_name.strip(),
                "recipient_type": payload.recipient_type.strip(),
                "contact_person": payload.contact_person.strip(),
                "contact_email": payload.contact_email.strip(),
                "contact_phone": payload.contact_phone.strip(),
                "pickup_address": payload.pickup_address.strip(),
                "latitude": payload.latitude,
                "longitude": payload.longitude,
                "accepted_categories_json": json.dumps(payload.accepted_categories),
                "max_capacity_portions": payload.max_capacity_portions,
                "operating_hours": payload.operating_hours,
                "pickup_window_start": payload.pickup_window_start,
                "pickup_window_end": payload.pickup_window_end,
                "is_verified": payload.is_verified,
                "is_active": payload.is_active,
                "priority_tier": payload.priority_tier,
                "is_synthetic": payload.is_synthetic,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )
        created = db.fetch_one("SELECT * FROM recipients WHERE id = ?", (rec_id,))
    if created:
        created["accepted_categories"] = json.loads(created.pop("accepted_categories_json", "[]"))
    return created or {}


# ---------------------------------------------------------------------------
# Surplus Listings, Matching, Offers, and Collections
# ---------------------------------------------------------------------------
@router.get("/surplus-listings", tags=["Surplus Management"])
def list_surplus_listings(
    status_filter: Optional[str] = Query(None),
    food_category: Optional[str] = Query(None),
    min_quantity: Optional[int] = Query(None, ge=0),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("surplus:read")),
) -> List[Dict[str, Any]]:
    """List surplus food listings with optional filters."""
    clauses: List[str] = []
    params: List[Any] = []
    if status_filter:
        clauses.append("status = ?")
        params.append(status_filter.upper().strip())
    if food_category:
        clauses.append("food_category = ?")
        params.append(food_category.strip())
    if min_quantity is not None:
        clauses.append("remaining_quantity_portions >= ?")
        params.append(min_quantity)

    # Recipient organizations only see safety-approved listings
    if current_user.get("role") == "recipient_org":
        clauses.append("safety_approved = 1")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return db.fetch_all(
        f"SELECT * FROM surplus_listings {where_sql} ORDER BY id DESC",
        params,
    )


@router.post("/surplus-listings", status_code=status.HTTP_201_CREATED, tags=["Surplus Management"])
def create_surplus_listing_endpoint(
    payload: SurplusListingCreate,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("surplus:write")),
) -> Dict[str, Any]:
    """Create a surplus food listing."""
    try:
        return create_surplus_listing(
            session=db,
            service_date=payload.service_date,
            meal_type=payload.meal_type,
            food_description=payload.food_description,
            food_category=payload.food_category,
            initial_quantity_portions=payload.initial_quantity_portions,
            preparation_time=payload.preparation_time,
            available_from_time=payload.available_from_time,
            pickup_deadline=payload.pickup_deadline,
            safe_use_deadline=payload.safe_use_deadline,
            storage_condition=payload.storage_condition,
            handling_notes=payload.handling_notes,
            pickup_location=payload.pickup_location,
            safety_approved=payload.safety_approved,
            created_by_user_id=current_user["id"],
            actor_role=current_user["role"],
            is_synthetic=payload.is_synthetic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/surplus-listings/{listing_id}/approve", tags=["Surplus Management"])
def approve_surplus_listing_endpoint(
    listing_id: int,
    payload: SurplusApprovalRequest,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("surplus:approve")),
) -> Dict[str, Any]:
    """Confirm human handling/safety review and transition listing to AVAILABLE."""
    try:
        return approve_surplus_listing_for_redistribution(
            session=db,
            listing_id=listing_id,
            approver_user_id=current_user["id"],
            actor_role=current_user["role"],
            handling_verification_notes=payload.handling_verification_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/surplus-listings/{listing_id}/matches", tags=["Recipients & Redistribution"])
def get_recipient_matches_for_listing(
    listing_id: int,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("recipients:read")),
) -> Dict[str, Any]:
    """Run the explainable recipient matching and ranking engine for a surplus listing."""
    listing = db.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (listing_id,))
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Surplus listing not found.")
    recipients = db.fetch_all("SELECT * FROM recipients ORDER BY id ASC")
    active_offers = db.fetch_all(
        "SELECT * FROM redistribution_offers WHERE listing_id = ?",
        (listing_id,),
    )
    return match_and_rank_recipients(
        listing=listing,
        recipients=recipients,
        active_offers_for_listing=active_offers,
    )


@router.post("/offers", status_code=status.HTTP_201_CREATED, tags=["Recipients & Redistribution"])
def create_offer_endpoint(
    payload: OfferCreateRequest,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("offers:create")),
) -> Dict[str, Any]:
    """Create a redistribution offer and atomically reserve surplus portions."""
    try:
        return create_redistribution_offer(
            session=db,
            listing_id=payload.listing_id,
            recipient_id=payload.recipient_id,
            offered_quantity_portions=payload.offered_quantity_portions,
            match_score=payload.match_score,
            match_reasons=payload.match_reasons,
            notes=payload.notes,
            actor_user_id=current_user["id"],
            actor_role=current_user["role"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/offers", tags=["Recipients & Redistribution"])
def list_offers(
    listing_id: Optional[int] = Query(None),
    recipient_id: Optional[int] = Query(None),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("offers:read")),
) -> List[Dict[str, Any]]:
    """List redistribution offers with listing and recipient metadata."""
    clauses: List[str] = []
    params: List[Any] = []
    if listing_id is not None:
        clauses.append("o.listing_id = ?")
        params.append(listing_id)
    if recipient_id is not None:
        clauses.append("o.recipient_id = ?")
        params.append(recipient_id)
    elif current_user.get("role") == "recipient_org" and current_user.get("recipient_id"):
        clauses.append("o.recipient_id = ?")
        params.append(int(current_user["recipient_id"]))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = db.fetch_all(
        f"""
        SELECT
            o.*,
            r.organization_name,
            r.contact_person,
            r.contact_phone,
            s.listing_code,
            s.food_description,
            s.food_category,
            s.pickup_deadline,
            s.safe_use_deadline,
            s.pickup_location
        FROM redistribution_offers o
        JOIN recipients r ON o.recipient_id = r.id
        JOIN surplus_listings s ON o.listing_id = s.id
        {where_sql}
        ORDER BY o.id DESC
        """,
        params,
    )
    for r in rows:
        r["match_reasons"] = json.loads(r.pop("match_reasons_json", "[]"))
    return rows


@router.post("/offers/{offer_id}/respond", tags=["Recipients & Redistribution"])
def respond_to_offer_endpoint(
    offer_id: int,
    payload: OfferRespondRequest,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("offers:respond")),
) -> Dict[str, Any]:
    """Allow a recipient organization (or project admin) to accept, decline, or cancel an offer."""
    offer = db.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
    if not offer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Offer not found.")

    # Enforce recipient ownership when user is a recipient_org
    if current_user.get("role") == "recipient_org":
        user_rec_id = current_user.get("recipient_id")
        if user_rec_id is None or int(user_rec_id) != int(offer["recipient_id"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Recipient account can only respond to offers addressed to its own organization.",
            )

    try:
        return respond_to_offer(
            session=db,
            offer_id=offer_id,
            action=payload.action,
            accepted_quantity_portions=payload.accepted_quantity_portions,
            actor_user_id=current_user["id"],
            actor_role=current_user["role"],
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/offers/{offer_id}/collect", tags=["Recipients & Redistribution"])
def confirm_collection_endpoint(
    offer_id: int,
    payload: CollectionConfirmRequest,
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("collections:confirm")),
) -> Dict[str, Any]:
    """Confirm physical collection and handover of an accepted offer."""
    offer = db.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
    if not offer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Offer not found.")

    if current_user.get("role") == "recipient_org":
        user_rec_id = current_user.get("recipient_id")
        if user_rec_id is None or int(user_rec_id) != int(offer["recipient_id"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Recipient account can only confirm collection for its own organization.",
            )

    try:
        return confirm_collection(
            session=db,
            offer_id=offer_id,
            collected_quantity_portions=payload.collected_quantity_portions,
            actor_user_id=current_user["id"],
            actor_role=current_user["role"],
            verification_notes=payload.verification_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ---------------------------------------------------------------------------
# Analytics, Impact Reports, and Audit Trail
# ---------------------------------------------------------------------------
@router.get("/analytics/impact", tags=["Analytics & Impact"])
def get_impact_report(
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("analytics:read")),
) -> Dict[str, Any]:
    """Return auditable impact metrics separated into predictions, simulations, recorded outcomes, and derived estimates."""
    return compute_system_impact_report(db)


@router.get("/analytics/transactions", tags=["Analytics & Impact"])
def list_redistribution_transactions(
    limit: int = Query(100, ge=1, le=500),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("analytics:read")),
) -> List[Dict[str, Any]]:
    """Return immutable redistribution transaction history."""
    return db.fetch_all(
        """
        SELECT
            t.*,
            r.organization_name,
            s.listing_code,
            s.food_description
        FROM redistribution_transactions t
        JOIN recipients r ON t.recipient_id = r.id
        JOIN surplus_listings s ON t.listing_id = s.id
        ORDER BY t.id DESC
        LIMIT ?
        """,
        (limit,),
    )


@router.get("/audit-events", tags=["System"])
def list_audit_events(
    limit: int = Query(100, ge=1, le=500),
    db: DatabaseSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_permission("audit:read")),
) -> List[Dict[str, Any]]:
    """Return system audit trail (Project Admin only)."""
    rows = db.fetch_all("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,))
    for r in rows:
        r["details"] = json.loads(r.pop("details_json", "{}"))
    return rows
