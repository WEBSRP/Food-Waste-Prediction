"""Database seeding and initial model training script for ORIGIN MINI 2026 demonstration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.core.config import settings
from backend.core.security import (
    ROLE_CAFETERIA_ADMIN,
    ROLE_PERMISSIONS,
    ROLE_PROJECT_ADMIN,
    ROLE_RECIPIENT_ORG,
    hash_password,
)
from backend.database.models import utc_now_iso
from backend.database.session import DatabaseSession, init_db
from backend.domain.production_planner import compute_production_recommendation
from backend.domain.surplus_detector import (
    confirm_collection,
    create_redistribution_offer,
    create_surplus_listing,
    respond_to_offer,
)
from backend.ml.predict_demand import predict_meal_demand
from backend.ml.train_model import train_and_evaluate_pipeline
from scripts.generate_demo_data import (
    generate_synthetic_meal_records,
    generate_synthetic_recipients,
    save_synthetic_artifacts,
)


def seed_all(db_path: str | None = None, reset_existing: bool = True) -> Dict[str, Any]:
    """Initialize schema, populate synthetic records, train models, and create sample workflows."""
    save_synthetic_artifacts()
    init_db(db_path=db_path)
    session = DatabaseSession(db_path=db_path)
    now_iso = utc_now_iso()

    try:
        with session.transaction():
            if reset_existing:
                for tbl in [
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
                ]:
                    session.execute(f"DELETE FROM {tbl};")

            # 1. Seed Roles
            for role_name, perms in ROLE_PERMISSIONS.items():
                session.insert(
                    "roles",
                    {
                        "name": role_name,
                        "description": f"Demo role for {role_name}",
                        "permissions_json": json.dumps(perms),
                        "created_at": now_iso,
                    },
                )

            # 2. Seed Recipients
            recipients_data = generate_synthetic_recipients()
            recipient_ids: List[int] = []
            for rec in recipients_data:
                r_id = session.insert(
                    "recipients",
                    {
                        "organization_name": rec["organization_name"],
                        "recipient_type": rec["recipient_type"],
                        "contact_person": rec["contact_person"],
                        "contact_email": rec["contact_email"],
                        "contact_phone": rec["contact_phone"],
                        "pickup_address": rec["pickup_address"],
                        "latitude": rec["latitude"],
                        "longitude": rec["longitude"],
                        "accepted_categories_json": json.dumps(rec["accepted_categories"]),
                        "max_capacity_portions": rec["max_capacity_portions"],
                        "operating_hours": rec["operating_hours"],
                        "pickup_window_start": rec["pickup_window_start"],
                        "pickup_window_end": rec["pickup_window_end"],
                        "is_verified": rec["is_verified"],
                        "is_active": rec["is_active"],
                        "priority_tier": rec["priority_tier"],
                        "is_synthetic": 1,
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    },
                )
                recipient_ids.append(r_id)

            # 3. Seed Demo Users (Never store plaintext passwords; PBKDF2-HMAC-SHA256 hashes only)
            cafeteria_admin_id = session.insert(
                "users",
                {
                    "username": "cafeteria_admin",
                    "full_name": "Ananya Deshmukh (Hostel Mess Administrator)",
                    "email": "mess.admin@galgotias-demo.edu.in",
                    "password_hash": hash_password("OriginCafeteria#2026"),
                    "role": ROLE_CAFETERIA_ADMIN,
                    "recipient_id": None,
                    "is_active": 1,
                    "is_demo_account": 1,
                    "created_at": now_iso,
                },
            )
            recipient_user_1_id = session.insert(
                "users",
                {
                    "username": "recipient_gnckt",
                    "full_name": "Aarav Sharma (Greater Noida Community Kitchen)",
                    "email": "dispatch.gnckt@example.org",
                    "password_hash": hash_password("OriginRecipient#2026"),
                    "role": ROLE_RECIPIENT_ORG,
                    "recipient_id": recipient_ids[0],
                    "is_active": 1,
                    "is_demo_account": 1,
                    "created_at": now_iso,
                },
            )
            recipient_user_2_id = session.insert(
                "users",
                {
                    "username": "recipient_yamuna",
                    "full_name": "Meera Verma (Yamuna Shelter Home)",
                    "email": "intake.yamunashelter@example.org",
                    "password_hash": hash_password("OriginRecipient#2026"),
                    "role": ROLE_RECIPIENT_ORG,
                    "recipient_id": recipient_ids[1],
                    "is_active": 1,
                    "is_demo_account": 1,
                    "created_at": now_iso,
                },
            )
            project_admin_id = session.insert(
                "users",
                {
                    "username": "project_admin",
                    "full_name": "Prof. Siddharth Khanna (ORIGIN MINI 2026 Project Lead)",
                    "email": "project.lead@galgotias-demo.edu.in",
                    "password_hash": hash_password("OriginAdmin#2026"),
                    "role": ROLE_PROJECT_ADMIN,
                    "recipient_id": None,
                    "is_active": 1,
                    "is_demo_account": 1,
                    "created_at": now_iso,
                },
            )

            # 4. Seed Application Settings
            default_settings = [
                ("breakfast_capacity", str(settings.DEFAULT_CAPACITIES["breakfast"]), "Maximum portions for breakfast"),
                ("lunch_capacity", str(settings.DEFAULT_CAPACITIES["lunch"]), "Maximum portions for lunch"),
                ("dinner_capacity", str(settings.DEFAULT_CAPACITIES["dinner"]), "Maximum portions for dinner"),
                ("shortage_risk_tolerance", str(settings.DEFAULT_SHORTAGE_RISK_TOLERANCE), "Target maximum stockout probability"),
                ("meal_unit_cost_inr", str(settings.DEFAULT_MEAL_UNIT_COST_INR), "Average production cost per meal portion in INR"),
                ("require_manual_safety_approval", "true", "Require explicit staff verification before surplus listing becomes AVAILABLE"),
            ]
            for s_key, s_val, s_desc in default_settings:
                session.insert(
                    "application_settings",
                    {
                        "setting_key": s_key,
                        "setting_value": s_val,
                        "description": s_desc,
                        "updated_by_user_id": project_admin_id,
                        "updated_at": now_iso,
                    },
                )

            # 5. Seed 12 Months of Synthetic Meal Service Records
            meals_df = generate_synthetic_meal_records()
            for _, row in meals_df.iterrows():
                session.insert(
                    "meal_service_records",
                    {
                        "service_date": str(row["service_date"]),
                        "meal_type": str(row["meal_type"]),
                        "menu_category": str(row["menu_category"]),
                        "menu_items": str(row["menu_items"]),
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
                        "is_synthetic": 1,
                        "recorded_by_user_id": cafeteria_admin_id,
                        "created_at": now_iso,
                    },
                )

        # 6. Train and Evaluate Forecasting Models on Seeded History
        eval_res = train_and_evaluate_pipeline(meals_df)
        with session.transaction():
            session.insert(
                "model_evaluation_runs",
                {
                    "model_version": eval_res["model_version"],
                    "model_name": eval_res["model_name"],
                    "baseline_name": eval_res["baseline_name"],
                    "train_start_date": eval_res["train_start_date"],
                    "train_end_date": eval_res["train_end_date"],
                    "eval_start_date": eval_res["eval_start_date"],
                    "eval_end_date": eval_res["eval_end_date"],
                    "train_samples": eval_res["train_samples"],
                    "eval_samples": eval_res["eval_samples"],
                    "ml_mae": eval_res["ml_mae"],
                    "ml_rmse": eval_res["ml_rmse"],
                    "ml_wape": eval_res["ml_wape"],
                    "ml_bias": eval_res["ml_bias"],
                    "baseline_mae": eval_res["baseline_mae"],
                    "baseline_rmse": eval_res["baseline_rmse"],
                    "baseline_wape": eval_res["baseline_wape"],
                    "baseline_bias": eval_res["baseline_bias"],
                    "selected_model": eval_res["selected_model"],
                    "residual_std": eval_res["residual_std"],
                    "residual_q85": eval_res["residual_q85"],
                    "feature_importances_json": json.dumps(eval_res["feature_importances"]),
                    "comparison_series_json": json.dumps(eval_res["comparison_series"]),
                    "artifact_path": eval_res["artifact_path"],
                    "created_at": now_iso,
                },
            )

            # 7. Seed Upcoming Forecasts & Production Plans for 2026-10-15 and 2026-10-16
            upcoming_specs = [
                ("2026-10-15", "breakfast", "south_indian_special", 360, 450),
                ("2026-10-15", "lunch", "north_indian_standard", 625, 750),
                ("2026-10-15", "dinner", "continental_fusion", 575, 650),
                ("2026-10-16", "breakfast", "north_indian_standard", 345, 450),
                ("2026-10-16", "lunch", "festive_thali", 650, 750),
                ("2026-10-16", "dinner", "healthy_millet_bowl", 490, 650),
            ]
            for f_date, m_type, m_cat, planned_qty, cap_qty in upcoming_specs:
                pred = predict_meal_demand(
                    historical_df=meals_df,
                    forecast_date=f_date,
                    meal_type=m_type,
                    menu_category=m_cat,
                )
                f_id = session.insert(
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
                        "is_fallback_baseline": 0,
                        "contributing_factors_json": json.dumps(pred["contributing_factors"]),
                        "created_by_user_id": cafeteria_admin_id,
                        "created_at": now_iso,
                    },
                )
                plan_rec = compute_production_recommendation(
                    predicted_demand=pred["predicted_demand"],
                    planned_production=planned_qty,
                    production_capacity=cap_qty,
                    uncertainty_std=eval_res["residual_std"],
                    shortage_risk_tolerance=0.15,
                )
                session.insert(
                    "production_plans",
                    {
                        "service_date": f_date,
                        "meal_type": m_type,
                        "forecast_id": f_id,
                        "predicted_demand": plan_rec["predicted_demand"],
                        "planned_production": plan_rec["planned_production"],
                        "recommended_production": plan_rec["recommended_production"],
                        "production_capacity": plan_rec["production_capacity"],
                        "shortage_risk_tolerance": plan_rec["shortage_risk_tolerance"],
                        "safety_buffer_portions": plan_rec["safety_buffer_portions"],
                        "expected_surplus_existing_plan": plan_rec["expected_surplus_existing_plan"],
                        "expected_surplus_recommended_plan": plan_rec["expected_surplus_recommended_plan"],
                        "expected_shortage_risk_existing": plan_rec["expected_shortage_risk_existing"],
                        "expected_shortage_risk_recommended": plan_rec["expected_shortage_risk_recommended"],
                        "actual_production": None,
                        "actual_served": None,
                        "actual_leftover": None,
                        "actual_discarded": None,
                        "unit_cost_inr": 65.0,
                        "shortage_penalty_inr": 110.0,
                        "explanation": plan_rec["explanation"],
                        "created_by_user_id": cafeteria_admin_id,
                        "created_at": now_iso,
                    },
                )

        # 8. Seed Sample Surplus Listings across Statuses
        # Listing 1: Completed collection from yesterday's lunch
        listing_1 = create_surplus_listing(
            session=session,
            service_date="2026-10-14",
            meal_type="lunch",
            food_description="Rajma Chawal & Steamed Basmati Trays (Synthetic Demo)",
            food_category="cooked_grains_legumes",
            initial_quantity_portions=110,
            preparation_time="2026-10-14T06:30:00Z",
            available_from_time="2026-10-14T09:00:00Z",
            pickup_deadline="2026-12-31T16:30:00Z",
            safe_use_deadline="2026-12-31T19:00:00Z",
            storage_condition="Hot-held at 65°C in insulated stainless containers",
            handling_notes="Verified safe by Mess Shift Supervisor.",
            safety_approved=True,
            created_by_user_id=cafeteria_admin_id,
            actor_role=ROLE_CAFETERIA_ADMIN,
            is_synthetic=1,
        )
        offer_1 = create_redistribution_offer(
            session=session,
            listing_id=int(listing_1["id"]),
            recipient_id=recipient_ids[0],
            offered_quantity_portions=110,
            match_score=91.4,
            match_reasons=["Tier 1 verified community kitchen", "Accepts cooked_grains_legumes", "Full 110-portion capacity coverage"],
            notes="Initial demo allocation to Greater Noida Community Kitchen.",
            actor_user_id=cafeteria_admin_id,
            actor_role=ROLE_CAFETERIA_ADMIN,
        )
        respond_to_offer(
            session=session,
            offer_id=int(offer_1["id"]),
            action="ACCEPT",
            accepted_quantity_portions=110,
            actor_user_id=recipient_user_1_id,
            actor_role=ROLE_RECIPIENT_ORG,
            notes="Van dispatched from Knowledge Park III.",
        )
        confirm_collection(
            session=session,
            offer_id=int(offer_1["id"]),
            collected_quantity_portions=110,
            actor_user_id=cafeteria_admin_id,
            actor_role=ROLE_CAFETERIA_ADMIN,
            verification_notes="110 portions handed over and signed at Gate A.",
        )

        # Listing 2: Active AVAILABLE listing ready for matching & offers
        listing_2 = create_surplus_listing(
            session=session,
            service_date="2026-10-15",
            meal_type="lunch",
            food_description="Paneer Makhani, Dal Tadka & Jeera Rice (Synthetic Demo)",
            food_category="cooked_grains_legumes",
            initial_quantity_portions=95,
            preparation_time="2026-10-15T06:45:00Z",
            available_from_time="2026-10-15T09:15:00Z",
            pickup_deadline="2026-12-31T17:30:00Z",
            safe_use_deadline="2026-12-31T20:00:00Z",
            storage_condition="Hot-held at 64°C in sealed gastronorm pans",
            handling_notes="Inspected and sealed for redistribution.",
            safety_approved=True,
            created_by_user_id=cafeteria_admin_id,
            actor_role=ROLE_CAFETERIA_ADMIN,
            is_synthetic=1,
        )

        # Listing 3: DRAFT listing awaiting explicit human safety approval
        listing_3 = create_surplus_listing(
            session=session,
            service_date="2026-10-15",
            meal_type="dinner",
            food_description="Whole Wheat Roti & Mixed Vegetable Curry (Synthetic Demo)",
            food_category="breads_bakery",
            initial_quantity_portions=60,
            preparation_time="2026-10-15T14:00:00Z",
            available_from_time="2026-10-15T16:00:00Z",
            pickup_deadline="2026-12-31T21:30:00Z",
            safe_use_deadline="2026-12-31T23:00:00Z",
            storage_condition="Food-grade insulated bread cassettes",
            handling_notes="Awaiting supervisor temperature log sign-off.",
            safety_approved=False,
            created_by_user_id=cafeteria_admin_id,
            actor_role=ROLE_CAFETERIA_ADMIN,
            is_synthetic=1,
        )

        return {
            "seeded_meal_records": len(meals_df),
            "seeded_recipients": len(recipient_ids),
            "model_evaluation": {
                "model_version": eval_res["model_version"],
                "ml_mae": eval_res["ml_mae"],
                "ml_rmse": eval_res["ml_rmse"],
                "ml_wape": eval_res["ml_wape"],
                "ml_bias": eval_res["ml_bias"],
                "baseline_mae": eval_res["baseline_mae"],
                "baseline_rmse": eval_res["baseline_rmse"],
                "baseline_wape": eval_res["baseline_wape"],
                "baseline_bias": eval_res["baseline_bias"],
                "selected_model": eval_res["selected_model"],
            },
            "sample_listings": [listing_1["listing_code"], listing_2["listing_code"], listing_3["listing_code"]],
        }
    finally:
        session.close()


if __name__ == "__main__":
    result = seed_all()
    print("Database seeded and models trained successfully:")
    print(json.dumps(result, indent=2))
