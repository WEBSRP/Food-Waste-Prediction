"""End-to-end 12-step scripted demonstration scenario runner (Section 15).

Executes every step of the demonstration scenario through the FastAPI REST endpoints
and exports a complete JSON audit report to data/exports/demo_scenario_execution_report.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.core.config import EXPORTS_DIR, SYNTHETIC_DIR
from backend.main import app
from scripts.seed_database import seed_all


def run_complete_demo_scenario() -> Dict[str, Any]:
    """Execute the 12-step demonstration scenario end to end against the REST API."""
    # Step 1: Seed 12 months of synthetic meal-service history & initial entities
    seed_summary = seed_all(reset_existing=True)
    client = TestClient(app)

    # Authenticate as Cafeteria Admin
    admin_login = client.post(
        "/api/v1/auth/login",
        json={"username": "cafeteria_admin", "password": "OriginCafeteria#2026"},
    )
    assert admin_login.status_code == 200, admin_login.text
    admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}

    # Authenticate as Recipient Organization (Yamuna Expressway Shelter Home)
    rec_login = client.post(
        "/api/v1/auth/login",
        json={"username": "recipient_yamuna", "password": "OriginRecipient#2026"},
    )
    assert rec_login.status_code == 200, rec_login.text
    rec_user = rec_login.json()["user"]
    rec_headers = {"Authorization": f"Bearer {rec_login.json()['access_token']}"}

    # Step 2: Validate sample CSV dataset via REST API
    sample_csv = (SYNTHETIC_DIR / "sample_cafeteria_upload_template.csv").read_bytes()
    val_resp = client.post(
        "/api/v1/meals/validate-csv",
        headers=admin_headers,
        files={"file": ("sample_cafeteria_upload_template.csv", sample_csv, "text/csv")},
    )
    assert val_resp.status_code == 200, val_resp.text

    # Step 3: Train and evaluate Baseline and ML models via REST API
    train_resp = client.post("/api/v1/models/train", headers=admin_headers)
    assert train_resp.status_code == 200, train_resp.text
    train_data = train_resp.json()

    # Step 4: Generate tomorrow's lunch forecast (2026-10-16)
    forecast_resp = client.post(
        "/api/v1/forecasts/predict",
        headers=admin_headers,
        json={
            "forecast_date": "2026-10-16",
            "meal_type": "lunch",
            "menu_category": "north_indian_standard",
            "is_holiday": 0,
            "special_event": "none",
            "weather_condition": "clear",
            "temperature_c": 28.0,
            "save_forecast": True,
        },
    )
    assert forecast_resp.status_code == 200, forecast_resp.text
    forecast_data = forecast_resp.json()

    # Step 5 & 6: Compare existing production plan (640 portions) with recommended quantity and show estimated surplus
    plan_resp = client.post(
        "/api/v1/production-plans",
        headers=admin_headers,
        json={
            "service_date": "2026-10-16",
            "meal_type": "lunch",
            "forecast_id": forecast_data.get("id"),
            "predicted_demand": forecast_data["predicted_demand"],
            "planned_production": 640,
            "production_capacity": 750,
            "uncertainty_std": train_data["residual_std"],
            "shortage_risk_tolerance": 0.15,
            "unit_cost_inr": 65.0,
            "shortage_penalty_inr": 110.0,
            "save_plan": True,
        },
    )
    assert plan_resp.status_code == 201, plan_resp.text
    plan_data = plan_resp.json()

    # Step 7: Create a confirmed surplus listing (in DRAFT first, then explicitly approve for redistribution)
    create_listing_resp = client.post(
        "/api/v1/surplus-listings",
        headers=admin_headers,
        json={
            "service_date": "2026-10-15",
            "meal_type": "lunch",
            "food_description": "Chole, Jeera Rice & Steamed Roti Batch (Demo Scenario)",
            "food_category": "cooked_grains_legumes",
            "initial_quantity_portions": 80,
            "preparation_time": "2026-10-15T06:30:00Z",
            "available_from_time": "2026-10-15T09:00:00Z",
            "pickup_deadline": "2026-12-31T18:00:00Z",
            "safe_use_deadline": "2026-12-31T20:30:00Z",
            "storage_condition": "Hot-held at 64°C in insulated stainless steel trays",
            "handling_notes": "Prepared in Central Mess Block A.",
            "safety_approved": False,
            "is_synthetic": 1,
        },
    )
    assert create_listing_resp.status_code == 201, create_listing_resp.text
    listing_id = int(create_listing_resp.json()["id"])

    approve_resp = client.post(
        f"/api/v1/surplus-listings/{listing_id}/approve",
        headers=admin_headers,
        json={"handling_verification_notes": "Core temperature verified at 64.5°C; hygienic seals intact."},
    )
    assert approve_resp.status_code == 200, approve_resp.text

    # Step 8: Find and rank eligible recipients for this confirmed surplus listing
    matches_resp = client.get(
        f"/api/v1/surplus-listings/{listing_id}/matches",
        headers=admin_headers,
    )
    assert matches_resp.status_code == 200, matches_resp.text
    matches_data = matches_resp.json()

    # Locate Yamuna Expressway Shelter Home in the eligible ranked matches
    yamuna_match = next(
        m for m in matches_data["eligible_matches"] if int(m["recipient_id"]) == int(rec_user["recipient_id"])
    )

    # Step 9: Create a redistribution offer for 80 portions
    offer_resp = client.post(
        "/api/v1/offers",
        headers=admin_headers,
        json={
            "listing_id": listing_id,
            "recipient_id": int(rec_user["recipient_id"]),
            "offered_quantity_portions": 80,
            "match_score": yamuna_match["match_score"],
            "match_reasons": yamuna_match["match_reasons"],
            "notes": "Ready for pickup at Mess Block A loading bay.",
        },
    )
    assert offer_resp.status_code == 201, offer_resp.text
    offer_id = int(offer_resp.json()["id"])

    # Step 10: Accept the offer using the Recipient Organization demo account
    accept_resp = client.post(
        f"/api/v1/offers/{offer_id}/respond",
        headers=rec_headers,
        json={
            "action": "ACCEPT",
            "accepted_quantity_portions": 80,
            "notes": "Accepted by Yamuna Shelter intake coordinator; vehicle arriving in 20 mins.",
        },
    )
    assert accept_resp.status_code == 200, accept_resp.text

    # Step 11: Confirm physical collection
    collect_resp = client.post(
        f"/api/v1/offers/{offer_id}/collect",
        headers=admin_headers,
        json={
            "collected_quantity_portions": 80,
            "verification_notes": "80 portions verified and loaded into insulated transport boxes.",
        },
    )
    assert collect_resp.status_code == 200, collect_resp.text

    # Step 12: Fetch updated inventory and auditable impact analytics
    impact_resp = client.get("/api/v1/analytics/impact", headers=admin_headers)
    assert impact_resp.status_code == 200, impact_resp.text
    impact_data = impact_resp.json()

    scenario_report = {
        "scenario_status": "COMPLETED_VERIFIED",
        "step_1_seed_summary": seed_summary,
        "step_2_csv_validation": val_resp.json(),
        "step_3_model_evaluation": {
            "model_version": train_data["model_version"],
            "model_name": train_data["model_name"],
            "baseline_name": train_data["baseline_name"],
            "train_samples": train_data["train_samples"],
            "eval_samples": train_data["eval_samples"],
            "ml_mae": train_data["ml_mae"],
            "ml_rmse": train_data["ml_rmse"],
            "ml_wape": train_data["ml_wape"],
            "ml_bias": train_data["ml_bias"],
            "baseline_mae": train_data["baseline_mae"],
            "baseline_rmse": train_data["baseline_rmse"],
            "baseline_wape": train_data["baseline_wape"],
            "baseline_bias": train_data["baseline_bias"],
            "selected_model": train_data["selected_model"],
            "top_features": dict(list(train_data["feature_importances"].items())[:6]),
        },
        "step_4_tomorrows_forecast": forecast_data,
        "step_5_6_production_plan_and_surplus": plan_data,
        "step_7_confirmed_surplus_listing": approve_resp.json(),
        "step_8_ranked_recipient_matches": {
            "eligible_count": len(matches_data["eligible_matches"]),
            "excluded_count": len(matches_data["excluded_recipients"]),
            "top_3_matches": matches_data["eligible_matches"][:3],
            "excluded_recipients": matches_data["excluded_recipients"],
        },
        "step_9_created_offer": offer_resp.json(),
        "step_10_recipient_acceptance": accept_resp.json(),
        "step_11_collection_confirmation": collect_resp.json(),
        "step_12_impact_analytics": impact_data,
    }

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPORTS_DIR / "demo_scenario_execution_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(scenario_report, f, indent=2)

    return scenario_report


if __name__ == "__main__":
    rep = run_complete_demo_scenario()
    print("12-Step Demo Scenario completed successfully!")
    print("Model Evaluation Summary:", json.dumps(rep["step_3_model_evaluation"], indent=2))
    print("Impact Analytics Summary:", json.dumps(rep["step_12_impact_analytics"], indent=2))
