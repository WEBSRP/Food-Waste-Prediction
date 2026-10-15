"""Auditable impact measurement engine with explicit denominators and category separation.

Strictly separates:
1. Model Predictions (forecast accuracy metrics on held-out chronological splits)
2. Simulated / Planned Outcomes (expected surplus under existing vs recommended production plans)
3. Actual User-Recorded Outcomes (meals prepared, served, leftover, discarded; surplus listed, offered, accepted, collected, uncollected/expired)
4. Derived Financial Estimates (qualified procurement value of collected surplus and avoided overproduction using supplied unit cost)

Does NOT fabricate carbon-emission reductions or present synthetic demo numbers as real-world impact.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.core.config import settings
from backend.database.session import DatabaseSession


def compute_system_impact_report(session: DatabaseSession) -> Dict[str, Any]:
    """Compute comprehensive, auditable impact metrics from the relational database."""
    # 1. Latest Model Evaluation Metrics
    latest_eval = session.fetch_one(
        "SELECT * FROM model_evaluation_runs ORDER BY id DESC LIMIT 1"
    )
    model_predictions_summary = {
        "has_trained_model": latest_eval is not None,
        "model_version": latest_eval["model_version"] if latest_eval else "untrained",
        "selected_model": latest_eval["selected_model"] if latest_eval else "baseline",
        "held_out_eval_samples": int(latest_eval["eval_samples"]) if latest_eval else 0,
        "ml_mae_portions": float(latest_eval["ml_mae"]) if latest_eval else None,
        "ml_rmse_portions": float(latest_eval["ml_rmse"]) if latest_eval else None,
        "ml_wape_percent": float(latest_eval["ml_wape"]) if latest_eval else None,
        "ml_bias_portions": float(latest_eval["ml_bias"]) if latest_eval else None,
        "baseline_mae_portions": float(latest_eval["baseline_mae"]) if latest_eval else None,
        "baseline_wape_percent": float(latest_eval["baseline_wape"]) if latest_eval else None,
        "mae_improvement_vs_baseline_percent": (
            round(
                ((float(latest_eval["baseline_mae"]) - float(latest_eval["ml_mae"])) / max(1e-6, float(latest_eval["baseline_mae"]))) * 100.0,
                2,
            )
            if latest_eval and float(latest_eval["baseline_mae"]) > 0
            else 0.0
        ),
    }

    # 2. Historical Meal Service Outcomes (Actual / Synthetic-Labeled Records)
    meal_agg = session.fetch_one(
        """
        SELECT
            COUNT(*) AS total_services,
            COALESCE(SUM(meals_prepared), 0) AS total_prepared,
            COALESCE(SUM(meals_served), 0) AS total_served,
            COALESCE(SUM(leftover_portions), 0) AS total_leftover,
            COALESCE(SUM(food_discarded_portions), 0) AS total_discarded,
            COALESCE(SUM(unmet_demand_portions), 0) AS total_unmet,
            COALESCE(SUM(is_synthetic), 0) AS synthetic_records_count
        FROM meal_service_records
        """
    ) or {}

    total_services = int(meal_agg.get("total_services", 0))
    total_prepared = int(meal_agg.get("total_prepared", 0))
    total_served = int(meal_agg.get("total_served", 0))
    total_leftover = int(meal_agg.get("total_leftover", 0))
    total_discarded = int(meal_agg.get("total_discarded", 0))
    total_unmet = int(meal_agg.get("total_unmet", 0))
    synthetic_count = int(meal_agg.get("synthetic_records_count", 0))

    # Production-plan deviation from observed demand: |meals_prepared - (meals_served + unmet_demand)|
    dev_row = session.fetch_one(
        """
        SELECT COALESCE(AVG(ABS(meals_prepared - (meals_served + unmet_demand_portions))), 0.0) AS mean_abs_plan_deviation
        FROM meal_service_records
        """
    ) or {}
    mean_abs_plan_deviation = round(float(dev_row.get("mean_abs_plan_deviation", 0.0)), 2)

    # 3. Simulated / Planned Production Recommendations
    plan_agg = session.fetch_one(
        """
        SELECT
            COUNT(*) AS total_plans,
            COALESCE(SUM(planned_production), 0) AS sum_planned,
            COALESCE(SUM(recommended_production), 0) AS sum_recommended,
            COALESCE(SUM(expected_surplus_existing_plan), 0) AS sum_exp_surplus_existing,
            COALESCE(SUM(expected_surplus_recommended_plan), 0) AS sum_exp_surplus_recommended,
            COALESCE(AVG(unit_cost_inr), 65.0) AS avg_unit_cost
        FROM production_plans
        """
    ) or {}
    sum_exp_existing = int(plan_agg.get("sum_exp_surplus_existing", 0))
    sum_exp_recommended = int(plan_agg.get("sum_exp_surplus_recommended", 0))
    simulated_overproduction_avoided = max(0, sum_exp_existing - sum_exp_recommended)
    avg_unit_cost = float(plan_agg.get("avg_unit_cost", settings.DEFAULT_MEAL_UNIT_COST_INR))

    # 4. Surplus Listings & Redistribution Outcomes (Avoiding double counting across offers)
    listing_agg = session.fetch_one(
        """
        SELECT
            COUNT(*) AS total_listings,
            COALESCE(SUM(CASE WHEN safety_approved = 1 THEN initial_quantity_portions ELSE 0 END), 0) AS confirmed_eligible_surplus_portions,
            COALESCE(SUM(initial_quantity_portions), 0) AS total_listed_portions,
            COALESCE(SUM(remaining_quantity_portions), 0) AS current_unallocated_portions,
            COALESCE(SUM(reserved_quantity_portions), 0) AS current_reserved_portions,
            COALESCE(SUM(collected_quantity_portions), 0) AS total_collected_from_listings,
            COALESCE(SUM(CASE WHEN status IN ('EXPIRED', 'CANCELLED') THEN remaining_quantity_portions ELSE 0 END), 0) AS expired_or_cancelled_portions
        FROM surplus_listings
        """
    ) or {}

    offer_agg = session.fetch_one(
        """
        SELECT
            COUNT(*) AS total_offers,
            COALESCE(SUM(offered_quantity_portions), 0) AS gross_offered_portions,
            COALESCE(SUM(CASE WHEN status IN ('ACCEPTED', 'COLLECTED') THEN accepted_quantity_portions ELSE 0 END), 0) AS net_accepted_portions,
            COALESCE(SUM(CASE WHEN status = 'COLLECTED' THEN collected_quantity_portions ELSE 0 END), 0) AS verified_collected_portions,
            COALESCE(SUM(uncollected_quantity_portions), 0) AS offer_uncollected_portions
        FROM redistribution_offers
        """
    ) or {}

    confirmed_eligible = int(listing_agg.get("confirmed_eligible_surplus_portions", 0))
    verified_collected = int(offer_agg.get("verified_collected_portions", 0))
    net_accepted = int(offer_agg.get("net_accepted_portions", 0))
    gross_offered = int(offer_agg.get("gross_offered_portions", 0))
    expired_or_uncollected = int(listing_agg.get("expired_or_cancelled_portions", 0)) + int(
        offer_agg.get("offer_uncollected_portions", 0)
    )

    # Explicit denominator 1: Verified Collected Portions / Confirmed Eligible Surplus Portions
    redistribution_recovery_rate = (
        round((verified_collected / confirmed_eligible) * 100.0, 2)
        if confirmed_eligible > 0
        else 0.0
    )
    # Explicit denominator 2: Verified Collected Portions / Net Accepted Portions
    collection_fulfillment_rate = (
        round((verified_collected / net_accepted) * 100.0, 2)
        if net_accepted > 0
        else 0.0
    )

    return {
        "data_provenance_notice": (
            f"Dataset contains {synthetic_count}/{total_services} synthetic demonstration meal records. "
            "Synthetic metrics demonstrate algorithmic behavior and must not be cited as real-world physical impact."
            if synthetic_count > 0
            else "All records are marked as user-supplied operational records."
        ),
        "model_predictions": model_predictions_summary,
        "simulated_planning_outcomes": {
            "evaluated_production_plans_count": int(plan_agg.get("total_plans", 0)),
            "total_planned_portions": int(plan_agg.get("sum_planned", 0)),
            "total_recommended_portions": int(plan_agg.get("sum_recommended", 0)),
            "expected_surplus_under_existing_plans_portions": sum_exp_existing,
            "expected_surplus_under_recommended_plans_portions": sum_exp_recommended,
            "simulated_overproduction_avoided_portions": simulated_overproduction_avoided,
        },
        "actual_recorded_outcomes": {
            "historical_meal_services_logged": total_services,
            "total_meals_prepared_portions": total_prepared,
            "total_meals_served_portions": total_served,
            "total_leftover_portions": total_leftover,
            "total_food_discarded_portions": total_discarded,
            "total_unmet_demand_portions": total_unmet,
            "mean_abs_production_plan_deviation_portions": mean_abs_plan_deviation,
            "surplus_listings_count": int(listing_agg.get("total_listings", 0)),
            "confirmed_eligible_surplus_portions": confirmed_eligible,
            "gross_offered_portions": gross_offered,
            "net_accepted_portions": net_accepted,
            "verified_collected_portions": verified_collected,
            "currently_reserved_pending_pickup_portions": int(listing_agg.get("current_reserved_portions", 0)),
            "currently_available_unallocated_portions": int(listing_agg.get("current_unallocated_portions", 0)),
            "uncollected_or_expired_portions": expired_or_uncollected,
        },
        "derived_estimates": {
            "redistribution_recovery_rate_percent": redistribution_recovery_rate,
            "redistribution_recovery_denominator": "verified_collected_portions / confirmed_eligible_surplus_portions",
            "collection_fulfillment_rate_percent": collection_fulfillment_rate,
            "collection_fulfillment_denominator": "verified_collected_portions / net_accepted_portions",
            "assumed_unit_meal_cost_inr": round(avg_unit_cost, 2),
            "estimated_cost_of_historical_discarded_food_inr": round(total_discarded * avg_unit_cost, 2),
            "estimated_value_of_collected_redistribution_inr": round(verified_collected * avg_unit_cost, 2),
            "simulated_production_cost_avoided_inr": round(simulated_overproduction_avoided * avg_unit_cost, 2),
            "qualification_note": (
                "Financial figures are derived estimates based on the configured per-portion production cost "
                f"(INR {avg_unit_cost:.2f}/portion). Carbon emission metrics are intentionally omitted unless verified life-cycle factors are supplied."
            ),
        },
    }
