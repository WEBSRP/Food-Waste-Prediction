"""Intelligent production planning engine with quantile safety buffers and capacity constraints."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from scipy.stats import norm  # type: ignore


def compute_production_recommendation(
    predicted_demand: float,
    planned_production: int,
    production_capacity: int,
    uncertainty_std: float = 20.0,
    shortage_risk_tolerance: float = 0.15,
    unit_cost_inr: float = 65.0,
    shortage_penalty_inr: float = 110.0,
    custom_buffer_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Recommend optimal meal production quantity balancing shortage risk against surplus waste.

    Policy:
      recommended_quantity = min(production_capacity, ceil(predicted_demand + safety_buffer))

    The safety buffer is derived from the configured `shortage_risk_tolerance` (alpha) and
    cost-asymmetry critical fractile using the held-out forecast error distribution.
    """
    if production_capacity <= 0:
        raise ValueError("production_capacity must be strictly positive.")
    if planned_production < 0:
        raise ValueError("planned_production cannot be negative.")
    if predicted_demand < 0:
        raise ValueError("predicted_demand cannot be negative.")

    # Handle zero demand edge case cleanly
    if predicted_demand == 0.0:
        return {
            "predicted_demand": 0.0,
            "rounded_demand": 0,
            "planned_production": int(planned_production),
            "recommended_production": 0,
            "production_capacity": int(production_capacity),
            "shortage_risk_tolerance": round(float(shortage_risk_tolerance), 3),
            "safety_buffer_portions": 0,
            "capacity_constrained": False,
            "unmet_capacity_gap": 0,
            "expected_surplus_existing_plan": int(planned_production),
            "expected_surplus_recommended_plan": 0,
            "potential_waste_reduction_portions": int(planned_production),
            "expected_shortage_risk_existing": 0.0,
            "expected_shortage_risk_recommended": 0.0,
            "estimated_savings_inr": round(float(planned_production) * float(unit_cost_inr), 2),
            "explanation": "Predicted demand is 0 portions (e.g., full mess closure); recommended production is 0 portions.",
        }

    tol = float(min(max(shortage_risk_tolerance, 0.01), 0.50))
    std = max(1.0, float(uncertainty_std))

    # Critical fractile from cost/penalty ratio blended with explicit risk tolerance
    if (unit_cost_inr + shortage_penalty_inr) > 0:
        cost_fractile = float(shortage_penalty_inr / (shortage_penalty_inr + unit_cost_inr))
    else:
        cost_fractile = 1.0 - tol
    target_service_level = 0.65 * (1.0 - tol) + 0.35 * cost_fractile
    target_service_level = min(max(target_service_level, 0.50), 0.99)

    z_score = float(norm.ppf(target_service_level))
    if custom_buffer_override is not None:
        safety_buffer = max(0, int(custom_buffer_override))
    else:
        safety_buffer = max(0, int(math.ceil(z_score * std)))

    uncapped_target = int(math.ceil(predicted_demand + safety_buffer))
    capacity_constrained = uncapped_target > production_capacity
    recommended_production = min(int(production_capacity), uncapped_target)
    unmet_capacity_gap = max(0, uncapped_target - int(production_capacity))

    # Expected surplus = E[max(0, Q - D)] under Normal(predicted_demand, std^2)
    def _expected_surplus(q: int) -> int:
        diff = float(q) - predicted_demand
        z = diff / std
        exp_overage = diff * float(norm.cdf(z)) + std * float(norm.pdf(z))
        return max(0, int(round(exp_overage)))

    # Shortage probability = P(D > Q) = 1 - Phi((Q - mu)/sigma)
    def _shortage_prob(q: int) -> float:
        z = (float(q) - predicted_demand) / std
        return round(float(1.0 - norm.cdf(z)), 4)

    exp_surplus_existing = _expected_surplus(planned_production)
    exp_surplus_recommended = _expected_surplus(recommended_production)
    risk_existing = _shortage_prob(planned_production)
    risk_recommended = _shortage_prob(recommended_production)

    potential_reduction = max(0, exp_surplus_existing - exp_surplus_recommended)
    estimated_savings_inr = round(float(potential_reduction) * float(unit_cost_inr), 2)

    reasons = [
        f"Forecast mean demand is {predicted_demand:.1f} portions (operational ceiling: {math.ceil(predicted_demand)} portions).",
        f"Applied a +{safety_buffer} portion safety buffer (target service level {target_service_level*100:.1f}%, σ={std:.1f}) to cap shortage risk at {risk_recommended*100:.1f}%.",
    ]
    if capacity_constrained:
        reasons.append(
            f"Uncapped recommendation ({uncapped_target} portions) exceeded kitchen capacity ({production_capacity} portions); capped at {production_capacity} portions."
        )
    if planned_production > recommended_production:
        reasons.append(
            f"Current plan ({planned_production} portions) exceeds recommendation by {planned_production - recommended_production} portions, creating an estimated {exp_surplus_existing} portions of potential surplus."
        )
    elif planned_production < recommended_production:
        reasons.append(
            f"Current plan ({planned_production} portions) carries a {risk_existing*100:.1f}% shortage risk; increasing to {recommended_production} portions reduces stockout probability to {risk_recommended*100:.1f}%."
        )

    return {
        "predicted_demand": round(float(predicted_demand), 1),
        "rounded_demand": int(math.ceil(predicted_demand)),
        "planned_production": int(planned_production),
        "recommended_production": int(recommended_production),
        "production_capacity": int(production_capacity),
        "shortage_risk_tolerance": round(tol, 3),
        "safety_buffer_portions": int(safety_buffer),
        "capacity_constrained": bool(capacity_constrained),
        "unmet_capacity_gap": int(unmet_capacity_gap),
        "expected_surplus_existing_plan": int(exp_surplus_existing),
        "expected_surplus_recommended_plan": int(exp_surplus_recommended),
        "potential_waste_reduction_portions": int(potential_reduction),
        "expected_shortage_risk_existing": risk_existing,
        "expected_shortage_risk_recommended": risk_recommended,
        "estimated_savings_inr": estimated_savings_inr,
        "explanation": " ".join(reasons),
    }
