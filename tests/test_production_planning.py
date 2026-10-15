"""Automated tests for intelligent production planning edge cases and constraints."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.domain.production_planner import compute_production_recommendation


class TestProductionPlanning(unittest.TestCase):
    def test_demand_lower_than_planned_production(self) -> None:
        res = compute_production_recommendation(
            predicted_demand=420.0,
            planned_production=550,
            production_capacity=700,
            uncertainty_std=20.0,
            shortage_risk_tolerance=0.15,
        )
        self.assertLess(res["recommended_production"], res["planned_production"])
        self.assertGreater(res["expected_surplus_existing_plan"], res["expected_surplus_recommended_plan"])
        self.assertGreater(res["potential_waste_reduction_portions"], 0)

    def test_demand_higher_than_production_capacity(self) -> None:
        res = compute_production_recommendation(
            predicted_demand=780.0,
            planned_production=700,
            production_capacity=750,
            uncertainty_std=25.0,
            shortage_risk_tolerance=0.10,
        )
        self.assertEqual(res["recommended_production"], 750)
        self.assertTrue(res["capacity_constrained"])
        self.assertGreater(res["unmet_capacity_gap"], 0)

    def test_zero_demand_edge_case(self) -> None:
        res = compute_production_recommendation(
            predicted_demand=0.0,
            planned_production=200,
            production_capacity=500,
            uncertainty_std=15.0,
        )
        self.assertEqual(res["recommended_production"], 0)
        self.assertEqual(res["safety_buffer_portions"], 0)
        self.assertEqual(res["expected_surplus_existing_plan"], 200)
        self.assertEqual(res["expected_surplus_recommended_plan"], 0)

    def test_uncertainty_buffers_and_integer_rounding(self) -> None:
        low_risk = compute_production_recommendation(
            predicted_demand=312.3,
            planned_production=350,
            production_capacity=500,
            uncertainty_std=18.0,
            shortage_risk_tolerance=0.05,  # Strict 5% risk tolerance -> larger buffer
        )
        high_risk = compute_production_recommendation(
            predicted_demand=312.3,
            planned_production=350,
            production_capacity=500,
            uncertainty_std=18.0,
            shortage_risk_tolerance=0.35,  # Looser 35% risk tolerance -> smaller buffer
        )
        self.assertIsInstance(low_risk["recommended_production"], int)
        self.assertGreater(low_risk["safety_buffer_portions"], high_risk["safety_buffer_portions"])
        self.assertGreater(low_risk["recommended_production"], high_risk["recommended_production"])


if __name__ == "__main__":
    unittest.main()
