"""Automated tests for synthetic data, validation, leakage prevention, baseline & ML forecasting, and cold-start fallback."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.ml.data_preprocessing import parse_and_validate_csv_bytes, validate_meal_dataframe
from backend.ml.evaluation import calculate_forecast_metrics, chronological_split
from backend.ml.feature_engineering import build_single_forecast_features, build_supervised_dataset
from backend.ml.predict_demand import predict_meal_demand
from backend.ml.train_model import train_and_evaluate_pipeline
from scripts.generate_demo_data import generate_synthetic_meal_records


class TestDataAndMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.synthetic_df = generate_synthetic_meal_records(num_days=120, seed=202604)

    def test_synthetic_dataset_generation_is_deterministic_and_labeled(self) -> None:
        df1 = generate_synthetic_meal_records(num_days=40, seed=99)
        df2 = generate_synthetic_meal_records(num_days=40, seed=99)
        self.assertEqual(len(df1), 120)
        pd.testing.assert_frame_equal(df1, df2)
        self.assertTrue((df1["is_synthetic"] == 1).all())
        self.assertTrue((df1["meals_served"] <= df1["meals_prepared"]).all())

    def test_validation_detects_missing_malformed_and_impossible_records(self) -> None:
        bad_df = self.synthetic_df.head(10).copy()
        # Introduce impossible value: meals_served > meals_prepared
        bad_df.loc[0, "meals_served"] = int(bad_df.loc[0, "meals_prepared"]) + 50
        # Introduce negative value
        bad_df.loc[1, "leftover_portions"] = -5
        cleaned, report = validate_meal_dataframe(bad_df)
        self.assertFalse(report.is_valid)
        self.assertIsNone(cleaned)
        self.assertGreaterEqual(report.impossible_value_count, 2)

        # Test invalid file extension rejection
        _, ext_report = parse_and_validate_csv_bytes(b"some content", filename="malicious.exe")
        self.assertFalse(ext_report.is_valid)

    def test_duplicate_detection_and_chronological_sorting(self) -> None:
        dup_df = pd.concat([self.synthetic_df.head(15), self.synthetic_df.head(3)], ignore_index=True)
        # Shuffle order
        dup_df = dup_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        cleaned, report = validate_meal_dataframe(dup_df)
        self.assertTrue(report.is_valid)
        self.assertIsNotNone(cleaned)
        self.assertEqual(report.duplicate_count, 3)
        self.assertEqual(len(cleaned), 15)
        # Verify chronological ordering
        dates = cleaned["service_date"].tolist()
        self.assertEqual(dates, sorted(dates))

    def test_chronological_splitting_has_no_temporal_overlap(self) -> None:
        train_df, val_df, test_df = chronological_split(self.synthetic_df, 0.70, 0.15)
        self.assertLessEqual(train_df["service_date"].max(), val_df["service_date"].min())
        self.assertLessEqual(val_df["service_date"].max(), test_df["service_date"].min())

    def test_prevention_of_target_leakage(self) -> None:
        """Verify that altering future target values has zero effect on past feature rows."""
        base_df = self.synthetic_df.head(60).copy()
        sup_original = build_supervised_dataset(base_df)

        mutated_df = base_df.copy()
        # Drastically alter the last 15 rows' meals_served
        mutated_df.loc[45:, "meals_served"] = 9999
        mutated_df.loc[45:, "meals_prepared"] = 9999
        sup_mutated = build_supervised_dataset(mutated_df)

        # Every feature in rows 0..44 must be 100% identical
        feature_cols = [
            "lag_1d_demand",
            "lag_7d_demand",
            "rolling_7d_mean",
            "rolling_14d_mean",
            "same_weekday_4w_mean",
        ]
        pd.testing.assert_frame_equal(
            sup_original.loc[:44, feature_cols],
            sup_mutated.loc[:44, feature_cols],
        )

    def test_metric_calculations_and_zero_actual_demand_wape_safety(self) -> None:
        y_zero = np.array([0.0, 0.0, 0.0])
        y_pred_zero = np.array([0.0, 0.0, 0.0])
        m_zero = calculate_forecast_metrics(y_zero, y_pred_zero)
        self.assertEqual(m_zero["wape"], 0.0)

        y_pred_nonzero = np.array([5.0, 2.0, 3.0])
        m_nz = calculate_forecast_metrics(y_zero, y_pred_nonzero)
        self.assertEqual(m_nz["wape"], 100.0)
        self.assertAlmostEqual(m_nz["mae"], 10.0 / 3.0, places=2)

    def test_baseline_and_ml_training_and_inference(self) -> None:
        eval_res = train_and_evaluate_pipeline(self.synthetic_df)
        self.assertIn("ml_mae", eval_res)
        self.assertIn("baseline_mae", eval_res)
        self.assertLess(eval_res["ml_mae"], eval_res["baseline_mae"])

        pred = predict_meal_demand(
            historical_df=self.synthetic_df,
            forecast_date="2026-10-16",
            meal_type="lunch",
            menu_category="festive_thali",
        )
        self.assertGreater(pred["predicted_demand"], 100.0)
        self.assertFalse(pred["is_fallback_baseline"])
        self.assertGreaterEqual(pred["upper_bound_demand"], pred["rounded_operational_demand"])

    def test_cold_start_fallback_when_history_insufficient(self) -> None:
        tiny_df = self.synthetic_df.head(6).copy()
        pred = predict_meal_demand(
            historical_df=tiny_df,
            forecast_date="2026-10-16",
            meal_type="breakfast",
            menu_category="north_indian_standard",
        )
        self.assertTrue(pred["is_fallback_baseline"])
        self.assertEqual(pred["model_version"], "cold_start_baseline_v1")
        self.assertGreater(pred["rounded_operational_demand"], 0)


if __name__ == "__main__":
    unittest.main()
