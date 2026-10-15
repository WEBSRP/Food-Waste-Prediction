# Testing & Chronological Model Evaluation Report (`docs/testing_and_evaluation.md`)

## 1. Automated Test Suite Execution Summary

Command executed:
```bash
python3 -m unittest discover -s tests -v
```

Result: **16 tests executed in 12.402s — 16 PASSED, 0 FAILED**

### Test Coverage Matrix

| Test Suite | Test Case | Verified Requirement | Status |
| :--- | :--- | :--- | :---: |
| `test_ml_pipeline.py` | `test_synthetic_dataset_generation_is_deterministic_and_labeled` | Deterministic seed, physical bounds (`served <= prepared`), `is_synthetic == 1` | PASS |
| `test_ml_pipeline.py` | `test_validation_detects_missing_malformed_and_impossible_records` | Rejects `meals_served > meals_prepared`, negative values, and non-`.csv` files | PASS |
| `test_ml_pipeline.py` | `test_duplicate_detection_and_chronological_sorting` | Deduplicates `(service_date, meal_type)` and enforces chronological order | PASS |
| `test_ml_pipeline.py` | `test_chronological_splitting_has_no_temporal_overlap` | Strict temporal separation (`train_max <= val_min <= test_min`) | PASS |
| `test_ml_pipeline.py` | `test_prevention_of_target_leakage` | Future target mutations (`9999`) have zero effect on past lag/rolling features | PASS |
| `test_ml_pipeline.py` | `test_metric_calculations_and_zero_actual_demand_wape_safety` | Accurate MAE, RMSE, Bias, and safe division-by-zero handling in WAPE | PASS |
| `test_ml_pipeline.py` | `test_baseline_and_ml_training_and_inference` | ML model trains, beats baseline on held-out split, and produces intervals | PASS |
| `test_ml_pipeline.py` | `test_cold_start_fallback_when_history_insufficient` | Falls back to `cold_start_baseline_v1` when `< 21` historical records exist | PASS |
| `test_production_planning.py` | `test_demand_lower_than_planned_production` | Identifies excess planned production and quantifies expected surplus reduction | PASS |
| `test_production_planning.py` | `test_demand_higher_than_production_capacity` | Caps recommendation at `production_capacity` and flags `unmet_capacity_gap` | PASS |
| `test_production_planning.py` | `test_zero_demand_edge_case` | Recommends `0` portions with `0` buffer when predicted demand is `0` | PASS |
| `test_production_planning.py` | `test_uncertainty_buffers_and_integer_rounding` | Stricter shortage risk tolerance (`0.05` vs `0.35`) increases integer buffer | PASS |
| `test_redistribution_and_api.py` | `test_recipient_matching_scenarios_none_one_and_multiple` | Verifies 0, 1, and multiple eligible recipient ranking and exclusion reasons | PASS |
| `test_redistribution_and_api.py` | `test_expired_listing_and_duplicate_reservation_and_capacity_limits` | Blocks capacity overflow, duplicate active offers, and expired listings | PASS |
| `test_redistribution_and_api.py` | `test_concurrent_allocation_attempts_prevent_overbooking` | Multi-threaded concurrent reservations respect atomic `BEGIN IMMEDIATE` lock | PASS |
| `test_redistribution_and_api.py` | `test_api_authentication_rbac_and_end_to_end_workflow` | Enforces `401`/`403` RBAC and verifies full forecast-to-collection API flow | PASS |

## 2. Held-Out Chronological Forecasting Evaluation

- **Training + Validation Window:** `2025-10-29` to `2026-08-22` (`895` meal services)
- **Held-Out Test Window:** `2026-08-23` to `2026-10-14` (`158` meal services)

| Model | MAE (Portions) | RMSE (Portions) | WAPE (%) | Forecast Bias (Portions) | Selected Primary |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Weekday Moving Average (Baseline)** | `49.023` | `63.455` | `11.178%` | `-4.541` | No |
| **Gradient Boosted Regression Trees (ML)** | **`19.898`** | **`25.616`** | **`4.537%`** | **`+2.210`** | **Yes (`ml_model`)** |

### Top Permutation Feature Importances (Held-Out Test Split)
1. `same_weekday_4w_mean`: `32.51%`
2. `menu_category_code`: `22.70%`
3. `rolling_7d_mean`: `19.72%`
4. `day_of_week`: `9.98%`
5. `is_holiday`: `4.20%`
6. `lag_1d_demand`: `3.89%`
