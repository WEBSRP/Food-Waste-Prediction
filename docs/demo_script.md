# Step-by-Step College Demonstration Script (`docs/demo_script.md`)

**Event:** ORIGIN MINI 2026 — Galgotias University (School of Computer Science & Engineering)  
**Problem Statement 04:** AI-Powered Food Waste Prediction and Redistribution Intelligence

## Automated One-Command Execution

To run all 12 steps automatically against the FastAPI backend and generate `data/exports/demo_scenario_execution_report.json`:
```bash
python3 scripts/run_demo_scenario.py
```

## Live Interactive Walkthrough (12 Steps)

1. **Import or Generate 12-Month Meal-Service History:**
   - Run `python3 scripts/generate_demo_data.py` and `python3 scripts/seed_database.py` (or open **Page 6: Data Management**). Show the evaluators that `1,095` synthetic meal records (`365` days × `3` meals) are loaded and explicitly flagged with `is_synthetic = 1`.
2. **Validate Dataset Integrity:**
   - On **Page 6: Data Management**, upload `data/synthetic/sample_cafeteria_upload_template.csv` and click **Validate CSV Only**. Point out the schema check, duplicate check, impossible-value check (`meals_served <= meals_prepared`), IQR outlier inspection, and right-censored demand check.
3. **Train and Evaluate Baseline vs. ML Models:**
   - Open **Page 1: Demand Forecasting** and inspect the held-out chronological evaluation (`158` test services from `2026-08-23` to `2026-10-14`). Show that the ML model achieves **MAE `19.90` portions (`4.54%` WAPE)** compared to the Seasonal Moving Average baseline's **MAE `49.02` portions (`11.18%` WAPE)** — a **59.41% error reduction**.
4. **Generate Tomorrow's Meal Forecast (`2026-10-16` Lunch):**
   - Select `2026-10-16`, `lunch`, `north_indian_standard` on **Page 1** and click **Predict Meal Demand**. Point out the predicted mean demand, rounded operational demand, Q85 planning buffer, and the table of contributing factors.
5. **Compare Existing Production Plan with Recommended Quantity:**
   - Open **Page 2: Production Planning**. Enter the cafeteria's habitual Friday plan (`640` portions) against the forecast demand (`~525` portions) with `750` capacity and `15%` shortage risk tolerance.
6. **Inspect Estimated Surplus & Shortage Risk:**
   - Show that the habitual `640`-portion plan creates **`115` portions of expected surplus**, whereas the recommended plan (`548` portions including a `+23` portion safety buffer) reduces expected surplus to **`26` portions** while keeping shortage risk within tolerance (`8.4%`), avoiding **`89` portions of simulated overproduction**.
7. **Create & Safety-Approve a Confirmed Surplus Listing:**
   - Open **Page 3: Food Surplus Management** as `cafeteria_admin`. Register `80` portions of `cooked_grains_legumes` (*Chole, Jeera Rice & Steamed Roti Batch*). Explain that listings start in `DRAFT` until a supervisor verifies storage temperature (`64°C` hot-held) and approves the listing to `AVAILABLE`.
8. **Find and Rank Eligible Recipients:**
   - Open **Page 4: Redistribution Intelligence** and select the newly approved `80`-portion listing. Show how the matching engine excludes `Kasna Industrial Worker Meal Bank` (does not accept `cooked_grains_legumes`) and `Unverified Student Volunteer Circle` (`is_verified = 0`), while ranking `Yamuna Expressway Shelter Home` (`Score 94.8`) and `Greater Noida Community Kitchen` (`Score 89.2`) at the top with transparent sub-scores.
9. **Create a Redistribution Offer & Atomic Reservation:**
   - Offer `80` portions to `Yamuna Expressway Shelter Home Foundation`. Show that the listing's `remaining_quantity_portions` atomically drops from `80` to `0`, `reserved_quantity_portions` becomes `80`, and status transitions to `RESERVED`.
10. **Accept the Offer Using a Recipient Demo Account:**
    - In the left sidebar, switch the active role from `cafeteria_admin` to `recipient_yamuna` (`Yamuna Expressway Shelter Home`). Verify that `recipient_yamuna` cannot access admin-only training or listing creation (`403 Forbidden`), then accept the `80`-portion offer (`status -> ACCEPTED`).
11. **Confirm Physical Collection:**
    - Switch back to `cafeteria_admin` (or confirm from `recipient_yamuna`) and record the physical handover of `80` portions (`status -> COLLECTED`).
12. **Review Updated Inventory & Auditable Impact Analytics:**
    - Open **Page 5: Impact Analytics**. Show the four strictly separated metric panels (*Model Predictions*, *Simulated Planning Outcomes*, *Actual Recorded Outcomes*, and *Derived Estimates*) with explicit denominators (`verified_collected_portions / confirmed_eligible_surplus_portions = 66.67%`) and export the JSON audit report.
