# Machine-Learning Forecasting & Production Planning Methodology (`docs/ml_methodology.md`)

## 1. Distinguishing Observed Consumption vs. True Demand

In cafeteria operations, recorded `meals_served` is right-censored whenever kitchen supply runs out (`meals_served == meals_prepared`).
- When `unmet_demand_portions` is recorded (e.g., via turnstile counts or queue logs after stockout), effective demand is defined as:
  $$D_{\text{effective}} = \text{meals\_served} + \text{unmet\_demand\_portions}$$
- When `unmet_demand_portions == 0` and `meals_served == meals_prepared`, `backend/ml/data_preprocessing.py` flags the observation as potentially right-censored and emits a validation warning so operators know observed sales may underestimate true demand.

## 2. Leakage-Free Chronological Feature Engineering

Every feature vector $\mathbf{x}_{t, m}$ for date $t$ and meal slot $m \in \{\text{breakfast}, \text{lunch}, \text{dinner}\}$ is constructed strictly from forecast-time calendar/menu attributes and historical observations $s < t$:
- **Calendar & Cyclical Features**: `meal_type_code`, `day_of_week`, `is_weekend`, `is_friday`, `is_holiday`, $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$, $\sin(2\pi \cdot \text{dow}/7)$, $\cos(2\pi \cdot \text{dow}/7)$.
- **Operational Context Features**: `menu_category_code`, `special_event_code`, `weather_code`, `temperature_c`.
- **Strictly Past Autoregressive Lags (Per Meal Type)**:
  - `lag_1d_demand`: Demand on the most recent prior service of meal $m$ ($t-1$).
  - `lag_7d_demand`: Demand 7 services prior for meal $m$ ($t-7$).
  - `rolling_7d_mean`, `rolling_14d_mean`, `rolling_7d_std`: Mean and volatility over the strictly preceding 7 and 14 services of meal $m$.
  - `same_weekday_4w_mean`: Mean demand across the last 4 occurrences of the same weekday for meal $m$.

An automated unit test (`test_prevention_of_target_leakage`) mutates future target values (`meals_served = 9999`) and verifies bit-for-bit equality of all preceding feature rows.

## 3. Comparable Forecasting Models

1. **Baseline Model (`SeasonalMovingAverageBaseline`)**:
   - Combines `0.65 * same_weekday_4w_mean + 0.35 * rolling_7d_mean`, multiplied by an empirical meal-specific holiday attendance ratio learned strictly on the training split.
2. **Machine-Learning Model (`HistGradientBoostingRegressor` / `NumpyGradientBoostedTreesRegressor`)**:
   - Fits a stagewise Gradient Boosted Regression Tree ensemble (`85` trees, depth `3`, learning rate `0.12`, minimum leaf size `6`) over the 19 forecast-time features, capturing non-linear interactions between menu category, weekday, holiday status, campus events, and recent demand trends.
3. **Cold-Start Fallback**:
   - If fewer than `21` historical meal records precede the requested `forecast_date`, `predict_meal_demand()` automatically falls back to `cold_start_baseline_v1` and sets `is_fallback_baseline = True`.

## 4. Chronological Evaluation Metrics

Evaluated strictly on held-out chronological test data (`70%` train, `15%` validation, `15%` test — never random shuffling):
- **MAE**: $\frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$
- **RMSE**: $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Safe WAPE (%)**: $\frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\max(\epsilon, \sum_{i=1}^N |y_i|)} \times 100$, returning `0.0%` if both actual and predicted demand are zero and `100.0%` if actual demand is zero while predictions are positive.
- **Forecast Bias**: $\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)$ (positive indicates over-prediction; negative indicates under-prediction).

## 5. Intelligent Production Planning Policy

Instead of blindly recommending the mean forecast $\hat{\mu}$, `compute_production_recommendation()` computes a target service level blending user-configured shortage risk tolerance $\alpha$ and the Newsvendor critical fractile:
$$F_{\text{target}} = 0.65(1 - \alpha) + 0.35\left(\frac{c_{\text{shortage}}}{c_{\text{shortage}} + c_{\text{unit}}}\right)$$
$$\text{safety\_buffer} = \lceil \Phi^{-1}(F_{\text{target}}) \cdot \hat{\sigma}_{\text{residual}} \rceil$$
$$Q_{\text{recommended}} = \min\left(Q_{\text{capacity}}, \lceil \hat{\mu} + \text{safety\_buffer} \rceil\right)$$
Expected surplus under any plan $Q$ is evaluated analytically via the normal loss integral $\mathbb{E}[\max(0, Q - D)] = (Q - \hat{\mu})\Phi(z) + \hat{\sigma}\phi(z)$ where $z = \frac{Q - \hat{\mu}}{\hat{\sigma}}$.
