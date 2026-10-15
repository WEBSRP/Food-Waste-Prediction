# REST API Reference (`docs/api_reference.md`)

Base URL: `http://127.0.0.1:8000/api/v1`  
Interactive Swagger UI: `http://127.0.0.1:8000/docs`

## 1. Authentication & System Endpoints

| Method | Endpoint | Permission Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Public | Returns service health, DB record count, and active model version |
| `POST` | `/api/v1/auth/login` | Public | Validates credentials and returns signed Bearer token |
| `GET` | `/api/v1/auth/me` | Authenticated | Returns active user profile and role |
| `GET` | `/api/v1/auth/demo-accounts` | Public | Lists synthetic demo accounts for college evaluation |
| `GET` | `/api/v1/audit-events` | `audit:read` (`project_admin`) | Lists immutable system audit events |

## 2. Historical Meal Records & CSV Validation

| Method | Endpoint | Permission Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/meals` | `meals:read` | Paginated query of historical meal-service records |
| `POST` | `/api/v1/meals` | `meals:write` | Creates a validated meal-service record (`409` if duplicate date+meal) |
| `PUT` | `/api/v1/meals/{record_id}` | `meals:write` | Updates prepared/served/leftover/discarded quantities |
| `POST` | `/api/v1/meals/validate-csv` | `meals:write` | Dry-run validation report for uploaded `.csv` file |
| `POST` | `/api/v1/meals/import-csv` | `meals:write` | Validates and upserts historical records from `.csv` upload |

## 3. Demand Forecasting & Model Evaluation

| Method | Endpoint | Permission Required | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/models/train` | `models:train` | Trains & evaluates Baseline and ML models on chronological splits |
| `GET` | `/api/v1/models/evaluations` | `forecasts:read` | Lists model evaluation runs, metrics, and feature importances |
| `POST` | `/api/v1/forecasts/predict` | `forecasts:write` | Generates meal demand forecast, uncertainty interval, and factors |
| `GET` | `/api/v1/forecasts/history` | `forecasts:read` | Lists generated forecasts joined with observed actuals |

## 4. Intelligent Production Planning

| Method | Endpoint | Permission Required | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/production-plans` | `plans:write` | Computes and saves recommended production quantity and risk analysis |
| `GET` | `/api/v1/production-plans` | `plans:read` | Lists saved production plans and post-service actuals |
| `PUT` | `/api/v1/production-plans/{plan_id}/actuals` | `plans:write` | Records post-service actual production, consumption, and waste |

## 5. Surplus Management, Recipient Matching & Redistribution

| Method | Endpoint | Permission Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/recipients` | `recipients:read` | Lists recipient directory (`active_only` filter supported) |
| `POST` | `/api/v1/recipients` | `recipients:write` | Registers a new recipient organization |
| `GET` | `/api/v1/surplus-listings` | `surplus:read` | Lists surplus listings (`status_filter`, `food_category`, `min_quantity`) |
| `POST` | `/api/v1/surplus-listings` | `surplus:write` | Creates a surplus listing (`DRAFT` or `AVAILABLE` if approved) |
| `POST` | `/api/v1/surplus-listings/{id}/approve` | `surplus:approve` | Confirms human safety check and transitions listing to `AVAILABLE` |
| `GET` | `/api/v1/surplus-listings/{id}/matches` | `recipients:read` | Runs hard-constraint filtering and multi-criteria recipient ranking |
| `POST` | `/api/v1/offers` | `offers:create` | Atomically reserves surplus portions and creates a `PENDING` offer |
| `GET` | `/api/v1/offers` | `offers:read` | Lists offers (scoped automatically for `recipient_org` users) |
| `POST` | `/api/v1/offers/{offer_id}/respond` | `offers:respond` | Accepts, declines, or cancels an offer and updates listing inventory |
| `POST` | `/api/v1/offers/{offer_id}/collect` | `collections:confirm` | Confirms physical handover and records `COLLECTION_CONFIRMED` |

## 6. Analytics & Impact Measurement

| Method | Endpoint | Permission Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/analytics/impact` | `analytics:read` | Returns 4-tier auditable impact report with explicit denominators |
| `GET` | `/api/v1/analytics/transactions` | `analytics:read` | Returns immutable redistribution transaction ledger |
