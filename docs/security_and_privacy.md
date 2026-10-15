# Security, Privacy & Operational Reliability (`docs/security_and_privacy.md`)

## 1. Authentication, Password Hashing & RBAC

- **Password Storage:** Passwords are never stored or logged in plaintext. `backend/core/security.py` hashes passwords using `PBKDF2-HMAC-SHA256` with `200,000` iterations and a 16-byte cryptographic salt (`secrets.token_hex(16)`), verified in constant time via `hmac.compare_digest`.
- **Signed Access Tokens:** Authentication tokens are signed with `HMAC-SHA256` using `SECRET_KEY` and enforce expiration (`exp`).
- **Backend Role Enforcement:** Every protected FastAPI route uses `Depends(require_permission("..."))`. Furthermore, `POST /api/v1/offers/{offer_id}/respond` and `/collect` verify that a `recipient_org` user's `recipient_id` matches the target offer's `recipient_id`, preventing horizontal privilege escalation.

## 2. Input Validation, Safe CSV Parsing & Concurrency Control

- **Parameterized Queries:** All SQL queries in `backend/database/session.py` use parameterized `?` placeholders, preventing SQL injection.
- **Safe CSV Upload:** `parse_and_validate_csv_bytes()` enforces a `.csv` extension check, a 10 MB payload limit, UTF-8 decoding, and strips leading spreadsheet formula triggers (`=`, `+`, `-`, `@`) from string columns. Uploaded files are never executed.
- **Transactional Concurrency Control:** Surplus reservations, offer responses, and collection confirmations execute inside `with session.transaction(immediate=True):` (`BEGIN IMMEDIATE` + thread lock), preventing concurrent requests from overbooking available surplus or producing negative inventory.

## 3. Prototype Controls vs. Production Hardening Checklist

| Security Control | Prototype Implementation (ORIGIN MINI 2026 MVP) | Required Production Hardening |
| :--- | :--- | :--- |
| **Identity Management** | Controlled local RBAC with pre-seeded demo accounts switcher for evaluators | Integrate institutional SSO (OAuth2 / OIDC / SAML) with MFA and token revocation |
| **Database Engine** | Local SQLite with `BEGIN IMMEDIATE` write locks | Migrate `DATABASE_URL` to managed PostgreSQL with row-level `SELECT ... FOR UPDATE` locks |
| **Rate Limiting** | In-process sliding-window rate limiter (`InMemoryRateLimiter`) | Deploy Redis-backed distributed rate limiting and reverse-proxy WAF (Nginx / Cloudflare) |
| **Transport Security** | Local HTTP (`127.0.0.1`) for classroom demo | Enforce TLS 1.3 (HTTPS) and strict CORS origins |
| **Secrets Management** | `.env` file loaded via `python-dotenv` | Store `SECRET_KEY` and DB credentials in AWS Secrets Manager / HashiCorp Vault |
