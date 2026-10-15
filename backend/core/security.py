"""Authentication, password hashing (PBKDF2-HMAC-SHA256), signed bearer tokens, and RBAC."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from backend.core.config import settings

# Role constants
ROLE_CAFETERIA_ADMIN = "cafeteria_admin"
ROLE_RECIPIENT_ORG = "recipient_org"
ROLE_PROJECT_ADMIN = "project_admin"

VALID_ROLES = {ROLE_CAFETERIA_ADMIN, ROLE_RECIPIENT_ORG, ROLE_PROJECT_ADMIN}

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    ROLE_CAFETERIA_ADMIN: [
        "meals:read",
        "meals:write",
        "forecasts:read",
        "forecasts:write",
        "models:train",
        "plans:read",
        "plans:write",
        "surplus:read",
        "surplus:write",
        "surplus:approve",
        "recipients:read",
        "offers:create",
        "offers:read",
        "collections:confirm",
        "analytics:read",
        "settings:read",
        "settings:write",
    ],
    ROLE_RECIPIENT_ORG: [
        "surplus:read",
        "recipients:read",
        "offers:read",
        "offers:respond",
        "collections:confirm",
        "analytics:read",
    ],
    ROLE_PROJECT_ADMIN: [
        "meals:read",
        "meals:write",
        "forecasts:read",
        "forecasts:write",
        "models:train",
        "plans:read",
        "plans:write",
        "surplus:read",
        "surplus:write",
        "surplus:approve",
        "recipients:read",
        "recipients:write",
        "offers:create",
        "offers:read",
        "offers:respond",
        "collections:confirm",
        "analytics:read",
        "settings:read",
        "settings:write",
        "users:read",
        "users:write",
        "audit:read",
    ],
}


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with 200,000 iterations."""
    if not password:
        raise ValueError("Password cannot be empty.")
    salt_hex = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_hex.encode("utf-8"),
        200_000,
    ).hex()
    return f"pbkdf2_sha256$200000${salt_hex}${derived}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash in constant time."""
    try:
        parts = stored_hash.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt_hex = parts[2]
        expected_hex = parts[3]
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt_hex.encode("utf-8"),
            iterations,
        ).hex()
        return hmac.compare_digest(derived, expected_hex)
    except Exception:
        return False


def create_access_token(
    user_id: int,
    username: str,
    role: str,
    recipient_id: Optional[int] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    """Create a cryptographically signed HMAC-SHA256 bearer token."""
    exp_mins = expires_minutes or settings.TOKEN_EXPIRE_MINUTES
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "username": username,
        "role": role,
        "recipient_id": recipient_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + int(exp_mins * 60),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    signature = hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        b64_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{b64_payload}.{signature}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a signed bearer token; returns None if invalid or expired."""
    try:
        if "." not in token:
            return None
        b64_payload, signature = token.split(".", 1)
        expected_sig = hmac.new(
            settings.SECRET_KEY.encode("utf-8"),
            b64_payload.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        padding = "=" * (-len(b64_payload) % 4)
        payload_bytes = base64.urlsafe_b64decode(b64_payload + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def has_permission(role: str, permission: str) -> bool:
    """Check whether a given role has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])


class InMemoryRateLimiter:
    """Simple sliding-window rate limiter suitable for prototype/local deployments."""

    def __init__(self, max_requests_per_minute: int = 240) -> None:
        self.max_requests = max_requests_per_minute
        self._buckets: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_key: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        timestamps = [t for t in self._buckets[client_key] if t > window_start]
        if len(timestamps) >= self.max_requests:
            self._buckets[client_key] = timestamps
            return False
        timestamps.append(now)
        self._buckets[client_key] = timestamps
        return True


rate_limiter = InMemoryRateLimiter(settings.RATE_LIMIT_PER_MINUTE)
