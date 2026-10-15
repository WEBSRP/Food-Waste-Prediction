"""FastAPI authentication, RBAC permission dependencies, and rate-limiting checks."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from backend.core.security import decode_access_token, has_permission, rate_limiter
from backend.database.session import DatabaseSession, get_db


def check_rate_limit(request: Request) -> None:
    """Enforce basic sliding-window API rate limit per client host."""
    client_host = request.client.host if request.client else "local"
    if not rate_limiter.is_allowed(client_host):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before issuing more requests.",
        )


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: DatabaseSession = Depends(get_db),
) -> Dict[str, Any]:
    """Authenticate Bearer token and return active user record."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization Bearer header.",
        )
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        )
    user = db.fetch_one(
        "SELECT id, username, full_name, email, role, recipient_id, is_active, is_demo_account FROM users WHERE id = ?",
        (int(payload["user_id"]),),
    )
    if not user or int(user.get("is_active", 0)) != 1:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or inactive.",
        )
    return user


def require_permission(permission: str) -> Callable[..., Dict[str, Any]]:
    """Dependency factory enforcing role-based permission on backend endpoints."""

    def _dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        role = str(current_user.get("role", ""))
        if not has_permission(role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' is not authorized for permission '{permission}'.",
            )
        return current_user

    return _dependency
