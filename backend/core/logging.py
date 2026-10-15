"""Structured application logging without exposing sensitive credentials."""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict

SENSITIVE_KEYS = {"password", "password_hash", "token", "access_token", "secret", "authorization"}


def redact_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive fields before writing to logs or audit trails."""
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in SENSITIVE_KEYS:
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, dict):
            sanitized[k] = redact_sensitive(v)
        else:
            sanitized[k] = v
    return sanitized


def get_logger(name: str = "food_waste_intelligence") -> logging.Logger:
    """Configure and return an application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
