"""Unified API client for the Streamlit interface.

Communicates with the FastAPI REST endpoints over HTTP when a separate Uvicorn server
is running, and automatically falls back to FastAPI's in-process ASGI TestClient when
invoked standalone so business logic is never duplicated inside Streamlit pages.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import os

import requests
from fastapi.testclient import TestClient

from backend.core.config import settings
from backend.main import app


class BackendAPIClient:
    """Client wrapping REST API calls with Bearer token headers."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("API_BASE_URL", settings.API_BASE_URL)).rstrip("/")
        self._in_process_client = TestClient(app)
        self.token: Optional[str] = None

    def set_token(self, token: Optional[str]) -> None:
        self.token = token

    def _headers(self) -> Dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Issue an API request via HTTP or in-process ASGI TestClient."""
        api_path = path if path.startswith("/api/") else f"/api/v1/{path.lstrip('/')}"
        try:
            resp = requests.request(
                method=method.upper(),
                url=f"{self.base_url}{api_path}",
                headers=self._headers(),
                json=json_body,
                params=params,
                files=files,
                timeout=3.5,
            )
            return {"status_code": resp.status_code, "data": resp.json() if resp.content else {}}
        except requests.RequestException:
            # Seamless fallback to in-process FastAPI ASGI app
            resp = self._in_process_client.request(
                method=method.upper(),
                url=api_path,
                headers=self._headers(),
                json=json_body,
                params=params,
                files=files,
            )
            return {"status_code": resp.status_code, "data": resp.json() if resp.content else {}}


api_client = BackendAPIClient()
