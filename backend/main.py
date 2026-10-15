"""FastAPI application entry point for AI-Powered Food Waste Prediction and Redistribution Intelligence."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from backend.core.config import settings
from backend.database.session import init_db

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "ORIGIN MINI 2026 — Problem Statement 04 (Galgotias University, SCSE). "
        "End-to-end ML demand forecasting, quantile production planning, surplus safety workflow, "
        "and explainable recipient redistribution API."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(api_router)
