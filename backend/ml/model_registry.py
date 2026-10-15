"""Model registry and persistence layer using joblib (with pickle compatibility fallback)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import pickle

from backend.core.config import MODELS_DIR

try:
    import joblib  # type: ignore
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


def save_model_artifact(artifact: Dict[str, Any], model_version: str) -> str:
    """Save a trained model bundle to data/models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"demand_model_{model_version}.pkl"
    latest_path = MODELS_DIR / "demand_model_latest.pkl"
    if _HAS_JOBLIB:
        joblib.dump(artifact, filepath)
        joblib.dump(artifact, latest_path)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
        with open(latest_path, "wb") as f:
            pickle.dump(artifact, f)
    return str(filepath)


def load_model_artifact(model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a trained model bundle by version or load the latest registered artifact."""
    if model_version:
        candidate = MODELS_DIR / f"demand_model_{model_version}.pkl"
        if candidate.exists():
            return _load_file(candidate)
    latest_path = MODELS_DIR / "demand_model_latest.pkl"
    if latest_path.exists():
        return _load_file(latest_path)
    return None


def _load_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if _HAS_JOBLIB:
            return joblib.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
