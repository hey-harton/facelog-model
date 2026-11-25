"""Minimal FastAPI serving entrypoint for facelog-model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORTED_MODEL = Path("models/exported/face_embedder.onnx")

app = FastAPI(title="Facelog Model API", version="0.1.0")


def _resolve_exported_model_path() -> Path:
    """Read embedder.exported_path from configs/model.yaml with safe fallback."""
    config_path = REPO_ROOT / "configs" / "model.yaml"
    if not config_path.exists():
        return REPO_ROOT / DEFAULT_EXPORTED_MODEL

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    exported_path = (
        data.get("embedder", {}).get("exported_path") or DEFAULT_EXPORTED_MODEL
    )
    candidate = Path(exported_path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return service health and exported model readiness."""
    model_path = _resolve_exported_model_path()
    return {
        "status": "ok",
        "service": "facelog-model",
        "model_path": str(model_path),
        "model_ready": model_path.exists(),
    }
