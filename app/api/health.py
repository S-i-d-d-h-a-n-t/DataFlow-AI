"""
Health check endpoints.
Provides liveness and readiness probes suitable for Docker / Kubernetes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel

from app.core.database import get_db
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    database: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health_check() -> HealthResponse:
    """
    Returns 200 if the application process is alive.
    Does NOT check downstream dependencies.
    """
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
def readiness_check(db: Session = Depends(get_db)) -> ReadinessResponse:
    """
    Returns 200 only when the application can reach the database.
    Use this as a readiness probe in orchestration systems.
    """
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
        logger.debug("Readiness check passed — database reachable.")
    except Exception as exc:
        logger.error(f"Readiness check failed — database unreachable: {exc}")
        db_status = "unreachable"

    overall = "ok" if db_status == "ok" else "degraded"
    return ReadinessResponse(status=overall, database=db_status)
