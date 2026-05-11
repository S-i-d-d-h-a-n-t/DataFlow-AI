"""
Security utilities — API key validation, token helpers.
Placeholder for Phase 11 when the public API is hardened.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Usage:
        @router.post("/run", dependencies=[Depends(verify_api_key)])

    NOTE: API key enforcement is disabled until Phase 11.
          Set API_KEY in .env to enable it.
    """
    expected = getattr(settings, "API_KEY", None)
    if not expected:
        # Security not yet configured — allow all requests
        return "unauthenticated"
    if api_key != expected:
        logger.warning("Rejected request with invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return api_key
