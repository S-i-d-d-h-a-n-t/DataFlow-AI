"""
Report retrieval endpoints.

Routes:
  GET  /api/v1/reports/{experiment_id}           Report metadata (paths + existence)
  GET  /api/v1/reports/{experiment_id}/markdown  Raw Markdown content
  GET  /api/v1/reports/{experiment_id}/html      Rendered HTML content

Business logic lives in ReportService.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import get_logger
from app.services.report_service import (
    ReportService,
    ReportNotFoundError,
    ExperimentNotFoundError as ReportExpNotFoundError,
    ReportServiceError,
)

logger = get_logger(__name__)
router = APIRouter()


# ── Dependency ────────────────────────────────────────────────────────────────

def get_report_service(db: Session = Depends(get_db)) -> ReportService:
    return ReportService(db)


# ── Exception → HTTP mapping ──────────────────────────────────────────────────

def _report_error_to_http(exc: ReportServiceError) -> HTTPException:
    if isinstance(exc, ReportExpNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ReportNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/{experiment_id}",
    response_model=dict,
    summary="Report metadata",
    description="Returns the file paths and existence flags for both report formats.",
)
def get_report_metadata(
    experiment_id: str,
    service: ReportService = Depends(get_report_service),
) -> dict:
    try:
        return service.get_report_paths(experiment_id)
    except ReportServiceError as exc:
        raise _report_error_to_http(exc) from exc


@router.get(
    "/{experiment_id}/markdown",
    response_class=PlainTextResponse,
    summary="Download Markdown report",
    description="Returns the raw Markdown report as plain text.",
)
def get_report_markdown(
    experiment_id: str,
    service: ReportService = Depends(get_report_service),
) -> str:
    try:
        return service.get_markdown(experiment_id)
    except ReportServiceError as exc:
        raise _report_error_to_http(exc) from exc


@router.get(
    "/{experiment_id}/html",
    response_class=HTMLResponse,
    summary="View HTML report",
    description="Returns the rendered HTML report. Open directly in a browser.",
)
def get_report_html(
    experiment_id: str,
    service: ReportService = Depends(get_report_service),
) -> str:
    try:
        return service.get_html(experiment_id)
    except ReportServiceError as exc:
        raise _report_error_to_http(exc) from exc
