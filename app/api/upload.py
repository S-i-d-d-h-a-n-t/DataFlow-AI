"""
Dataset upload and management endpoints.

Routes:
  POST   /api/v1/datasets/upload          Upload a CSV file
  GET    /api/v1/datasets                 List all datasets (paginated)
  GET    /api/v1/datasets/{id}            Get full dataset metadata
  GET    /api/v1/datasets/{id}/preview    Preview first N rows
  GET    /api/v1/datasets/{id}/structure  Full structural analysis
  PATCH  /api/v1/datasets/{id}            Update name / description
  DELETE /api/v1/datasets/{id}            Delete dataset + file

Business logic lives entirely in DatasetService — this file only handles
HTTP concerns: request parsing, dependency injection, error → HTTP mapping.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import DatasetRead, DatasetSummary, DatasetUpdate
from app.services.dataset_service import (
    DatasetService,
    DatasetNotFoundError,
    InvalidFileTypeError,
    FileTooLargeError,
    ParseError,
    DatasetServiceError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Dependency ────────────────────────────────────────────────────────────────

def get_dataset_service(db: Session = Depends(get_db)) -> DatasetService:
    """Build the service with its repository — injected into every endpoint."""
    return DatasetService(repo=DatasetRepository(db))


# ── Exception → HTTP mapping ──────────────────────────────────────────────────

def _handle_service_error(exc: DatasetServiceError) -> HTTPException:
    if isinstance(exc, DatasetNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, InvalidFileTypeError):
        return HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))
    if isinstance(exc, FileTooLargeError):
        return HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
    if isinstance(exc, ParseError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV dataset",
    description=(
        "Upload a CSV file. The platform will automatically detect column types, "
        "count rows/columns, and store structural metadata. "
        "Supported encodings: UTF-8, Latin-1, CP1252. Max size: 500 MB."
    ),
)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file to upload"),
    name: str | None = Form(None, description="Human-readable dataset name (defaults to filename)"),
    description: str | None = Form(None, description="Optional description"),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetRead:
    logger.info(f"Upload request: filename={file.filename} content_type={file.content_type}")

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    try:
        content = await file.read()
        dataset = service.upload(
            filename=file.filename,
            file_content=content,
            name=name,
            description=description,
        )
        return DatasetRead.model_validate(dataset)
    except DatasetServiceError as exc:
        raise _handle_service_error(exc) from exc
    finally:
        await file.close()


@router.get(
    "",
    response_model=list[DatasetSummary],
    summary="List all datasets",
)
def list_datasets(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=200, description="Maximum records to return"),
    service: DatasetService = Depends(get_dataset_service),
) -> list[DatasetSummary]:
    datasets = service.list_all(skip=skip, limit=limit)
    return [DatasetSummary.model_validate(d) for d in datasets]


@router.get(
    "/{dataset_id}",
    response_model=DatasetRead,
    summary="Get dataset metadata",
)
def get_dataset(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetRead:
    try:
        dataset = service.get_by_id(dataset_id)
        return DatasetRead.model_validate(dataset)
    except DatasetServiceError as exc:
        raise _handle_service_error(exc) from exc


@router.get(
    "/{dataset_id}/preview",
    summary="Preview first N rows of a dataset",
    response_model=dict,
)
def preview_dataset(
    dataset_id: str,
    n_rows: int = Query(5, ge=1, le=100, description="Number of rows to return"),
    service: DatasetService = Depends(get_dataset_service),
) -> dict:
    try:
        return service.get_preview(dataset_id, n_rows=n_rows)
    except DatasetServiceError as exc:
        raise _handle_service_error(exc) from exc


@router.get(
    "/{dataset_id}/structure",
    summary="Full structural analysis of a dataset",
    response_model=dict,
)
def dataset_structure(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> dict:
    """
    Returns column types, missing value report, cardinality, per-column stats,
    and memory usage. This is the payload consumed by the Planner Agent.
    """
    try:
        return service.detect_structure(dataset_id)
    except DatasetServiceError as exc:
        raise _handle_service_error(exc) from exc


@router.patch(
    "/{dataset_id}",
    response_model=DatasetRead,
    summary="Update dataset name or description",
)
def update_dataset(
    dataset_id: str,
    payload: DatasetUpdate,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetRead:
    try:
        dataset = service.update(dataset_id, payload)
        return DatasetRead.model_validate(dataset)
    except DatasetServiceError as exc:
        raise _handle_service_error(exc) from exc


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dataset and its file",
    response_model=None,
)
def delete_dataset(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> None:
    try:
        service.delete(dataset_id)
    except DatasetServiceError as exc:
        raise _handle_service_error(exc) from exc
