"""
Workflow endpoints — trigger and monitor ML pipeline runs.

Routes:
  POST  /api/v1/workflow/run           Trigger a full pipeline run (synchronous)
  GET   /api/v1/workflow/{id}/status   Poll experiment status
  GET   /api/v1/workflow/{id}          Full experiment detail
  GET   /api/v1/workflow               List experiments for a dataset
  GET   /api/v1/workflow/schema        Pipeline graph schema (nodes + edges)

Business logic lives in PipelineService and ExperimentService.
This file handles only HTTP concerns: request parsing, DI, error → HTTP mapping.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.experiment_repository import ExperimentRepository
from app.schemas.experiment import ExperimentRead, ExperimentSummary
from app.schemas.workflow import (
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowStatusResponse,
)
from app.services.pipeline_service import PipelineService, PipelineServiceError
from app.services.experiment_service import (
    ExperimentService,
    ExperimentNotFoundError,
)
from app.workflows.graph import get_graph_schema

logger = get_logger(__name__)
router = APIRouter()


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_pipeline_service(db: Session = Depends(get_db)) -> PipelineService:
    return PipelineService(db)


def get_experiment_service(db: Session = Depends(get_db)) -> ExperimentService:
    return ExperimentService(
        experiment_repo=ExperimentRepository(db),
        dataset_repo=DatasetRepository(db),
    )


# ── Exception → HTTP mapping ──────────────────────────────────────────────────

def _pipeline_error_to_http(exc: PipelineServiceError) -> HTTPException:
    msg = str(exc)
    if "not found" in msg.lower():
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/run",
    response_model=WorkflowRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger a full ML pipeline run",
    description=(
        "Runs the complete pipeline synchronously: "
        "Planner → EDA → Cleaning → Feature Engineering → "
        "Training (parallel) → Evaluation → Report. "
        "Returns the full result including best model, metrics, and report path."
    ),
)
def run_pipeline(
    request: WorkflowRunRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> WorkflowRunResponse:
    logger.info(
        f"Pipeline run requested: dataset={request.dataset_id} "
        f"target={request.target_column} task={request.task_type}"
    )
    try:
        return service.run(request)
    except PipelineServiceError as exc:
        raise _pipeline_error_to_http(exc) from exc


@router.get(
    "/schema",
    response_model=dict,
    summary="Pipeline graph schema",
    description=(
        "Returns the full LangGraph pipeline topology: "
        "all nodes with their phases, all edges with routing conditions, "
        "entry point, terminal nodes, and parallel execution nodes."
    ),
)
def pipeline_schema() -> dict:
    return get_graph_schema()


@router.get(
    "/{experiment_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Poll experiment status",
    description="Lightweight endpoint for polling the status of a running or completed experiment.",
)
def get_experiment_status(
    experiment_id: str,
    service: PipelineService = Depends(get_pipeline_service),
) -> WorkflowStatusResponse:
    try:
        snap = service.get_experiment_status(experiment_id)
        return WorkflowStatusResponse(
            experiment_id=snap["experiment_id"],
            status=snap["status"],
            message=snap.get("error_message"),
        )
    except PipelineServiceError as exc:
        raise _pipeline_error_to_http(exc) from exc


@router.get(
    "/{experiment_id}",
    response_model=ExperimentRead,
    summary="Get full experiment detail",
)
def get_experiment(
    experiment_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentRead:
    try:
        exp = service.get_by_id(experiment_id)
        return ExperimentRead.model_validate(exp)
    except ExperimentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.get(
    "",
    response_model=list[ExperimentSummary],
    summary="List experiments for a dataset",
)
def list_experiments(
    dataset_id: str = Query(..., description="UUID of the dataset"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    service: ExperimentService = Depends(get_experiment_service),
) -> list[ExperimentSummary]:
    experiments = service.list_by_dataset(dataset_id, skip=skip, limit=limit)
    return [ExperimentSummary.model_validate(e) for e in experiments]
