"""
Pipeline Service — orchestrates a full end-to-end ML pipeline run.

Responsibilities:
  1. Accept a WorkflowRunRequest and resolve the dataset file path.
  2. Create an Experiment record (PENDING).
  3. Build the initial WorkflowState.
  4. Invoke the compiled LangGraph pipeline (with timeout guard).
  5. Persist the final status, metrics, and report path back to the DB.
  6. Return a WorkflowRunResponse to the API layer.

Rules:
  - This service is the only place that imports the LangGraph graph.
  - It coordinates ExperimentService and DatasetService but never touches
    repositories directly.
  - Any unhandled exception during graph execution marks the experiment FAILED
    and re-raises as PipelineServiceError so the API layer can return 500.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.schemas.workflow import WorkflowRunRequest, WorkflowRunResponse, NodeResult
from app.schemas.experiment import ExperimentCreate
from app.services.experiment_service import ExperimentService, ExperimentServiceError
from app.services.dataset_service import DatasetService, DatasetNotFoundError
from app.repositories.experiment_repository import ExperimentRepository
from app.repositories.dataset_repository import DatasetRepository
from app.enums.workflow_status import WorkflowNodeStatus
from app.state.workflow_state import WorkflowState
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Domain exception ──────────────────────────────────────────────────────────

class PipelineServiceError(Exception):
    """Raised when the pipeline cannot be started or completed."""


# ── Service ───────────────────────────────────────────────────────────────────

class PipelineService:
    """Coordinates dataset resolution, experiment lifecycle, and graph execution."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._exp_service = ExperimentService(
            experiment_repo=ExperimentRepository(db),
            dataset_repo=DatasetRepository(db),
        )
        self._ds_service = DatasetService(repo=DatasetRepository(db))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, request: WorkflowRunRequest) -> WorkflowRunResponse:
        """
        Execute the full ML pipeline synchronously and return the result.

        The experiment is always persisted (COMPLETED or FAILED) before
        this method returns, even if the graph raises an unexpected exception.
        """
        # ── 1. Resolve dataset ────────────────────────────────────────────────
        try:
            dataset = self._ds_service.get_by_id(request.dataset_id)
        except DatasetNotFoundError as exc:
            raise PipelineServiceError(str(exc)) from exc

        # ── 2. Create experiment ──────────────────────────────────────────────
        exp_payload = ExperimentCreate(
            dataset_id=request.dataset_id,
            name=request.experiment_name,
            target_column=request.target_column,
            task_type=request.task_type,
            pipeline_config=request.pipeline_config,
        )
        try:
            experiment = self._exp_service.create(exp_payload)
        except ExperimentServiceError as exc:
            raise PipelineServiceError(str(exc)) from exc

        logger.info(
            f"[pipeline] Experiment created: id={experiment.id} "
            f"dataset={request.dataset_id} target={request.target_column}"
        )

        # ── 3. Mark running ───────────────────────────────────────────────────
        self._exp_service.mark_running(experiment.id)

        # ── 4. Build initial WorkflowState ────────────────────────────────────
        initial_state: WorkflowState = {
            "experiment_id": experiment.id,
            "dataset_id": request.dataset_id,
            "dataset_path": dataset.file_path,
            "target_column": request.target_column,
            "task_type": request.task_type,
            "pipeline_config": request.pipeline_config or {},
            "node_results": [],
            "errors": [],
        }

        # ── 5. Run the graph (always persist outcome) ─────────────────────────
        final_state: dict[str, Any] = {}
        graph_exception: Exception | None = None

        try:
            final_state = self._invoke_graph(initial_state)
        except Exception as exc:
            graph_exception = exc
            logger.exception(
                f"[pipeline] Unhandled exception during graph execution "
                f"for experiment {experiment.id}: {exc}"
            )

        # ── 6. Persist outcome ────────────────────────────────────────────────
        if graph_exception is not None:
            # Catastrophic failure — mark failed and re-raise
            error_msg = f"Graph execution failed: {graph_exception}"
            self._exp_service.mark_failed(experiment.id, error_msg)
            raise PipelineServiceError(error_msg) from graph_exception

        errors: list[str] = final_state.get("errors") or []
        node_results_raw: list[dict] = final_state.get("node_results") or []

        if errors:
            error_msg = "; ".join(errors)
            experiment = self._exp_service.mark_failed(experiment.id, error_msg)
            logger.error(
                f"[pipeline] Experiment {experiment.id} FAILED: {error_msg}"
            )
        else:
            best_model = final_state.get("best_model_name") or "none"
            metrics = final_state.get("evaluation_results") or {}
            report_path = final_state.get("report_markdown_path") or None
            experiment = self._exp_service.mark_completed(
                experiment.id,
                metrics=metrics,
                best_model=best_model,
                report_path=report_path,
            )
            logger.info(
                f"[pipeline] Experiment {experiment.id} COMPLETED "
                f"best_model={best_model}"
            )

        # ── 7. Build and return response ──────────────────────────────────────
        node_results = [
            NodeResult(
                node_name=nr.get("node_name", "unknown"),
                status=nr.get("status", WorkflowNodeStatus.DONE),
                duration_seconds=nr.get("duration_seconds"),
                output_summary=nr.get("output_summary"),
                error=nr.get("error"),
            )
            for nr in node_results_raw
        ]

        return WorkflowRunResponse(
            experiment_id=experiment.id,
            status=experiment.status,
            best_model=experiment.best_model,
            metrics=experiment.metrics,
            report_path=experiment.report_path,
            node_results=node_results,
            started_at=experiment.started_at,
            completed_at=experiment.completed_at,
        )

    def get_experiment_status(self, experiment_id: str) -> dict[str, Any]:
        """
        Return a lightweight status snapshot for an experiment.
        Used by the polling endpoint GET /api/v1/workflow/{id}/status.
        """
        from app.services.experiment_service import ExperimentNotFoundError
        try:
            exp = self._exp_service.get_by_id(experiment_id)
        except ExperimentNotFoundError as exc:
            raise PipelineServiceError(str(exc)) from exc

        return {
            "experiment_id": exp.id,
            "status": exp.status.value,
            "best_model": exp.best_model,
            "report_path": exp.report_path,
            "started_at": exp.started_at.isoformat() if exp.started_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
            "error_message": exp.error_message,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _invoke_graph(initial_state: WorkflowState) -> dict[str, Any]:
        """
        Import and invoke the compiled LangGraph pipeline.
        Isolated in a static method so it can be mocked in tests.
        """
        from app.workflows.graph import pipeline_graph
        logger.info("[pipeline] Invoking LangGraph pipeline...")
        result = pipeline_graph.invoke(initial_state)
        logger.info("[pipeline] LangGraph pipeline finished.")
        return result
