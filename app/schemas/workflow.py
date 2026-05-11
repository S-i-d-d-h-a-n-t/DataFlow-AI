"""
Pydantic schemas for workflow execution requests and responses.
These are the API-facing contracts for triggering and monitoring LangGraph runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from app.enums.workflow_status import ExperimentStatus, WorkflowNodeStatus
from app.enums.task_type import TaskType


class WorkflowRunRequest(BaseModel):
    """Request body to trigger a full pipeline run."""

    dataset_id: str = Field(..., description="UUID of the uploaded dataset")
    target_column: str = Field(..., description="Column to predict")
    task_type: TaskType
    experiment_name: str = Field(
        ..., min_length=1, max_length=255, examples=["full_pipeline_v1"]
    )
    pipeline_config: dict[str, Any] | None = Field(
        None,
        description="Optional agent-level overrides, e.g. {'models': ['xgboost']}",
    )


class NodeResult(BaseModel):
    """Result snapshot for a single LangGraph node."""

    node_name: str
    status: WorkflowNodeStatus
    duration_seconds: float | None = None
    output_summary: dict[str, Any] | None = None
    error: str | None = None


class WorkflowRunResponse(BaseModel):
    """Response returned after a workflow run completes or fails."""

    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    status: ExperimentStatus
    best_model: str | None
    metrics: dict[str, Any] | None
    report_path: str | None
    node_results: list[NodeResult] = Field(default_factory=list)
    started_at: datetime | None
    completed_at: datetime | None


class WorkflowStatusResponse(BaseModel):
    """Lightweight polling response for async workflow runs."""

    experiment_id: str
    status: ExperimentStatus
    message: str | None = None
