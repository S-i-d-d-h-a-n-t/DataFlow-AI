"""
Pydantic schemas for the Experiment resource.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from app.enums.workflow_status import ExperimentStatus
from app.enums.task_type import TaskType


class ExperimentCreate(BaseModel):
    """Payload required to kick off a new experiment."""

    dataset_id: str = Field(..., description="UUID of the target dataset")
    name: str = Field(..., min_length=1, max_length=255, examples=["baseline_run_v1"])
    target_column: str = Field(..., description="Column the model should predict")
    task_type: TaskType = Field(..., description="classification | regression")
    pipeline_config: dict[str, Any] | None = Field(
        None, description="Optional overrides for the pipeline (e.g. model list)"
    )


class ExperimentRead(BaseModel):
    """Full experiment representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    name: str
    status: ExperimentStatus
    target_column: str | None
    task_type: TaskType | None
    pipeline_config: dict[str, Any] | None
    metrics: dict[str, Any] | None
    best_model: str | None
    report_path: str | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class ExperimentSummary(BaseModel):
    """Lightweight projection used in list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: ExperimentStatus
    best_model: str | None
    created_at: datetime


class ExperimentStatusUpdate(BaseModel):
    """Internal schema used by services to patch experiment status."""

    status: ExperimentStatus
    error_message: str | None = None
