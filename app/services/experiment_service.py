"""
Experiment Service — business logic for experiment lifecycle management.

Responsibilities:
  - Create and persist Experiment records.
  - Transition experiment status (pending → running → completed/failed).
  - Retrieve experiments by id or dataset.
  - Provide the ExperimentRepository as the single DB access point.

Rules:
  - No SQLAlchemy imports — all DB access goes through ExperimentRepository.
  - No FastAPI imports — framework-agnostic.
  - Raises domain-level exceptions that the API layer translates to HTTP errors.
"""

from __future__ import annotations

from typing import Sequence

from app.models.experiment import Experiment
from app.repositories.experiment_repository import ExperimentRepository
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.experiment import ExperimentCreate
from app.enums.workflow_status import ExperimentStatus
from app.enums.task_type import TaskType
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Domain exceptions ─────────────────────────────────────────────────────────

class ExperimentServiceError(Exception):
    """Base class for experiment service errors."""


class ExperimentNotFoundError(ExperimentServiceError):
    """Raised when a requested experiment does not exist."""


class DatasetNotFoundError(ExperimentServiceError):
    """Raised when the referenced dataset does not exist."""


class InvalidTransitionError(ExperimentServiceError):
    """Raised when a status transition is not allowed."""


# ── Service ───────────────────────────────────────────────────────────────────

class ExperimentService:
    """Orchestrates experiment lifecycle operations."""

    def __init__(
        self,
        experiment_repo: ExperimentRepository,
        dataset_repo: DatasetRepository,
    ) -> None:
        self._exp_repo = experiment_repo
        self._ds_repo = dataset_repo

    # ── Create ────────────────────────────────────────────────────────────────

    def create(self, payload: ExperimentCreate) -> Experiment:
        """
        Validate the referenced dataset exists, then create the experiment.
        Status starts as PENDING.
        """
        dataset = self._ds_repo.get_by_id(payload.dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(
                f"Dataset '{payload.dataset_id}' not found. "
                "Upload a dataset before creating an experiment."
            )

        experiment = Experiment(
            dataset_id=payload.dataset_id,
            name=payload.name,
            target_column=payload.target_column,
            task_type=payload.task_type,
            pipeline_config=payload.pipeline_config,
            status=ExperimentStatus.PENDING,
        )
        return self._exp_repo.create(experiment)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, experiment_id: str) -> Experiment:
        """Return an Experiment or raise ExperimentNotFoundError."""
        exp = self._exp_repo.get_by_id(experiment_id)
        if exp is None:
            raise ExperimentNotFoundError(
                f"Experiment '{experiment_id}' not found."
            )
        return exp

    def list_by_dataset(
        self, dataset_id: str, skip: int = 0, limit: int = 100
    ) -> Sequence[Experiment]:
        return self._exp_repo.list_by_dataset(dataset_id, skip=skip, limit=limit)

    # ── Status transitions ────────────────────────────────────────────────────

    def mark_running(self, experiment_id: str) -> Experiment:
        exp = self.get_by_id(experiment_id)
        if exp.status != ExperimentStatus.PENDING:
            raise InvalidTransitionError(
                f"Cannot start experiment in status '{exp.status}'. "
                "Only PENDING experiments can be started."
            )
        return self._exp_repo.update_status(exp, ExperimentStatus.RUNNING)

    def mark_completed(
        self,
        experiment_id: str,
        metrics: dict,
        best_model: str,
        report_path: str | None = None,
    ) -> Experiment:
        exp = self.get_by_id(experiment_id)
        exp = self._exp_repo.save_metrics(exp, metrics, best_model, report_path)
        return self._exp_repo.update_status(exp, ExperimentStatus.COMPLETED)

    def mark_failed(self, experiment_id: str, error_message: str) -> Experiment:
        exp = self.get_by_id(experiment_id)
        return self._exp_repo.update_status(
            exp, ExperimentStatus.FAILED, error_message=error_message
        )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, experiment_id: str) -> None:
        exp = self.get_by_id(experiment_id)
        self._exp_repo.delete(exp)
        logger.info(f"Experiment {experiment_id} deleted.")
