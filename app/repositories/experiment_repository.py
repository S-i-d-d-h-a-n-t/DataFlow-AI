"""
Repository for Experiment persistence operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.models.experiment import Experiment
from app.enums.workflow_status import ExperimentStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


class ExperimentRepository:
    """CRUD operations for the Experiment ORM model."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Create ────────────────────────────────────────────────────────────────

    def create(self, experiment: Experiment) -> Experiment:
        self._db.add(experiment)
        self._db.commit()
        self._db.refresh(experiment)
        logger.info(f"Experiment created: id={experiment.id} name={experiment.name}")
        return experiment

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, experiment_id: str) -> Experiment | None:
        return self._db.get(Experiment, experiment_id)

    def list_by_dataset(
        self, dataset_id: str, skip: int = 0, limit: int = 100
    ) -> Sequence[Experiment]:
        stmt = (
            select(Experiment)
            .where(Experiment.dataset_id == dataset_id)
            .order_by(Experiment.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return self._db.execute(stmt).scalars().all()

    def list_by_status(self, status: ExperimentStatus) -> Sequence[Experiment]:
        stmt = select(Experiment).where(Experiment.status == status)
        return self._db.execute(stmt).scalars().all()

    def count_by_dataset(self, dataset_id: str) -> int:
        stmt = (
            select(func.count())
            .select_from(Experiment)
            .where(Experiment.dataset_id == dataset_id)
        )
        return self._db.execute(stmt).scalar_one()

    # ── Update ────────────────────────────────────────────────────────────────

    def update_status(
        self,
        experiment: Experiment,
        status: ExperimentStatus,
        error_message: str | None = None,
    ) -> Experiment:
        """Transition an experiment to a new status."""
        experiment.status = status
        if status == ExperimentStatus.RUNNING:
            experiment.started_at = datetime.utcnow()
        elif status.is_terminal():
            experiment.completed_at = datetime.utcnow()
        if error_message is not None:
            experiment.error_message = error_message
        self._db.commit()
        self._db.refresh(experiment)
        logger.info(f"Experiment {experiment.id} → status={status}")
        return experiment

    def save_metrics(
        self,
        experiment: Experiment,
        metrics: dict,
        best_model: str,
        report_path: str | None = None,
    ) -> Experiment:
        """Persist final metrics after training completes."""
        experiment.metrics = metrics
        experiment.best_model = best_model
        experiment.report_path = report_path
        self._db.commit()
        self._db.refresh(experiment)
        logger.info(f"Metrics saved for experiment {experiment.id}, best={best_model}")
        return experiment

    def update(self, experiment: Experiment, updates: dict) -> Experiment:
        for field, value in updates.items():
            if hasattr(experiment, field):
                setattr(experiment, field, value)
        self._db.commit()
        self._db.refresh(experiment)
        return experiment

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, experiment: Experiment) -> None:
        self._db.delete(experiment)
        self._db.commit()
        logger.info(f"Experiment deleted: id={experiment.id}")
