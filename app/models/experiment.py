"""
ORM model for ML experiments.
Tracks each pipeline run — its configuration, status, and resulting metrics.

NOTE: Business enums live in app/enums/ — never define them here.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Text, JSON, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.enums.workflow_status import ExperimentStatus
from app.enums.task_type import TaskType


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    dataset_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ExperimentStatus] = mapped_column(
        SAEnum(ExperimentStatus), nullable=False, default=ExperimentStatus.PENDING
    )
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    task_type: Mapped[TaskType | None] = mapped_column(
        SAEnum(TaskType), nullable=True
    )
    pipeline_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    best_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationship back to dataset (lazy loaded)
    dataset: Mapped["Dataset"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Dataset", foreign_keys=[dataset_id], lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} name={self.name} status={self.status}>"
