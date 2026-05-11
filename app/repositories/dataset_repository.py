"""
Repository for Dataset persistence operations.

Rules:
  - All SQLAlchemy queries live here — nowhere else.
  - Methods accept and return ORM model instances.
  - Services call repositories; repositories never call services.
"""

from __future__ import annotations

from typing import Sequence

from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.models.dataset import Dataset
from app.core.logging import get_logger

logger = get_logger(__name__)


class DatasetRepository:
    """CRUD operations for the Dataset ORM model."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Create ────────────────────────────────────────────────────────────────

    def create(self, dataset: Dataset) -> Dataset:
        """Persist a new Dataset row and return it with its generated id."""
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        logger.info(f"Dataset created: id={dataset.id} name={dataset.name}")
        return dataset

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, dataset_id: str) -> Dataset | None:
        """Return a single Dataset by primary key, or None."""
        return self._db.get(Dataset, dataset_id)

    def get_by_name(self, name: str) -> Dataset | None:
        """Return the first Dataset matching the given name, or None."""
        stmt = select(Dataset).where(Dataset.name == name)
        return self._db.execute(stmt).scalars().first()

    def list_all(self, skip: int = 0, limit: int = 100) -> Sequence[Dataset]:
        """Return a paginated list of all datasets ordered by creation date."""
        stmt = (
            select(Dataset)
            .order_by(Dataset.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return self._db.execute(stmt).scalars().all()

    def count(self) -> int:
        """Return total number of datasets."""
        stmt = select(func.count()).select_from(Dataset)
        return self._db.execute(stmt).scalar_one()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dataset: Dataset, updates: dict) -> Dataset:
        """Apply a dict of field updates to an existing Dataset row."""
        for field, value in updates.items():
            if hasattr(dataset, field):
                setattr(dataset, field, value)
        self._db.commit()
        self._db.refresh(dataset)
        logger.info(f"Dataset updated: id={dataset.id} fields={list(updates.keys())}")
        return dataset

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, dataset: Dataset) -> None:
        """Hard-delete a Dataset row."""
        self._db.delete(dataset)
        self._db.commit()
        logger.info(f"Dataset deleted: id={dataset.id}")
