"""
ML task type enumeration.
Shared across agents, services, schemas, and ORM models.
"""

import enum


class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"  # unsupervised — future support

    def is_supervised(self) -> bool:
        return self in (TaskType.CLASSIFICATION, TaskType.REGRESSION)
