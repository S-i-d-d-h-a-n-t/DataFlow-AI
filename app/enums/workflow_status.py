"""
Status enumerations for experiments and workflow graph nodes.
Centralised here so every layer (ORM, schema, agent) imports from one place.
"""

import enum


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Return True if no further transitions are expected."""
        return self in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)


class WorkflowNodeStatus(str, enum.Enum):
    """Fine-grained status for individual LangGraph nodes."""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"
