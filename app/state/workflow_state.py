"""
LangGraph shared workflow state.

WorkflowState is the single TypedDict that flows through every node in the
LangGraph StateGraph.  Each agent reads from it and writes its outputs back
into it — no agent communicates with another directly.

Design rules:
  - All fields are Optional so nodes can be added/removed without breaking others.
  - Agents append to list fields (e.g. node_results) rather than overwriting.
  - Heavy objects (DataFrames) are stored as file paths, not in-memory blobs.
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus


class NodeResult(TypedDict, total=False):
    """Snapshot written by each agent after it finishes."""

    node_name: str
    status: WorkflowNodeStatus
    duration_seconds: float
    output_summary: dict[str, Any]
    error: str


class WorkflowState(TypedDict, total=False):
    """
    Shared state object passed between all LangGraph nodes.

    Lifecycle:
        API layer populates the initial fields (experiment_id … pipeline_config).
        Each agent reads what it needs and writes its outputs back.
        The final node persists results to the database.
    """

    # ── Identifiers ───────────────────────────────────────────────────────────
    experiment_id: str
    dataset_id: str

    # ── Input configuration ───────────────────────────────────────────────────
    dataset_path: str          # Absolute path to the raw CSV on disk
    target_column: str
    task_type: TaskType
    pipeline_config: dict[str, Any]

    # ── Planner outputs ───────────────────────────────────────────────────────
    plan: dict[str, Any]       # Ordered list of steps + agent assignments

    # ── EDA outputs ───────────────────────────────────────────────────────────
    eda_summary: dict[str, Any]
    chart_paths: list[str]

    # ── Cleaning outputs ──────────────────────────────────────────────────────
    cleaned_dataset_path: str
    cleaning_report: dict[str, Any]

    # ── Feature engineering outputs ───────────────────────────────────────────
    feature_dataset_path: str
    feature_report: dict[str, Any]
    selected_features: list[str]

    # ── Training outputs (populated in parallel) ──────────────────────────────
    trained_model_paths: dict[str, str]   # {"random_forest": "/outputs/models/rf.pkl", …}
    training_metrics: dict[str, Any]      # Per-model raw metrics

    # ── Evaluation outputs ────────────────────────────────────────────────────
    evaluation_results: dict[str, Any]    # Ranked model comparison
    best_model_name: str
    best_model_path: str

    # ── Report outputs ────────────────────────────────────────────────────────
    report_markdown_path: str
    report_html_path: str

    # ── Execution metadata ────────────────────────────────────────────────────
    node_results: list[NodeResult]        # Appended by each agent
    errors: list[str]                     # Non-fatal warnings / errors
