"""
Planner Agent — the first node in the LangGraph pipeline.

Responsibilities:
  1. Load the dataset from disk and run structural analysis.
  2. Confirm or infer the task type (classification / regression).
  3. Validate that the target column exists and is suitable.
  4. Produce a deterministic, ordered execution plan that every downstream
     agent can read from WorkflowState["plan"].
  5. Write a NodeResult entry so the orchestrator can track its outcome.

Design rules:
  - Pure function signature: (WorkflowState) → WorkflowState patch dict.
  - No database access — the orchestrator handles DB persistence.
  - No LangGraph imports — the agent is just a callable; the graph wires it.
  - Raises PlannerError on unrecoverable problems so the graph can route
    to a failure node.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.utils.dataframe_utils import (
    infer_column_dtypes,
    missing_value_report,
    cardinality_report,
    detect_target_type,
    column_stats,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Models available per task type — used to build the plan
_CLASSIFICATION_MODELS = ["random_forest", "logistic_regression", "xgboost"]
_REGRESSION_MODELS = ["random_forest", "linear_regression", "xgboost"]

# Thresholds for cleaning recommendations
_HIGH_MISSING_PCT = 50.0   # columns above this will be flagged for dropping
_LOW_MISSING_PCT = 0.0     # any missing triggers imputation recommendation


# ── Domain exception ──────────────────────────────────────────────────────────

class PlannerError(Exception):
    """Raised when the Planner cannot produce a valid plan."""


# ── Agent callable ────────────────────────────────────────────────────────────

def planner_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the Planner Agent.

    Reads from state:
        dataset_path, target_column, task_type, pipeline_config

    Writes to state:
        plan, node_results (appended)

    Returns a dict that LangGraph merges into WorkflowState.
    """
    start = time.perf_counter()
    node_name = "planner"
    logger.info(f"[{node_name}] Starting — dataset={state.get('dataset_path')}")

    try:
        result = _run_planner(state)
        duration = round(time.perf_counter() - start, 3)

        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "task_type": result["plan"]["task_type"],
                "target_column": result["plan"]["target_column"],
                "num_steps": len(result["plan"]["steps"]),
                "models_selected": result["plan"]["models"],
                "dataset_shape": result["plan"]["dataset_shape"],
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"task={result['plan']['task_type']} "
            f"models={result['plan']['models']}"
        )

    except PlannerError as exc:
        duration = round(time.perf_counter() - start, 3)
        node_result = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.ERROR,
            "duration_seconds": duration,
            "error": str(exc),
        }
        logger.error(f"[{node_name}] Failed: {exc}")
        # Propagate so the graph can route to the error handler
        existing = list(state.get("node_results") or [])
        existing.append(node_result)
        return {"node_results": existing, "errors": [str(exc)]}

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core planning logic ───────────────────────────────────────────────────────

def _run_planner(state: WorkflowState) -> dict[str, Any]:
    """
    Pure planning logic — separated from the node wrapper for testability.

    Returns a dict ready to be merged into WorkflowState.
    """
    dataset_path = state.get("dataset_path", "")
    target_column = state.get("target_column", "")
    requested_task_type: TaskType | None = state.get("task_type")
    pipeline_config: dict[str, Any] = state.get("pipeline_config") or {}

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    path = Path(dataset_path)
    if not path.exists():
        raise PlannerError(f"Dataset file not found: '{dataset_path}'")

    df = _load_csv(path)
    logger.debug(f"[planner] Loaded dataset shape={df.shape}")

    # ── 2. Validate target column ─────────────────────────────────────────────
    if not target_column:
        raise PlannerError("target_column is required but was not provided.")
    if target_column not in df.columns:
        available = list(df.columns[:10])
        raise PlannerError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns (first 10): {available}"
        )

    # ── 3. Infer or confirm task type ─────────────────────────────────────────
    inferred = detect_target_type(df[target_column])
    task_type: TaskType = requested_task_type or TaskType(inferred)

    if requested_task_type and requested_task_type.value != inferred:
        logger.warning(
            f"[planner] Requested task_type={requested_task_type.value} but "
            f"heuristic suggests '{inferred}' for column '{target_column}'. "
            f"Honouring the requested type."
        )

    # ── 4. Structural analysis ────────────────────────────────────────────────
    dtypes = infer_column_dtypes(df)
    missing = missing_value_report(df)
    cardinality = cardinality_report(df)
    stats = column_stats(df)

    feature_columns = [c for c in df.columns if c != target_column]
    numeric_features = [c for c in feature_columns if dtypes.get(c) == "numeric"]
    categorical_features = [c for c in feature_columns if dtypes.get(c) == "categorical"]
    high_missing_cols = [
        c for c, v in missing.items() if v["pct"] > _HIGH_MISSING_PCT
    ]

    # ── 5. Select models ──────────────────────────────────────────────────────
    if task_type == TaskType.CLASSIFICATION:
        default_models = _CLASSIFICATION_MODELS
    else:
        default_models = _REGRESSION_MODELS

    # Allow pipeline_config to override the model list
    models: list[str] = pipeline_config.get("models", default_models)

    # ── 6. Build execution plan ───────────────────────────────────────────────
    steps = _build_steps(
        task_type=task_type,
        has_missing=bool(missing),
        has_categoricals=bool(categorical_features),
        models=models,
    )

    plan: dict[str, Any] = {
        "task_type": task_type.value,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": models,
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "data_quality": {
            "missing_columns": list(missing.keys()),
            "high_missing_columns": high_missing_cols,
            "column_dtypes": dtypes,
            "cardinality": cardinality,
        },
        "column_stats": stats,
        "steps": steps,
        "pipeline_config": pipeline_config,
    }

    return {"plan": plan, "task_type": task_type}


def _build_steps(
    task_type: TaskType,
    has_missing: bool,
    has_categoricals: bool,
    models: list[str],
) -> list[dict[str, Any]]:
    """
    Build the ordered list of pipeline steps.
    Each step maps to a downstream agent node.
    """
    steps: list[dict[str, Any]] = [
        {
            "order": 1,
            "agent": "eda",
            "description": "Exploratory data analysis — statistics and charts",
            "required": True,
        },
        {
            "order": 2,
            "agent": "cleaning",
            "description": "Data cleaning — imputation, outlier handling, type fixes",
            "required": True,
            "config": {
                "impute_missing": has_missing,
                "encode_categoricals": has_categoricals,
            },
        },
        {
            "order": 3,
            "agent": "feature_engineering",
            "description": "Feature engineering — scaling, encoding, selection",
            "required": True,
            "config": {
                "encode_categoricals": has_categoricals,
                "task_type": task_type.value,
            },
        },
        {
            "order": 4,
            "agent": "training",
            "description": f"Parallel model training — {', '.join(models)}",
            "required": True,
            "config": {
                "models": models,
                "task_type": task_type.value,
                "parallel": True,
            },
        },
        {
            "order": 5,
            "agent": "evaluation",
            "description": "Model evaluation and ranking",
            "required": True,
            "config": {"task_type": task_type.value},
        },
        {
            "order": 6,
            "agent": "report",
            "description": "Markdown and HTML report generation",
            "required": True,
        },
    ]
    return steps


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise PlannerError(f"Could not read CSV file: {path}")
