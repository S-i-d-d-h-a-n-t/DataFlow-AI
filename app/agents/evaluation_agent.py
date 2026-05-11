"""
Evaluation Agent — sixth node in the LangGraph pipeline.

Responsibilities:
  1. Consume training_metrics from WorkflowState (produced by the Training Agent).
  2. Rank all trained models by their primary metric:
       - Classification → f1_weighted  (higher is better)
       - Regression     → rmse         (lower is better)
  3. Identify the best model and its artifact path.
  4. Compute a per-metric comparison table across all models.
  5. Optionally load each model from disk and re-evaluate on the feature
     dataset to produce a cross-validated leaderboard (configurable).
  6. Write evaluation_results, best_model_name, best_model_path back into
     WorkflowState.
  7. Append a NodeResult.

Design rules:
  - Pure function signature: (WorkflowState) → dict patch.
  - No database access.
  - No LangGraph imports.
  - Raises EvaluationError on unrecoverable problems.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.utils.metrics import rank_models
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Primary metric per task type ──────────────────────────────────────────────
_PRIMARY_METRIC: dict[str, str] = {
    "classification": "f1_weighted",
    "regression": "rmse",
}

# Higher-is-better flag per task type
_HIGHER_IS_BETTER: dict[str, bool] = {
    "classification": True,
    "regression": False,
}


# ── Domain exception ──────────────────────────────────────────────────────────

class EvaluationError(Exception):
    """Raised when the Evaluation Agent cannot complete its work."""


# ── Agent callable ────────────────────────────────────────────────────────────

def evaluation_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the Evaluation Agent.

    Reads from state:
        training_metrics, trained_model_paths, task_type,
        experiment_id, pipeline_config

    Writes to state:
        evaluation_results, best_model_name, best_model_path,
        node_results (appended)
    """
    start = time.perf_counter()
    node_name = "evaluation"
    logger.info(f"[{node_name}] Starting — evaluating trained models")

    try:
        result = _run_evaluation(state)
        duration = round(time.perf_counter() - start, 3)

        ev = result["evaluation_results"]
        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "models_evaluated": ev["models_evaluated"],
                "best_model": result["best_model_name"],
                "primary_metric": ev["primary_metric"],
                "best_score": ev["best_score"],
                "ranking": [
                    {"rank": r["rank"], "model": r["model"],
                     "score": r["primary_score"]}
                    for r in ev["ranked_models"]
                ],
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"best={result['best_model_name']} "
            f"({ev['primary_metric']}={ev['best_score']})"
        )

    except EvaluationError as exc:
        duration = round(time.perf_counter() - start, 3)
        node_result = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.ERROR,
            "duration_seconds": duration,
            "error": str(exc),
        }
        logger.error(f"[{node_name}] Failed: {exc}")
        existing = list(state.get("node_results") or [])
        existing.append(node_result)
        errors = list(state.get("errors") or [])
        errors.append(str(exc))
        return {
            "node_results": existing,
            "errors": errors,
            "evaluation_results": {},
            "best_model_name": "",
            "best_model_path": "",
        }

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core evaluation logic ─────────────────────────────────────────────────────

def _run_evaluation(state: WorkflowState) -> dict[str, Any]:
    """
    Pure evaluation logic — separated from the node wrapper for testability.
    Returns a dict ready to be merged into WorkflowState.
    """
    training_metrics: dict[str, Any] = state.get("training_metrics") or {}
    trained_model_paths: dict[str, str] = state.get("trained_model_paths") or {}
    task_type: TaskType = state.get("task_type") or TaskType.CLASSIFICATION

    # ── 1. Validate inputs ────────────────────────────────────────────────────
    if not training_metrics:
        raise EvaluationError(
            "No training_metrics found in state. "
            "Ensure the Training Agent ran successfully before Evaluation."
        )
    if not trained_model_paths:
        raise EvaluationError(
            "No trained_model_paths found in state. "
            "Ensure the Training Agent ran successfully before Evaluation."
        )

    # ── 2. Strip non-metric keys from training_metrics ────────────────────────
    # training_metrics may contain bookkeeping keys (n_train, n_test, duration)
    # that are not ML metrics. We keep them in the full record but exclude them
    # from ranking.
    _BOOKKEEPING_KEYS = {"train_duration_seconds", "n_train", "n_test"}

    pure_metrics: dict[str, dict[str, float]] = {
        model: {k: v for k, v in m.items() if k not in _BOOKKEEPING_KEYS}
        for model, m in training_metrics.items()
        if isinstance(m, dict)
    }

    # ── 3. Rank models ────────────────────────────────────────────────────────
    ranked = rank_models(pure_metrics, task_type)

    if not ranked:
        raise EvaluationError("rank_models returned an empty list.")

    # ── 4. Identify best model ────────────────────────────────────────────────
    best_entry = ranked[0]   # rank_models returns best-first
    best_model_name: str = best_entry["model"]
    best_model_path: str = trained_model_paths.get(best_model_name, "")

    primary_metric = _PRIMARY_METRIC.get(task_type.value, "f1_weighted")
    best_score: float = best_entry["primary_score"]

    logger.info(
        f"[evaluation] Best model: {best_model_name} "
        f"({primary_metric}={best_score})"
    )

    # ── 5. Build per-metric comparison table ──────────────────────────────────
    # Collect all metric names across all models
    all_metric_keys: list[str] = []
    for m in pure_metrics.values():
        for k in m:
            if k not in all_metric_keys:
                all_metric_keys.append(k)

    comparison_table: list[dict[str, Any]] = []
    for entry in ranked:
        model_name = entry["model"]
        row: dict[str, Any] = {
            "rank": entry["rank"],
            "model": model_name,
            "primary_score": entry["primary_score"],
        }
        for key in all_metric_keys:
            row[key] = pure_metrics[model_name].get(key)
        # Add bookkeeping info back for transparency
        bk = training_metrics.get(model_name, {})
        row["train_duration_seconds"] = bk.get("train_duration_seconds")
        row["n_train"] = bk.get("n_train")
        row["n_test"] = bk.get("n_test")
        comparison_table.append(row)

    # ── 6. Verify best model artifact exists on disk ──────────────────────────
    best_model_exists = Path(best_model_path).exists() if best_model_path else False
    if not best_model_exists:
        logger.warning(
            f"[evaluation] Best model artifact not found on disk: "
            f"'{best_model_path}'"
        )

    # ── 7. Build evaluation results ───────────────────────────────────────────
    evaluation_results: dict[str, Any] = {
        "models_evaluated": list(training_metrics.keys()),
        "task_type": task_type.value,
        "primary_metric": primary_metric,
        "higher_is_better": _HIGHER_IS_BETTER.get(task_type.value, True),
        "best_model": best_model_name,
        "best_score": best_score,
        "ranked_models": ranked,
        "comparison_table": comparison_table,
        "best_model_artifact_exists": best_model_exists,
    }

    return {
        "evaluation_results": evaluation_results,
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
    }
