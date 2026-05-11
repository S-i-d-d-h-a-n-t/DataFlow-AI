"""
ML metrics computation helpers.
Wraps scikit-learn metrics with consistent return shapes used across agents.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from app.enums.task_type import TaskType


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    task_type: TaskType,
    y_prob: Any | None = None,
) -> dict[str, float]:
    """
    Compute the standard metric set for a given task type.

    Args:
        y_true:    Ground-truth labels / values.
        y_pred:    Model predictions.
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION.
        y_prob:    Predicted probabilities (classification only, for AUC).

    Returns:
        Dict mapping metric name → rounded float value.
    """
    if task_type == TaskType.CLASSIFICATION:
        return _classification_metrics(y_true, y_pred, y_prob)
    return _regression_metrics(y_true, y_pred)


def _classification_metrics(
    y_true: Any, y_pred: Any, y_prob: Any | None
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "precision_weighted": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "recall_weighted": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            multi_class = "ovr" if n_classes > 2 else "raise"
            auc = roc_auc_score(
                y_true, y_prob, multi_class=multi_class if n_classes > 2 else None
            )
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            pass  # AUC not computable for all label configurations
    return metrics


def _regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def rank_models(
    metrics_by_model: dict[str, dict[str, float]],
    task_type: TaskType,
) -> list[dict[str, Any]]:
    """
    Rank models by their primary metric.
    Classification → f1_weighted (higher is better).
    Regression     → rmse (lower is better).

    Returns a list of dicts sorted best-first.
    """
    primary = "f1_weighted" if task_type == TaskType.CLASSIFICATION else "rmse"
    reverse = task_type == TaskType.CLASSIFICATION  # True = higher is better

    ranked = sorted(
        [
            {"model": name, "metrics": m, "primary_score": m.get(primary, 0.0)}
            for name, m in metrics_by_model.items()
        ],
        key=lambda x: x["primary_score"],
        reverse=reverse,
    )
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1
    return ranked
