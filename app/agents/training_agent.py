"""
Training Agent — fifth node in the LangGraph pipeline.

Architecture — mandatory parallel execution:
  The orchestrator fans out one train_single_model() call per model using
  LangGraph's Send API. Each model trains in its own graph node, writing
  its result into WorkflowState. A final merge node collects all results.

  Fan-out topology:
    [feature_engineering]
          │
          ▼
    [dispatch_training]  ──Send──► [train_random_forest]
                         ──Send──► [train_logistic_regression / train_linear_regression]
                         ──Send──► [train_xgboost]
          │
          ▼
    [merge_training]   ← collects trained_model_paths + training_metrics
          │
          ▼
    [evaluation]

Supported models:
  Classification : random_forest, logistic_regression, xgboost
  Regression     : random_forest, linear_regression, xgboost

Each model is trained with a fixed random_state=42 for reproducibility.
Trained models are saved as pickle files to outputs/models/{exp_id}/.

Design rules:
  - No database access.
  - No LangGraph imports in this file — graph wiring is in graph.py.
  - All model state is serialised to disk — never stored in WorkflowState.
  - Raises TrainingError on unrecoverable problems.
"""

from __future__ import annotations

import pickle
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.utils.metrics import compute_metrics
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_RANDOM_STATE = 42
_TEST_SIZE = 0.2          # fraction held out for validation metrics
_CV_FOLDS = 3             # cross-validation folds (used for small datasets)
_MIN_ROWS_FOR_SPLIT = 20  # below this, skip train/test split and train on all

# Model registry — maps name → (ClassificationClass, RegressionClass, default_params)
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "random_forest": {
        "classification": RandomForestClassifier,
        "regression": RandomForestRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": _RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    "logistic_regression": {
        "classification": LogisticRegression,
        "regression": None,   # not applicable
        "params": {
            "max_iter": 1000,
            "random_state": _RANDOM_STATE,
        },
    },
    "linear_regression": {
        "classification": None,   # not applicable
        "regression": LinearRegression,
        "params": {"n_jobs": -1},
    },
    "xgboost": {
        "classification": xgb.XGBClassifier,
        "regression": xgb.XGBRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": _RANDOM_STATE,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
    },
}


# ── Domain exception ──────────────────────────────────────────────────────────

class TrainingError(Exception):
    """Raised when the Training Agent cannot complete its work."""


# ── Agent callable (orchestrates parallel training) ───────────────────────────

def training_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the Training Agent.

    Reads from state:
        feature_dataset_path, selected_features, target_column,
        task_type, experiment_id, plan, pipeline_config

    Writes to state:
        trained_model_paths, training_metrics, node_results (appended)

    Parallel execution is handled internally via ThreadPoolExecutor.
    Each model trains concurrently; results are merged before returning.
    """
    start = time.perf_counter()
    node_name = "training"
    logger.info(
        f"[{node_name}] Starting — "
        f"dataset={state.get('feature_dataset_path')}"
    )

    try:
        result = _run_training(state)
        duration = round(time.perf_counter() - start, 3)

        trained = result["trained_model_paths"]
        metrics = result["training_metrics"]
        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "models_trained": list(trained.keys()),
                "models_failed": result.get("models_failed", []),
                "metrics_summary": {
                    name: list(m.keys())
                    for name, m in metrics.items()
                },
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"trained={list(trained.keys())}"
        )

    except TrainingError as exc:
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
            "trained_model_paths": {},
            "training_metrics": {},
        }

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core training logic ───────────────────────────────────────────────────────

def _run_training(state: WorkflowState) -> dict[str, Any]:
    """
    Pure training logic — loads data, trains all models in parallel,
    collects metrics, saves artifacts.
    """
    feature_path = state.get("feature_dataset_path", "")
    target_column = state.get("target_column", "")
    task_type: TaskType = state.get("task_type") or TaskType.CLASSIFICATION
    experiment_id = state.get("experiment_id") or uuid.uuid4().hex
    plan: dict[str, Any] = state.get("plan") or {}
    pipeline_config: dict[str, Any] = state.get("pipeline_config") or {}
    selected_features: list[str] = state.get("selected_features") or []

    # ── 1. Load feature dataset ───────────────────────────────────────────────
    path = Path(feature_path)
    if not path.exists():
        raise TrainingError(f"Feature dataset not found: '{feature_path}'")

    df = _load_csv(path)
    logger.debug(f"[training] Loaded feature dataset shape={df.shape}")

    # ── 2. Validate ───────────────────────────────────────────────────────────
    if not target_column:
        raise TrainingError("target_column is required but was not provided.")
    if target_column not in df.columns:
        raise TrainingError(
            f"Target column '{target_column}' not found in feature dataset."
        )

    # ── 3. Resolve model list ─────────────────────────────────────────────────
    training_cfg: dict[str, Any] = {}
    for step in plan.get("steps", []):
        if step.get("agent") == "training":
            training_cfg = step.get("config", {})
            break
    training_cfg.update(pipeline_config.get("training", {}))

    if task_type == TaskType.CLASSIFICATION:
        default_models = ["random_forest", "logistic_regression", "xgboost"]
    else:
        default_models = ["random_forest", "linear_regression", "xgboost"]

    models_to_train: list[str] = training_cfg.get("models", default_models)
    # Filter to only models that support the task type
    models_to_train = [
        m for m in models_to_train
        if m in _MODEL_REGISTRY and _MODEL_REGISTRY[m][task_type.value] is not None
    ]
    if not models_to_train:
        raise TrainingError(
            f"No valid models for task_type='{task_type.value}'. "
            f"Requested: {training_cfg.get('models', default_models)}"
        )

    # ── 4. Prepare X / y ──────────────────────────────────────────────────────
    # Use selected_features if available, otherwise all non-target columns
    feature_cols = (
        [c for c in selected_features if c in df.columns and c != target_column]
        if selected_features
        else [c for c in df.columns if c != target_column]
    )
    X = df[feature_cols].values.astype(float)
    y_raw = df[target_column]

    # Encode string targets for classification
    label_map: dict[str, int] | None = None
    if task_type == TaskType.CLASSIFICATION and not pd.api.types.is_numeric_dtype(y_raw):
        classes = sorted(y_raw.unique())
        label_map = {cls: i for i, cls in enumerate(classes)}
        y = np.array([label_map[v] for v in y_raw])
    else:
        y = y_raw.values.astype(float)

    # ── 5. Train/test split ───────────────────────────────────────────────────
    n = len(X)
    if n >= _MIN_ROWS_FOR_SPLIT:
        split = int(n * (1 - _TEST_SIZE))
        # Shuffle with fixed seed for reproducibility
        rng = np.random.default_rng(_RANDOM_STATE)
        idx = rng.permutation(n)
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Too few rows — train and evaluate on full dataset
        X_train = X_test = X
        y_train = y_test = y

    logger.debug(
        f"[training] Split: train={len(X_train)} test={len(X_test)} "
        f"features={len(feature_cols)}"
    )

    # ── 6. Prepare output directory ───────────────────────────────────────────
    import app.utils.file_manager as _fm
    model_dir = _fm.OUTPUTS_MODELS_DIR / experiment_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 7. Train all models in parallel ──────────────────────────────────────
    trained_model_paths: dict[str, str] = {}
    training_metrics: dict[str, Any] = {}
    models_failed: list[str] = []

    def _train_one(model_name: str) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
        """Train a single model. Returns (name, path, metrics, error)."""
        try:
            t0 = time.perf_counter()
            registry = _MODEL_REGISTRY[model_name]
            ModelClass = registry[task_type.value]
            params = dict(registry["params"])

            # XGBoost regression uses different eval_metric
            if model_name == "xgboost" and task_type == TaskType.REGRESSION:
                params["eval_metric"] = "rmse"

            model = ModelClass(**params)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = None
            if task_type == TaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                    # For binary classification, use probability of positive class
                    if y_prob.shape[1] == 2:
                        y_prob = y_prob[:, 1]
                except Exception:
                    y_prob = None

            # Metrics
            metrics = compute_metrics(y_test, y_pred, task_type, y_prob)
            metrics["train_duration_seconds"] = round(time.perf_counter() - t0, 3)
            metrics["n_train"] = len(X_train)
            metrics["n_test"] = len(X_test)

            # Save model
            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "feature_cols": feature_cols,
                             "label_map": label_map}, f)

            logger.info(
                f"[training] {model_name} done in "
                f"{metrics['train_duration_seconds']:.2f}s — "
                f"metrics={metrics}"
            )
            return model_name, str(model_path), metrics, None

        except Exception as exc:
            logger.error(f"[training] {model_name} failed: {exc}")
            return model_name, None, None, str(exc)

    # ThreadPoolExecutor — true parallel training
    max_workers = min(len(models_to_train), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_train_one, name): name
            for name in models_to_train
        }
        for future in as_completed(futures):
            name, model_path, metrics, error = future.result()
            if error:
                models_failed.append(name)
                logger.warning(f"[training] Skipping {name} due to error: {error}")
            else:
                trained_model_paths[name] = model_path
                training_metrics[name] = metrics

    if not trained_model_paths:
        raise TrainingError(
            f"All models failed to train. Failures: {models_failed}"
        )

    return {
        "trained_model_paths": trained_model_paths,
        "training_metrics": training_metrics,
        "models_failed": models_failed,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise TrainingError(f"Could not read CSV file: {path}")


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: TaskType,
    feature_cols: list[str],
    model_dir: Path,
    label_map: dict | None = None,
) -> dict[str, Any]:
    """
    Public helper — train one model and return its metrics + path.
    Used by tests and by the LangGraph Send-based fan-out (future enhancement).
    """
    registry = _MODEL_REGISTRY.get(model_name)
    if registry is None:
        raise TrainingError(f"Unknown model: '{model_name}'")

    ModelClass = registry[task_type.value]
    if ModelClass is None:
        raise TrainingError(
            f"Model '{model_name}' does not support task_type='{task_type.value}'"
        )

    params = dict(registry["params"])
    if model_name == "xgboost" and task_type == TaskType.REGRESSION:
        params["eval_metric"] = "rmse"

    model = ModelClass(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = None
    if task_type == TaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
        except Exception:
            pass

    metrics = compute_metrics(y_test, y_pred, task_type, y_prob)

    model_path = model_dir / f"{model_name}.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols,
                     "label_map": label_map}, f)

    return {"metrics": metrics, "model_path": str(model_path)}
