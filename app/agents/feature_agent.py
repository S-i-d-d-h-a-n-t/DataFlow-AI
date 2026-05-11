"""
Feature Engineering Agent — fourth node in the LangGraph pipeline.

Responsibilities:
  1. Load the cleaned dataset from disk (cleaned_dataset_path from WorkflowState).
  2. Apply a reproducible feature engineering pipeline:
       a. Separate features (X) from target (y).
       b. Ordinal-encode low-cardinality categorical columns (≤ threshold unique values).
       c. One-hot encode remaining categorical columns (drop_first to avoid multicollinearity).
       d. Standard-scale all numeric feature columns (zero mean, unit variance).
       e. Drop near-zero-variance features (std < threshold).
       f. Select top-k features by mutual information (configurable, default k=20).
  3. Save the engineered feature matrix + target as a new CSV to
     outputs/artifacts/{exp_id}/feature_dataset.csv.
  4. Persist the fitted transformers (scaler, encoders) as a pickle to
     outputs/artifacts/{exp_id}/transformers.pkl — needed by the training agent.
  5. Write feature_dataset_path, selected_features, and feature_report
     back into WorkflowState.
  6. Append a NodeResult.

Design rules:
  - Pure function signature: (WorkflowState) → dict patch.
  - No database access.
  - No LangGraph imports.
  - All transformer state is serialised to disk — never stored in WorkflowState.
  - Raises FeatureError on unrecoverable problems.
"""

from __future__ import annotations

import pickle
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.utils.dataframe_utils import infer_column_dtypes
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_ORDINAL_CARDINALITY_THRESHOLD = 10   # ≤ this → ordinal encode; > this → one-hot
_NEAR_ZERO_VAR_THRESHOLD = 1e-6       # drop feature if std < this after scaling
_DEFAULT_TOP_K_FEATURES = 20          # max features to keep after MI selection
_OHE_MAX_CATEGORIES = 50              # safety cap on one-hot columns per feature


# ── Domain exception ──────────────────────────────────────────────────────────

class FeatureError(Exception):
    """Raised when the Feature Engineering Agent cannot complete its work."""


# ── Agent callable ────────────────────────────────────────────────────────────

def feature_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the Feature Engineering Agent.

    Reads from state:
        cleaned_dataset_path, target_column, task_type,
        experiment_id, plan, pipeline_config

    Writes to state:
        feature_dataset_path, selected_features, feature_report,
        node_results (appended)
    """
    start = time.perf_counter()
    node_name = "feature_engineering"
    logger.info(
        f"[{node_name}] Starting — "
        f"dataset={state.get('cleaned_dataset_path')}"
    )

    try:
        result = _run_feature_engineering(state)
        duration = round(time.perf_counter() - start, 3)

        report = result["feature_report"]
        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "features_before": report["features_before"],
                "features_after": report["features_after"],
                "features_dropped_low_variance": report["features_dropped_low_variance"],
                "features_selected": report["features_selected"],
                "numeric_scaled": report["numeric_scaled"],
                "categorical_ordinal_encoded": report["categorical_ordinal_encoded"],
                "categorical_ohe_encoded": report["categorical_ohe_encoded"],
                "feature_dataset_path": result["feature_dataset_path"],
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"{report['features_before']}→{report['features_after']} features, "
            f"selected={report['features_selected']}"
        )

    except FeatureError as exc:
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
        # Fall back to cleaned path so downstream agents can still attempt to run
        plan = state.get("plan") or {}
        return {
            "node_results": existing,
            "errors": errors,
            "feature_dataset_path": state.get("cleaned_dataset_path", ""),
            "feature_report": {},
            "selected_features": plan.get("feature_columns", []),
        }

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core feature engineering logic ────────────────────────────────────────────

def _run_feature_engineering(state: WorkflowState) -> dict[str, Any]:
    """
    Pure feature engineering logic — separated from the node wrapper for testability.
    Returns a dict ready to be merged into WorkflowState.
    """
    cleaned_path = state.get("cleaned_dataset_path", "")
    target_column = state.get("target_column", "")
    task_type: TaskType = state.get("task_type") or TaskType.CLASSIFICATION
    experiment_id = state.get("experiment_id") or uuid.uuid4().hex
    plan: dict[str, Any] = state.get("plan") or {}
    pipeline_config: dict[str, Any] = state.get("pipeline_config") or {}

    # ── 1. Load cleaned dataset ───────────────────────────────────────────────
    path = Path(cleaned_path)
    if not path.exists():
        raise FeatureError(f"Cleaned dataset not found: '{cleaned_path}'")

    df = _load_csv(path)
    logger.debug(f"[feature] Loaded cleaned dataset shape={df.shape}")

    # ── 2. Validate target column ─────────────────────────────────────────────
    if not target_column:
        raise FeatureError("target_column is required but was not provided.")
    if target_column not in df.columns:
        raise FeatureError(
            f"Target column '{target_column}' not found in cleaned dataset."
        )

    # ── 3. Resolve feature engineering config ─────────────────────────────────
    fe_cfg: dict[str, Any] = {}
    for step in plan.get("steps", []):
        if step.get("agent") == "feature_engineering":
            fe_cfg = step.get("config", {})
            break
    fe_cfg.update(pipeline_config.get("feature_engineering", {}))

    top_k: int = fe_cfg.get("top_k_features", _DEFAULT_TOP_K_FEATURES)
    ordinal_threshold: int = fe_cfg.get(
        "ordinal_cardinality_threshold", _ORDINAL_CARDINALITY_THRESHOLD
    )
    drop_low_variance: bool = fe_cfg.get("drop_low_variance", True)
    run_feature_selection: bool = fe_cfg.get("feature_selection", True)

    # ── 4. Separate X and y ───────────────────────────────────────────────────
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    feature_cols_before = list(X.columns)
    dtypes = infer_column_dtypes(X)
    numeric_cols = [c for c, t in dtypes.items() if t == "numeric"]
    categorical_cols = [c for c, t in dtypes.items() if t == "categorical"]

    # ── 5. Encode categorical features ───────────────────────────────────────
    ordinal_encoded: list[str] = []
    ohe_encoded: list[str] = []
    encoders: dict[str, Any] = {}

    # Ordinal encode low-cardinality categoricals
    ordinal_cols = [
        c for c in categorical_cols
        if X[c].nunique() <= ordinal_threshold
    ]
    if ordinal_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        X[ordinal_cols] = enc.fit_transform(X[ordinal_cols].astype(str))
        encoders["ordinal"] = {"encoder": enc, "columns": ordinal_cols}
        ordinal_encoded = ordinal_cols
        logger.debug(f"[feature] Ordinal-encoded: {ordinal_cols}")

    # One-hot encode high-cardinality categoricals
    ohe_cols = [
        c for c in categorical_cols
        if c not in ordinal_cols
    ]
    if ohe_cols:
        # Cap categories to avoid explosion
        for col in ohe_cols:
            top_cats = X[col].value_counts().nlargest(_OHE_MAX_CATEGORIES).index
            X[col] = X[col].where(X[col].isin(top_cats), other="__other__")

        X = pd.get_dummies(X, columns=ohe_cols, drop_first=True, dtype=float)
        encoders["ohe"] = {"columns": ohe_cols}
        ohe_encoded = ohe_cols
        logger.debug(f"[feature] One-hot encoded: {ohe_cols}")

    # ── 6. Standard-scale numeric features ───────────────────────────────────
    # Re-identify numeric columns after encoding (OHE adds new float cols)
    current_dtypes = infer_column_dtypes(X)
    scale_cols = [
        c for c, t in current_dtypes.items()
        if t == "numeric" and c in X.columns
    ]
    scaler = StandardScaler()
    if scale_cols:
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
        logger.debug(f"[feature] Scaled {len(scale_cols)} numeric columns.")
    encoders["scaler"] = {"scaler": scaler, "columns": scale_cols}

    # ── 7. Drop near-zero-variance features ──────────────────────────────────
    dropped_low_var: list[str] = []
    if drop_low_variance:
        stds = X.std(numeric_only=True)
        low_var = stds[stds < _NEAR_ZERO_VAR_THRESHOLD].index.tolist()
        if low_var:
            X = X.drop(columns=low_var)
            dropped_low_var = low_var
            logger.debug(f"[feature] Dropped {len(low_var)} near-zero-variance features.")

    features_after_variance = list(X.columns)

    # ── 8. Mutual information feature selection ───────────────────────────────
    selected_features: list[str] = list(X.columns)
    mi_scores: dict[str, float] = {}

    if run_feature_selection and len(X.columns) > top_k:
        try:
            mi_fn = (
                mutual_info_classif
                if task_type == TaskType.CLASSIFICATION
                else mutual_info_regression
            )
            # Ensure y is numeric for MI computation
            y_for_mi = y.copy()
            if not pd.api.types.is_numeric_dtype(y_for_mi):
                y_for_mi = pd.factorize(y_for_mi)[0]

            scores = mi_fn(X.fillna(0), y_for_mi, random_state=42)
            mi_scores = {
                col: round(float(s), 6)
                for col, s in zip(X.columns, scores)
            }
            # Keep top-k by MI score
            top_cols = sorted(mi_scores, key=mi_scores.get, reverse=True)[:top_k]  # type: ignore[arg-type]
            X = X[top_cols]
            selected_features = top_cols
            logger.debug(
                f"[feature] MI selection: {len(features_after_variance)}"
                f"→{len(selected_features)} features."
            )
        except Exception as exc:
            # MI can fail on edge cases — log and keep all features
            logger.warning(f"[feature] MI selection failed ({exc}), keeping all features.")
            selected_features = features_after_variance

    # ── 9. Reconstruct full dataset (features + target) ───────────────────────
    X[target_column] = y.values

    # ── 10. Save feature dataset ──────────────────────────────────────────────
    import app.utils.file_manager as _fm
    artifact_dir = _fm.OUTPUTS_ARTIFACTS_DIR / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    feature_path = artifact_dir / "feature_dataset.csv"
    X.to_csv(feature_path, index=False)
    logger.info(f"[feature] Saved feature dataset → {feature_path}")

    # Save fitted transformers for use by the training agent
    transformer_path = artifact_dir / "transformers.pkl"
    with open(transformer_path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"[feature] Saved transformers → {transformer_path}")

    # ── 11. Build report ──────────────────────────────────────────────────────
    feature_report: dict[str, Any] = {
        "features_before": len(feature_cols_before),
        "features_after": len(selected_features),
        "features_dropped_low_variance": dropped_low_var,
        "features_selected": selected_features,
        "numeric_scaled": scale_cols,
        "categorical_ordinal_encoded": ordinal_encoded,
        "categorical_ohe_encoded": ohe_encoded,
        "mi_scores": mi_scores,
        "top_k": top_k,
        "ordinal_cardinality_threshold": ordinal_threshold,
        "transformer_path": str(transformer_path),
        "task_type": task_type.value,
    }

    return {
        "feature_dataset_path": str(feature_path),
        "selected_features": selected_features,
        "feature_report": feature_report,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise FeatureError(f"Could not read CSV file: {path}")
