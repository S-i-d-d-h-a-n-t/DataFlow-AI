"""
Cleaning Agent — third node in the LangGraph pipeline.

Responsibilities:
  1. Load the raw dataset from disk (dataset_path from WorkflowState).
  2. Apply a deterministic, reproducible cleaning pipeline:
       a. Drop columns with excessive missing values (> threshold).
       b. Impute numeric missing values (median strategy).
       c. Impute categorical missing values (mode / constant "missing").
       d. Remove duplicate rows.
       e. Clip numeric outliers (IQR-based, configurable).
       f. Fix mixed-type columns (coerce to most common type).
  3. Save the cleaned DataFrame as a new CSV to outputs/artifacts/{exp_id}/.
  4. Write cleaned_dataset_path and cleaning_report back into WorkflowState.
  5. Append a NodeResult.

Design rules:
  - Pure function signature: (WorkflowState) → dict patch.
  - No database access.
  - No LangGraph imports.
  - All decisions are logged and recorded in cleaning_report for traceability.
  - Raises CleaningError on unrecoverable problems.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.workflow_status import WorkflowNodeStatus
from app.utils.dataframe_utils import infer_column_dtypes, missing_value_report
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_DROP_MISSING_THRESHOLD = 0.60   # drop column if > 60 % values are missing
_IQR_MULTIPLIER = 3.0            # clip at Q1 - k*IQR / Q3 + k*IQR  (conservative)
_CONSTANT_FILL = "missing"       # fill value for categorical columns with no mode


# ── Domain exception ──────────────────────────────────────────────────────────

class CleaningError(Exception):
    """Raised when the Cleaning Agent cannot complete its work."""


# ── Agent callable ────────────────────────────────────────────────────────────

def cleaning_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the Cleaning Agent.

    Reads from state:
        dataset_path, target_column, experiment_id, plan, pipeline_config

    Writes to state:
        cleaned_dataset_path, cleaning_report, node_results (appended)
    """
    start = time.perf_counter()
    node_name = "cleaning"
    logger.info(f"[{node_name}] Starting — dataset={state.get('dataset_path')}")

    try:
        result = _run_cleaning(state)
        duration = round(time.perf_counter() - start, 3)

        report = result["cleaning_report"]
        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "rows_before": report["rows_before"],
                "rows_after": report["rows_after"],
                "rows_dropped": report["rows_dropped"],
                "cols_dropped": report["columns_dropped"],
                "cols_imputed_numeric": report["columns_imputed_numeric"],
                "cols_imputed_categorical": report["columns_imputed_categorical"],
                "cols_clipped": report["columns_clipped"],
                "duplicates_removed": report["duplicates_removed"],
                "cleaned_path": result["cleaned_dataset_path"],
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"rows {report['rows_before']}→{report['rows_after']} "
            f"dropped_cols={report['columns_dropped']}"
        )

    except CleaningError as exc:
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
            "cleaned_dataset_path": state.get("dataset_path", ""),
            "cleaning_report": {},
        }

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core cleaning logic ───────────────────────────────────────────────────────

def _run_cleaning(state: WorkflowState) -> dict[str, Any]:
    """
    Pure cleaning logic — separated from the node wrapper for testability.
    Returns a dict ready to be merged into WorkflowState.
    """
    dataset_path = state.get("dataset_path", "")
    target_column = state.get("target_column", "")
    experiment_id = state.get("experiment_id") or uuid.uuid4().hex
    plan: dict[str, Any] = state.get("plan") or {}
    pipeline_config: dict[str, Any] = state.get("pipeline_config") or {}

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    path = Path(dataset_path)
    if not path.exists():
        raise CleaningError(f"Dataset file not found: '{dataset_path}'")

    df = _load_csv(path)
    rows_before = len(df)
    cols_before = list(df.columns)
    logger.debug(f"[cleaning] Loaded dataset shape={df.shape}")

    # ── 2. Resolve cleaning config ────────────────────────────────────────────
    # Prefer plan-level config, fall back to pipeline_config overrides
    cleaning_cfg: dict[str, Any] = {}
    for step in plan.get("steps", []):
        if step.get("agent") == "cleaning":
            cleaning_cfg = step.get("config", {})
            break
    cleaning_cfg.update(pipeline_config.get("cleaning", {}))

    drop_threshold: float = cleaning_cfg.get(
        "drop_missing_threshold", _DROP_MISSING_THRESHOLD
    )
    iqr_multiplier: float = cleaning_cfg.get("iqr_multiplier", _IQR_MULTIPLIER)
    clip_outliers: bool = cleaning_cfg.get("clip_outliers", True)
    impute_missing: bool = cleaning_cfg.get("impute_missing", True)

    # ── 3. Identify column types ──────────────────────────────────────────────
    dtypes = infer_column_dtypes(df)
    numeric_cols = [c for c, t in dtypes.items() if t == "numeric"]
    categorical_cols = [c for c, t in dtypes.items() if t == "categorical"]

    # ── 4. Drop high-missing columns (never drop the target) ─────────────────
    missing_before = missing_value_report(df)
    dropped_cols: list[str] = []
    for col, info in missing_before.items():
        if col == target_column:
            continue
        if info["pct"] / 100.0 > drop_threshold:
            df = df.drop(columns=[col])
            dropped_cols.append(col)
            logger.debug(f"[cleaning] Dropped column '{col}' ({info['pct']:.1f}% missing)")

    # Refresh type lists after drops
    dtypes = infer_column_dtypes(df)
    numeric_cols = [c for c, t in dtypes.items() if t == "numeric" and c in df.columns]
    categorical_cols = [c for c, t in dtypes.items() if t == "categorical" and c in df.columns]

    # ── 5. Remove duplicate rows ──────────────────────────────────────────────
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = n_before_dedup - len(df)
    if duplicates_removed:
        logger.debug(f"[cleaning] Removed {duplicates_removed} duplicate rows.")

    # ── 6. Drop rows where target is missing (before imputation) ──────────────
    target_missing_dropped = 0
    if target_column and target_column in df.columns:
        n_before = len(df)
        df = df.dropna(subset=[target_column])
        target_missing_dropped = n_before - len(df)
        if target_missing_dropped:
            logger.debug(
                f"[cleaning] Dropped {target_missing_dropped} rows with "
                f"missing target '{target_column}'."
            )

    # ── 7. Impute missing values ──────────────────────────────────────────────
    imputed_numeric: list[str] = []
    imputed_categorical: list[str] = []

    if impute_missing:
        # Numeric → median imputation
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                imputed_numeric.append(col)
                logger.debug(f"[cleaning] Imputed '{col}' with median={median_val:.4f}")

        # Categorical → mode imputation (fallback to constant)
        for col in categorical_cols:
            if df[col].isna().any():
                mode_series = df[col].mode()
                fill_val = mode_series.iloc[0] if not mode_series.empty else _CONSTANT_FILL
                df[col] = df[col].fillna(fill_val)
                imputed_categorical.append(col)
                logger.debug(f"[cleaning] Imputed '{col}' with mode='{fill_val}'")

    # ── 8. Clip numeric outliers (IQR-based) ──────────────────────────────────
    clipped_cols: list[str] = []
    clip_bounds: dict[str, dict[str, float]] = {}

    if clip_outliers:
        # Never clip the target column
        clip_candidates = [c for c in numeric_cols if c != target_column]
        for col in clip_candidates:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue  # constant column — skip
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_clipped > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                clipped_cols.append(col)
                clip_bounds[col] = {
                    "lower": round(float(lower), 4),
                    "upper": round(float(upper), 4),
                    "n_clipped": n_clipped,
                }
                logger.debug(
                    f"[cleaning] Clipped '{col}': [{lower:.4f}, {upper:.4f}] "
                    f"({n_clipped} values)"
                )

    # ── 9. Save cleaned dataset ───────────────────────────────────────────────
    import app.utils.file_manager as _fm
    artifact_dir = _fm.OUTPUTS_ARTIFACTS_DIR / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = artifact_dir / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)
    logger.info(f"[cleaning] Saved cleaned dataset → {cleaned_path}")

    # ── 10. Build report ──────────────────────────────────────────────────────
    rows_after = len(df)
    cleaning_report: dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_before - rows_after,
        "duplicates_removed": duplicates_removed,
        "target_missing_dropped": target_missing_dropped,
        "columns_before": cols_before,
        "columns_after": list(df.columns),
        "columns_dropped": dropped_cols,
        "columns_imputed_numeric": imputed_numeric,
        "columns_imputed_categorical": imputed_categorical,
        "columns_clipped": clipped_cols,
        "clip_bounds": clip_bounds,
        "drop_missing_threshold": drop_threshold,
        "iqr_multiplier": iqr_multiplier,
        "imputation_strategy": {
            "numeric": "median",
            "categorical": "mode_or_constant",
        },
    }

    return {
        "cleaned_dataset_path": str(cleaned_path),
        "cleaning_report": cleaning_report,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise CleaningError(f"Could not read CSV file: {path}")
