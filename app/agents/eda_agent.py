"""
EDA Agent — second node in the LangGraph pipeline.

Responsibilities:
  1. Load the dataset from disk (path comes from WorkflowState).
  2. Compute a comprehensive statistical summary.
  3. Generate and save charts to outputs/charts/{experiment_id}/.
  4. Write eda_summary and chart_paths back into WorkflowState.
  5. Append a NodeResult so the orchestrator can track its outcome.

Charts produced (all saved as PNG):
  - distributions/  : histogram + KDE for every numeric column
  - boxplots/       : box plot per numeric column
  - correlation/    : heatmap of the Pearson correlation matrix
  - target/         : target distribution (bar for classification, hist for regression)
  - missing/        : horizontal bar chart of missing-value percentages (if any)

Design rules:
  - Pure function signature: (WorkflowState) → dict patch.
  - No database access.
  - No LangGraph imports.
  - All matplotlib figures are explicitly closed after saving to prevent memory leaks.
  - Raises EDAError on unrecoverable problems.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in server context
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.workflow_status import WorkflowNodeStatus
from app.utils.dataframe_utils import (
    infer_column_dtypes,
    missing_value_report,
    cardinality_report,
    column_stats,
)
from app.utils.file_manager import ensure_directories
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_CATEGORIES_IN_BAR = 20   # cap bar charts to avoid unreadable plots
_FIGURE_DPI = 100
_FIGURE_SIZE = (10, 5)


# ── Domain exception ──────────────────────────────────────────────────────────

class EDAError(Exception):
    """Raised when the EDA agent cannot complete its analysis."""


# ── Agent callable ────────────────────────────────────────────────────────────

def eda_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the EDA Agent.

    Reads from state:
        dataset_path, target_column, experiment_id, plan

    Writes to state:
        eda_summary, chart_paths, node_results (appended)
    """
    start = time.perf_counter()
    node_name = "eda"
    logger.info(f"[{node_name}] Starting — dataset={state.get('dataset_path')}")

    try:
        result = _run_eda(state)
        duration = round(time.perf_counter() - start, 3)

        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "num_charts": len(result["chart_paths"]),
                "numeric_columns": result["eda_summary"]["numeric_columns"],
                "categorical_columns": result["eda_summary"]["categorical_columns"],
                "missing_columns": result["eda_summary"]["missing_columns"],
                "total_rows": result["eda_summary"]["shape"]["rows"],
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"{len(result['chart_paths'])} charts generated."
        )

    except EDAError as exc:
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
            "eda_summary": {},
            "chart_paths": [],
        }

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core EDA logic ────────────────────────────────────────────────────────────

def _run_eda(state: WorkflowState) -> dict[str, Any]:
    """
    Pure EDA logic — separated from the node wrapper for testability.
    Returns a dict ready to be merged into WorkflowState.
    """
    dataset_path = state.get("dataset_path", "")
    target_column = state.get("target_column", "")
    experiment_id = state.get("experiment_id", "unknown")
    plan: dict[str, Any] = state.get("plan") or {}

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    path = Path(dataset_path)
    if not path.exists():
        raise EDAError(f"Dataset file not found: '{dataset_path}'")

    df = _load_csv(path)
    logger.debug(f"[eda] Loaded dataset shape={df.shape}")

    # ── 2. Validate target column ─────────────────────────────────────────────
    if target_column and target_column not in df.columns:
        raise EDAError(
            f"Target column '{target_column}' not found in dataset."
        )

    # ── 3. Structural analysis ────────────────────────────────────────────────
    dtypes = infer_column_dtypes(df)
    missing = missing_value_report(df)
    cardinality = cardinality_report(df)
    stats = column_stats(df)

    numeric_cols = [c for c, t in dtypes.items() if t == "numeric"]
    categorical_cols = [c for c, t in dtypes.items() if t == "categorical"]
    feature_cols: list[str] = plan.get("feature_columns") or [
        c for c in df.columns if c != target_column
    ]

    # ── 4. Correlation matrix (numeric features only) ─────────────────────────
    numeric_features = [c for c in feature_cols if c in numeric_cols]
    correlation_matrix: dict[str, Any] = {}
    if len(numeric_features) >= 2:
        corr = df[numeric_features].corr(method="pearson")
        # Replace NaN with None for JSON safety
        correlation_matrix = {
            col: {
                other: (None if np.isnan(v) else round(float(v), 4))
                for other, v in row.items()
            }
            for col, row in corr.to_dict().items()
        }

    # ── 5. Target analysis ────────────────────────────────────────────────────
    target_analysis: dict[str, Any] = {}
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        if dtypes.get(target_column) == "numeric":
            target_analysis = {
                "type": "numeric",
                "mean": round(float(target_series.mean()), 4),
                "std": round(float(target_series.std()), 4),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "median": float(target_series.median()),
            }
        else:
            vc = target_series.value_counts()
            target_analysis = {
                "type": "categorical",
                "class_counts": {str(k): int(v) for k, v in vc.items()},
                "class_balance": {
                    str(k): round(float(v) / len(target_series) * 100, 2)
                    for k, v in vc.items()
                },
            }

    # ── 6. Generate charts ────────────────────────────────────────────────────
    ensure_directories()
    # Import at call time so tests can patch app.utils.file_manager.OUTPUTS_CHARTS_DIR
    import app.utils.file_manager as _fm
    chart_dir = _fm.OUTPUTS_CHARTS_DIR / experiment_id
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: list[str] = []
    chart_paths.extend(_plot_distributions(df, numeric_cols, chart_dir))
    chart_paths.extend(_plot_boxplots(df, numeric_cols, chart_dir))
    chart_paths.extend(_plot_categorical_bars(df, categorical_cols, chart_dir))
    if len(numeric_features) >= 2:
        chart_paths.extend(
            _plot_correlation_heatmap(df, numeric_features, chart_dir)
        )
    if target_column and target_column in df.columns:
        chart_paths.extend(
            _plot_target_distribution(df, target_column, dtypes, chart_dir)
        )
    if missing:
        chart_paths.extend(_plot_missing_values(missing, chart_dir))

    # ── 7. Build summary ──────────────────────────────────────────────────────
    eda_summary: dict[str, Any] = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_columns": list(missing.keys()),
        "missing_report": missing,
        "cardinality": cardinality,
        "column_stats": stats,
        "correlation_matrix": correlation_matrix,
        "target_analysis": target_analysis,
        "chart_dir": str(chart_dir),
    }

    return {"eda_summary": eda_summary, "chart_paths": chart_paths}


# ── Chart generators ──────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: Path) -> str:
    """Save a figure, close it, and return the path string."""
    fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_distributions(
    df: pd.DataFrame, numeric_cols: list[str], chart_dir: Path
) -> list[str]:
    """Histogram + KDE for every numeric column."""
    paths: list[str] = []
    dist_dir = chart_dir / "distributions"
    dist_dir.mkdir(exist_ok=True)

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
        ax.hist(series, bins=30, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", label="Histogram")
        try:
            series.plot.kde(ax=ax, color="darkblue", linewidth=2, label="KDE")
        except Exception:
            pass  # KDE can fail on degenerate distributions
        ax.set_title(f"Distribution — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        paths.append(_save_fig(fig, dist_dir / f"{_safe_name(col)}.png"))

    return paths


def _plot_boxplots(
    df: pd.DataFrame, numeric_cols: list[str], chart_dir: Path
) -> list[str]:
    """Box plot per numeric column."""
    if not numeric_cols:
        return []

    box_dir = chart_dir / "boxplots"
    box_dir.mkdir(exist_ok=True)
    paths: list[str] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(series, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.7))
        ax.set_title(f"Box Plot — {col}")
        ax.set_ylabel(col)
        ax.set_xticks([])
        paths.append(_save_fig(fig, box_dir / f"{_safe_name(col)}.png"))

    return paths


def _plot_categorical_bars(
    df: pd.DataFrame, categorical_cols: list[str], chart_dir: Path
) -> list[str]:
    """Horizontal bar chart of value counts for categorical columns."""
    if not categorical_cols:
        return []

    cat_dir = chart_dir / "categorical"
    cat_dir.mkdir(exist_ok=True)
    paths: list[str] = []

    for col in categorical_cols:
        vc = df[col].value_counts().head(_MAX_CATEGORIES_IN_BAR)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
        vc.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Value Counts — {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        paths.append(_save_fig(fig, cat_dir / f"{_safe_name(col)}.png"))

    return paths


def _plot_correlation_heatmap(
    df: pd.DataFrame, numeric_features: list[str], chart_dir: Path
) -> list[str]:
    """Pearson correlation heatmap for numeric features."""
    corr = df[numeric_features].corr(method="pearson")
    n = len(numeric_features)
    fig_size = (max(8, n), max(6, n - 1))
    fig, ax = plt.subplots(figsize=fig_size)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": max(7, 10 - n // 3)},
    )
    ax.set_title("Pearson Correlation Matrix")
    path = chart_dir / "correlation_heatmap.png"
    return [_save_fig(fig, path)]


def _plot_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    dtypes: dict[str, str],
    chart_dir: Path,
) -> list[str]:
    """Target column distribution — bar for classification, hist for regression."""
    series = df[target_column].dropna()
    fig, ax = plt.subplots(figsize=_FIGURE_SIZE)

    if dtypes.get(target_column) == "numeric":
        ax.hist(series, bins=30, color="coral", edgecolor="white", alpha=0.8)
        ax.set_title(f"Target Distribution — {target_column} (regression)")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Count")
    else:
        vc = series.value_counts().head(_MAX_CATEGORIES_IN_BAR)
        vc.plot.bar(ax=ax, color="coral", edgecolor="white", alpha=0.8)
        ax.set_title(f"Target Distribution — {target_column} (classification)")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    path = chart_dir / "target_distribution.png"
    return [_save_fig(fig, path)]


def _plot_missing_values(
    missing: dict[str, Any], chart_dir: Path
) -> list[str]:
    """Horizontal bar chart of missing-value percentages."""
    cols = list(missing.keys())
    pcts = [missing[c]["pct"] for c in cols]

    fig, ax = plt.subplots(figsize=(8, max(3, len(cols) * 0.5)))
    bars = ax.barh(cols, pcts, color="salmon", edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Value Percentages")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=9,
        )
    path = chart_dir / "missing_values.png"
    return [_save_fig(fig, path)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_name(col: str) -> str:
    """Convert a column name to a filesystem-safe string."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in col)[:60]


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise EDAError(f"Could not read CSV file: {path}")
