"""
File system helpers for managing uploaded datasets and output artifacts.

All path construction goes through this module so the rest of the codebase
never hard-codes directory names.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Base directories (resolved relative to project root) ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = _PROJECT_ROOT / "datasets"
OUTPUTS_MODELS_DIR = _PROJECT_ROOT / "outputs" / "models"
OUTPUTS_CHARTS_DIR = _PROJECT_ROOT / "outputs" / "charts"
OUTPUTS_ARTIFACTS_DIR = _PROJECT_ROOT / "outputs" / "artifacts"
REPORTS_MARKDOWN_DIR = _PROJECT_ROOT / "reports" / "markdown"
REPORTS_HTML_DIR = _PROJECT_ROOT / "reports" / "html"

_ALL_DIRS = [
    DATASETS_DIR,
    OUTPUTS_MODELS_DIR,
    OUTPUTS_CHARTS_DIR,
    OUTPUTS_ARTIFACTS_DIR,
    REPORTS_MARKDOWN_DIR,
    REPORTS_HTML_DIR,
]


def ensure_directories() -> None:
    """Create all required output directories if they don't exist."""
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    logger.debug("Output directories verified.")


def dataset_upload_path(original_filename: str) -> Path:
    """
    Return a unique, safe path for a newly uploaded dataset file.
    The UUID prefix prevents collisions and path-traversal attacks.
    """
    safe_name = Path(original_filename).name  # strip any directory components
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    return DATASETS_DIR / unique_name


def model_artifact_path(experiment_id: str, model_name: str) -> Path:
    """Return the path where a trained model pickle should be saved."""
    return OUTPUTS_MODELS_DIR / experiment_id / f"{model_name}.pkl"


def chart_path(experiment_id: str, chart_name: str) -> Path:
    """Return the path for a generated chart image."""
    return OUTPUTS_CHARTS_DIR / experiment_id / f"{chart_name}.png"


def report_markdown_path(experiment_id: str) -> Path:
    return REPORTS_MARKDOWN_DIR / f"{experiment_id}.md"


def report_html_path(experiment_id: str) -> Path:
    return REPORTS_HTML_DIR / f"{experiment_id}.html"


def get_file_size(path: str | Path) -> int:
    """Return file size in bytes, or 0 if the file does not exist."""
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def delete_file(path: str | Path) -> None:
    """Delete a file silently if it exists."""
    p = Path(path)
    if p.exists():
        p.unlink()
        logger.debug(f"Deleted file: {p}")


def delete_directory(path: str | Path) -> None:
    """Recursively delete a directory silently if it exists."""
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
        logger.debug(f"Deleted directory: {p}")
