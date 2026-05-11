"""
Report Service — serves generated report files.

Responsibilities:
  - Resolve report file paths from experiment records.
  - Read and return Markdown / HTML content.
  - Raise domain exceptions that the API layer maps to HTTP errors.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.repositories.experiment_repository import ExperimentRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


class ReportServiceError(Exception):
    """Base class for report service errors."""


class ReportNotFoundError(ReportServiceError):
    """Raised when no report exists for the given experiment."""


class ExperimentNotFoundError(ReportServiceError):
    """Raised when the experiment does not exist."""


class ReportService:
    """Serves generated report files for completed experiments."""

    def __init__(self, db: Session) -> None:
        self._repo = ExperimentRepository(db)

    def get_markdown(self, experiment_id: str) -> str:
        """Return the Markdown report content for an experiment."""
        exp = self._repo.get_by_id(experiment_id)
        if exp is None:
            raise ExperimentNotFoundError(
                f"Experiment '{experiment_id}' not found."
            )
        if not exp.report_path:
            raise ReportNotFoundError(
                f"No report has been generated for experiment '{experiment_id}'."
            )
        path = Path(exp.report_path)
        if not path.exists():
            raise ReportNotFoundError(
                f"Report file not found on disk: '{exp.report_path}'"
            )
        return path.read_text(encoding="utf-8")

    def get_html(self, experiment_id: str) -> str:
        """Return the HTML report content for an experiment."""
        md_path_str = self._get_report_path(experiment_id)
        # Derive HTML path from Markdown path
        html_path = Path(md_path_str).with_suffix(".html")
        # HTML lives in reports/html/ not reports/markdown/
        html_path = Path(str(html_path).replace(
            "reports/markdown", "reports/html"
        ).replace(
            "reports\\markdown", "reports\\html"
        ))
        if not html_path.exists():
            raise ReportNotFoundError(
                f"HTML report not found on disk: '{html_path}'"
            )
        return html_path.read_text(encoding="utf-8")

    def get_report_paths(self, experiment_id: str) -> dict[str, str]:
        """Return both report file paths for an experiment."""
        md_path_str = self._get_report_path(experiment_id)
        html_path = Path(md_path_str).with_suffix(".html")
        html_path = Path(str(html_path).replace(
            "reports/markdown", "reports/html"
        ).replace(
            "reports\\markdown", "reports\\html"
        ))
        return {
            "markdown_path": md_path_str,
            "html_path": str(html_path),
            "markdown_exists": Path(md_path_str).exists(),
            "html_exists": html_path.exists(),
        }

    def _get_report_path(self, experiment_id: str) -> str:
        exp = self._repo.get_by_id(experiment_id)
        if exp is None:
            raise ExperimentNotFoundError(
                f"Experiment '{experiment_id}' not found."
            )
        if not exp.report_path:
            raise ReportNotFoundError(
                f"No report has been generated for experiment '{experiment_id}'."
            )
        return exp.report_path
