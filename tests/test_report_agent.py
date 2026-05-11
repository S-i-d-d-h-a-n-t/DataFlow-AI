"""
Phase 9 tests — Report Agent.

Groups:
  1. TestRunReport        — pure _run_report() logic (content, structure)
  2. TestMarkdownContent  — Markdown sections and data accuracy
  3. TestHTMLContent      — HTML validity and content
  4. TestReportNode       — report_agent() node wrapper (state, errors)
  5. TestReportFiles      — files written to disk correctly
  6. TestGraphReport      — full graph confirms report node runs last
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agents.report_agent import report_agent, _run_report, _render_markdown, ReportError
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.state.workflow_state import WorkflowState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _classification_df(rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.uniform(0, 1, rows),
        "f2": rng.uniform(-1, 1, rows),
        "f3": rng.uniform(10, 100, rows),
        "f4": rng.uniform(0, 50, rows),
        "target": rng.integers(0, 2, rows),
    })


def _full_state(exp_id: str = "exp-report-001") -> WorkflowState:
    """Build a realistic WorkflowState with all prior agent outputs populated."""
    return {
        "experiment_id": exp_id,
        "dataset_id": "ds-001",
        "dataset_path": "/tmp/raw.csv",
        "target_column": "target",
        "task_type": TaskType.CLASSIFICATION,
        "pipeline_config": {},
        "plan": {
            "task_type": "classification",
            "target_column": "target",
            "feature_columns": ["f1", "f2", "f3"],
            "models": ["random_forest", "logistic_regression", "xgboost"],
        },
        "eda_summary": {
            "shape": {"rows": 150, "columns": 5},
            "numeric_columns": ["f1", "f2", "f3", "f4"],
            "categorical_columns": [],
            "missing_columns": ["f2"],
            "missing_report": {"f2": {"count": 5, "pct": 3.33}},
            "target_analysis": {
                "type": "categorical",
                "class_counts": {"0": 75, "1": 75},
                "class_balance": {"0": 50.0, "1": 50.0},
            },
        },
        "chart_paths": ["/outputs/charts/exp-001/distributions/f1.png",
                        "/outputs/charts/exp-001/correlation_heatmap.png"],
        "cleaning_report": {
            "rows_before": 150,
            "rows_after": 145,
            "rows_dropped": 5,
            "duplicates_removed": 3,
            "target_missing_dropped": 2,
            "columns_dropped": [],
            "columns_imputed_numeric": ["f2"],
            "columns_imputed_categorical": [],
            "columns_clipped": ["f3"],
        },
        "feature_report": {
            "features_before": 4,
            "features_after": 3,
            "features_dropped_low_variance": [],
            "features_selected": ["f1", "f2", "f3"],
            "numeric_scaled": ["f1", "f2", "f3", "f4"],
            "categorical_ordinal_encoded": [],
            "categorical_ohe_encoded": [],
            "top_k": 20,
        },
        "selected_features": ["f1", "f2", "f3"],
        "training_metrics": {
            "random_forest": {
                "accuracy": 0.82, "f1_weighted": 0.82,
                "precision_weighted": 0.83, "recall_weighted": 0.82,
                "train_duration_seconds": 0.45, "n_train": 116, "n_test": 29,
            },
            "logistic_regression": {
                "accuracy": 0.76, "f1_weighted": 0.76,
                "precision_weighted": 0.77, "recall_weighted": 0.76,
                "train_duration_seconds": 0.12, "n_train": 116, "n_test": 29,
            },
            "xgboost": {
                "accuracy": 0.86, "f1_weighted": 0.86,
                "precision_weighted": 0.87, "recall_weighted": 0.86,
                "train_duration_seconds": 0.38, "n_train": 116, "n_test": 29,
            },
        },
        "trained_model_paths": {
            "random_forest": "/outputs/models/exp-001/random_forest.pkl",
            "logistic_regression": "/outputs/models/exp-001/logistic_regression.pkl",
            "xgboost": "/outputs/models/exp-001/xgboost.pkl",
        },
        "evaluation_results": {
            "models_evaluated": ["random_forest", "logistic_regression", "xgboost"],
            "task_type": "classification",
            "primary_metric": "f1_weighted",
            "higher_is_better": True,
            "best_model": "xgboost",
            "best_score": 0.86,
            "ranked_models": [
                {"rank": 1, "model": "xgboost", "primary_score": 0.86, "metrics": {}},
                {"rank": 2, "model": "random_forest", "primary_score": 0.82, "metrics": {}},
                {"rank": 3, "model": "logistic_regression", "primary_score": 0.76, "metrics": {}},
            ],
            "comparison_table": [],
        },
        "best_model_name": "xgboost",
        "best_model_path": "/outputs/models/exp-001/xgboost.pkl",
        "node_results": [
            {"node_name": "planner", "status": WorkflowNodeStatus.DONE,
             "duration_seconds": 0.1},
            {"node_name": "eda", "status": WorkflowNodeStatus.DONE,
             "duration_seconds": 2.3},
            {"node_name": "cleaning", "status": WorkflowNodeStatus.DONE,
             "duration_seconds": 0.4},
            {"node_name": "feature_engineering", "status": WorkflowNodeStatus.DONE,
             "duration_seconds": 0.6},
            {"node_name": "training", "status": WorkflowNodeStatus.DONE,
             "duration_seconds": 3.2},
            {"node_name": "evaluation", "status": WorkflowNodeStatus.DONE,
             "duration_seconds": 0.05},
        ],
        "errors": [],
    }


@pytest.fixture()
def report_dirs(tmp_path):
    """Patch both report directories to tmp_path."""
    import app.utils.file_manager as fm
    orig_md = fm.REPORTS_MARKDOWN_DIR
    orig_html = fm.REPORTS_HTML_DIR
    fm.REPORTS_MARKDOWN_DIR = tmp_path / "markdown"
    fm.REPORTS_HTML_DIR = tmp_path / "html"
    yield tmp_path
    fm.REPORTS_MARKDOWN_DIR = orig_md
    fm.REPORTS_HTML_DIR = orig_html


@pytest.fixture()
def all_dirs(tmp_path):
    """Patch all output dirs for full graph tests."""
    import app.utils.file_manager as fm
    orig = {
        "models": fm.OUTPUTS_MODELS_DIR,
        "artifacts": fm.OUTPUTS_ARTIFACTS_DIR,
        "charts": fm.OUTPUTS_CHARTS_DIR,
        "md": fm.REPORTS_MARKDOWN_DIR,
        "html": fm.REPORTS_HTML_DIR,
    }
    fm.OUTPUTS_MODELS_DIR = tmp_path / "models"
    fm.OUTPUTS_ARTIFACTS_DIR = tmp_path / "artifacts"
    fm.OUTPUTS_CHARTS_DIR = tmp_path / "charts"
    fm.REPORTS_MARKDOWN_DIR = tmp_path / "markdown"
    fm.REPORTS_HTML_DIR = tmp_path / "html"
    yield tmp_path
    fm.OUTPUTS_MODELS_DIR = orig["models"]
    fm.OUTPUTS_ARTIFACTS_DIR = orig["artifacts"]
    fm.OUTPUTS_CHARTS_DIR = orig["charts"]
    fm.REPORTS_MARKDOWN_DIR = orig["md"]
    fm.REPORTS_HTML_DIR = orig["html"]


# ── 1. Pure report logic ──────────────────────────────────────────────────────

class TestRunReport:

    def test_returns_required_keys(self, report_dirs):
        result = _run_report(_full_state())
        assert "report_markdown_path" in result
        assert "report_html_path" in result

    def test_markdown_path_ends_with_md(self, report_dirs):
        result = _run_report(_full_state())
        assert result["report_markdown_path"].endswith(".md")

    def test_html_path_ends_with_html(self, report_dirs):
        result = _run_report(_full_state())
        assert result["report_html_path"].endswith(".html")

    def test_experiment_id_in_filenames(self, report_dirs):
        result = _run_report(_full_state(exp_id="my-exp-xyz"))
        assert "my-exp-xyz" in result["report_markdown_path"]
        assert "my-exp-xyz" in result["report_html_path"]

    def test_files_exist_after_run(self, report_dirs):
        result = _run_report(_full_state())
        assert Path(result["report_markdown_path"]).exists()
        assert Path(result["report_html_path"]).exists()

    def test_markdown_file_not_empty(self, report_dirs):
        result = _run_report(_full_state())
        content = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert len(content) > 100

    def test_html_file_not_empty(self, report_dirs):
        result = _run_report(_full_state())
        content = Path(result["report_html_path"]).read_text(encoding="utf-8")
        assert len(content) > 200


# ── 2. Markdown content ───────────────────────────────────────────────────────

class TestMarkdownContent:

    def _md(self, state=None) -> str:
        return _render_markdown(state or _full_state())

    def test_contains_experiment_id(self):
        md = self._md(_full_state(exp_id="test-exp-999"))
        assert "test-exp-999" in md

    def test_contains_target_column(self):
        md = self._md()
        assert "target" in md

    def test_contains_task_type(self):
        md = self._md()
        assert "classification" in md

    def test_contains_all_section_headers(self):
        md = self._md()
        for section in (
            "Executive Summary",
            "Dataset Overview",
            "Data Quality",
            "Cleaning Summary",
            "Feature Engineering",
            "Model Training Results",
            "Model Evaluation",
            "Best Model",
            "Pipeline Execution Timeline",
            "Appendix",
        ):
            assert section in md, f"Missing section: {section}"

    def test_contains_best_model_name(self):
        md = self._md()
        assert "xgboost" in md

    def test_contains_best_score(self):
        md = self._md()
        assert "0.86" in md

    def test_contains_all_model_names(self):
        md = self._md()
        for model in ("random_forest", "logistic_regression", "xgboost"):
            assert model in md

    def test_contains_row_counts(self):
        md = self._md()
        assert "150" in md   # rows_before
        assert "145" in md   # rows_after

    def test_contains_missing_column(self):
        md = self._md()
        assert "f2" in md

    def test_contains_chart_paths(self):
        md = self._md()
        assert "distributions/f1.png" in md

    def test_contains_node_names_in_timeline(self):
        md = self._md()
        for node in ("planner", "eda", "cleaning", "training", "evaluation"):
            assert node in md

    def test_ranking_medals_present(self):
        md = self._md()
        assert "🥇" in md
        assert "🥈" in md
        assert "🥉" in md

    def test_empty_state_renders_gracefully(self):
        """Report must not crash on a minimal state."""
        state: WorkflowState = {
            "experiment_id": "empty-exp",
            "task_type": TaskType.CLASSIFICATION,
        }
        md = _render_markdown(state)
        assert "empty-exp" in md
        assert "Executive Summary" in md

    def test_regression_task_type_in_report(self):
        state = _full_state()
        state["task_type"] = TaskType.REGRESSION
        state["evaluation_results"]["primary_metric"] = "rmse"  # type: ignore[index]
        md = _render_markdown(state)
        assert "regression" in md


# ── 3. HTML content ───────────────────────────────────────────────────────────

class TestHTMLContent:

    def _html(self, report_dirs, state=None) -> str:
        result = _run_report(state or _full_state())
        return Path(result["report_html_path"]).read_text(encoding="utf-8")

    def test_html_has_doctype(self, report_dirs):
        content = self._html(report_dirs)
        assert "<!DOCTYPE html>" in content

    def test_html_has_head_and_body(self, report_dirs):
        content = self._html(report_dirs)
        assert "<head>" in content
        assert "<body>" in content

    def test_html_has_title(self, report_dirs):
        content = self._html(report_dirs)
        assert "<title>" in content

    def test_html_has_css(self, report_dirs):
        content = self._html(report_dirs)
        assert "<style>" in content

    def test_html_has_table(self, report_dirs):
        content = self._html(report_dirs)
        assert "<table>" in content

    def test_html_has_footer(self, report_dirs):
        content = self._html(report_dirs)
        assert "<footer>" in content

    def test_html_contains_experiment_id(self, report_dirs):
        state = _full_state(exp_id="html-test-exp")
        content = self._html(report_dirs, state)
        assert "html-test-exp" in content

    def test_html_best_model_row_highlighted(self, report_dirs):
        """Best model row should have class='best'."""
        content = self._html(report_dirs)
        assert 'class="best"' in content

    def test_html_contains_h1(self, report_dirs):
        content = self._html(report_dirs)
        assert "<h1>" in content

    def test_html_contains_h2_sections(self, report_dirs):
        content = self._html(report_dirs)
        assert content.count("<h2>") >= 5


# ── 4. Node wrapper ───────────────────────────────────────────────────────────

class TestReportNode:

    def test_node_result_appended_on_success(self, report_dirs):
        state = _full_state()
        result = report_agent(state)
        assert len(result["node_results"]) == len(state["node_results"]) + 1
        nr = result["node_results"][-1]
        assert nr["node_name"] == "report"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-dur"))
        assert result["node_results"][-1]["duration_seconds"] >= 0

    def test_output_summary_has_paths(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-summary"))
        summary = result["node_results"][-1]["output_summary"]
        assert "markdown_path" in summary
        assert "html_path" in summary

    def test_report_paths_written_to_state(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-paths"))
        assert "report_markdown_path" in result
        assert "report_html_path" in result
        assert result["report_markdown_path"].endswith(".md")
        assert result["report_html_path"].endswith(".html")

    def test_existing_node_results_preserved(self, report_dirs):
        state = _full_state(exp_id="exp-preserve")
        n_before = len(state["node_results"])
        result = report_agent(state)
        assert len(result["node_results"]) == n_before + 1

    def test_node_result_error_on_bad_report_dir(self):
        """If report generation raises, agent should return error state."""
        from unittest.mock import patch
        state = _full_state(exp_id="exp-bad-dir")
        with patch(
            "app.agents.report_agent._run_report",
            side_effect=ReportError("Simulated write failure"),
        ):
            result = report_agent(state)
        nr = result["node_results"][-1]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr


# ── 5. Report files on disk ───────────────────────────────────────────────────

class TestReportFiles:

    def test_markdown_file_exists(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-md-file"))
        assert Path(result["report_markdown_path"]).exists()

    def test_html_file_exists(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-html-file"))
        assert Path(result["report_html_path"]).exists()

    def test_markdown_file_is_utf8(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-utf8"))
        content = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
        assert isinstance(content, str)

    def test_html_file_is_utf8(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-html-utf8"))
        content = Path(result["report_html_path"]).read_text(encoding="utf-8")
        assert isinstance(content, str)

    def test_markdown_file_size_reasonable(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-size"))
        size = Path(result["report_markdown_path"]).stat().st_size
        assert size > 500, f"Markdown too small: {size} bytes"

    def test_html_file_size_reasonable(self, report_dirs):
        result = report_agent(_full_state(exp_id="exp-html-size"))
        size = Path(result["report_html_path"]).stat().st_size
        assert size > 1000, f"HTML too small: {size} bytes"


# ── 6. Full graph integration ─────────────────────────────────────────────────

class TestGraphReport:

    def test_report_node_runs_in_graph(self, all_dirs):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-report-001",
            "dataset_id": "ds-001",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert "report" in node_names
        report_nr = next(
            nr for nr in final["node_results"] if nr["node_name"] == "report"
        )
        assert report_nr["status"] == WorkflowNodeStatus.DONE

    def test_report_paths_in_final_state(self, all_dirs):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-report-002",
            "dataset_id": "ds-002",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "report_markdown_path" in final
        assert "report_html_path" in final
        assert Path(final["report_markdown_path"]).exists()
        assert Path(final["report_html_path"]).exists()

    def test_report_is_last_node(self, all_dirs):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-report-003",
            "dataset_id": "ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert node_names[-1] == "report"

    def test_full_pipeline_no_errors(self, all_dirs):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-report-004",
            "dataset_id": "ds-004",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert final.get("errors", []) == []

    def test_all_seven_nodes_ran(self, all_dirs):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-report-005",
            "dataset_id": "ds-005",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        expected = [
            "planner", "eda", "cleaning",
            "feature_engineering", "training", "evaluation", "report",
        ]
        assert node_names == expected

    def test_report_contains_best_model_from_evaluation(self, all_dirs):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-report-006",
            "dataset_id": "ds-006",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        best = final.get("best_model_name", "")
        md_content = Path(final["report_markdown_path"]).read_text(encoding="utf-8")
        assert best in md_content
