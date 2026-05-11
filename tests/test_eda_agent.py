"""
Phase 4 tests — EDA Agent.

Groups:
  1. TestRunEDA        — pure _run_eda() logic (stats, structure, chart paths)
  2. TestEDANode       — eda_agent() node wrapper (state patching, error handling)
  3. TestChartFiles    — charts are actually written to disk with correct names
  4. TestGraphEDA      — full graph invocation confirms EDA node runs and produces output
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agents.eda_agent import eda_agent, _run_eda, EDAError
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


def _mixed_df(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":        rng.integers(18, 70, rows).astype(float),
        "salary":     rng.uniform(30_000, 120_000, rows),
        "score":      rng.uniform(0, 100, rows),
        "department": rng.choice(["eng", "sales", "hr"], rows),
        "region":     rng.choice(["north", "south", "east", "west"], rows),
        "target":     rng.integers(0, 2, rows),
    })


def _regression_df(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, rows),
        "x2": rng.uniform(-5, 5, rows),
        "target": rng.uniform(0, 100, rows),   # continuous → regression
    })


def _df_with_missing(rows: int = 60) -> pd.DataFrame:
    df = _mixed_df(rows)
    df.loc[::4, "salary"] = np.nan
    df.loc[::6, "department"] = np.nan
    return df


@pytest.fixture()
def chart_output_dir(tmp_path):
    """Provide a temp directory and patch OUTPUTS_CHARTS_DIR."""
    import app.utils.file_manager as fm
    original = fm.OUTPUTS_CHARTS_DIR
    fm.OUTPUTS_CHARTS_DIR = tmp_path
    yield tmp_path
    fm.OUTPUTS_CHARTS_DIR = original


# ── 1. Pure EDA logic ─────────────────────────────────────────────────────────

class TestRunEDA:
    def test_returns_eda_summary_and_chart_paths(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-001",
        }
        result = _run_eda(state)
        assert "eda_summary" in result
        assert "chart_paths" in result

    def test_shape_correct(self, chart_output_dir):
        df = _mixed_df(rows=50)
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-shape",
        }
        result = _run_eda(state)
        assert result["eda_summary"]["shape"]["rows"] == 50
        assert result["eda_summary"]["shape"]["columns"] == 6

    def test_numeric_columns_detected(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-num",
        }
        result = _run_eda(state)
        numeric = result["eda_summary"]["numeric_columns"]
        assert "age" in numeric
        assert "salary" in numeric
        assert "score" in numeric

    def test_categorical_columns_detected(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-cat",
        }
        result = _run_eda(state)
        cats = result["eda_summary"]["categorical_columns"]
        assert "department" in cats
        assert "region" in cats

    def test_missing_columns_detected(self, chart_output_dir):
        path = _write_csv(_df_with_missing())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-miss",
        }
        result = _run_eda(state)
        missing_cols = result["eda_summary"]["missing_columns"]
        assert "salary" in missing_cols

    def test_missing_report_has_pct(self, chart_output_dir):
        path = _write_csv(_df_with_missing())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-misspct",
        }
        result = _run_eda(state)
        report = result["eda_summary"]["missing_report"]
        assert "salary" in report
        assert "pct" in report["salary"]
        assert report["salary"]["pct"] > 0

    def test_correlation_matrix_present_for_numeric(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-corr",
        }
        result = _run_eda(state)
        corr = result["eda_summary"]["correlation_matrix"]
        assert isinstance(corr, dict)
        assert len(corr) > 0

    def test_correlation_values_in_range(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-corrval",
        }
        result = _run_eda(state)
        corr = result["eda_summary"]["correlation_matrix"]
        for col, row in corr.items():
            for other, val in row.items():
                if val is not None:
                    assert -1.0 <= val <= 1.0, f"Correlation out of range: {col}/{other}={val}"

    def test_target_analysis_classification(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-tgt-cls",
        }
        result = _run_eda(state)
        ta = result["eda_summary"]["target_analysis"]
        assert ta["type"] == "numeric"   # target is 0/1 int → numeric dtype

    def test_target_analysis_regression(self, chart_output_dir):
        path = _write_csv(_regression_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-tgt-reg",
        }
        result = _run_eda(state)
        ta = result["eda_summary"]["target_analysis"]
        assert ta["type"] == "numeric"
        assert "mean" in ta
        assert "std" in ta

    def test_column_stats_present(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-stats",
        }
        result = _run_eda(state)
        stats = result["eda_summary"]["column_stats"]
        assert "age" in stats
        assert stats["age"]["type"] == "numeric"
        assert "mean" in stats["age"]

    def test_raises_if_dataset_not_found(self, chart_output_dir):
        state: WorkflowState = {
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "experiment_id": "exp-err",
        }
        with pytest.raises(EDAError, match="not found"):
            _run_eda(state)

    def test_raises_if_target_column_missing(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "nonexistent",
            "experiment_id": "exp-badtgt",
        }
        with pytest.raises(EDAError, match="not found in dataset"):
            _run_eda(state)

    def test_works_without_target_column(self, chart_output_dir):
        """EDA should run even if target_column is empty (unsupervised use case)."""
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "",
            "experiment_id": "exp-notgt",
        }
        result = _run_eda(state)
        assert "eda_summary" in result


# ── 2. Node wrapper ───────────────────────────────────────────────────────────

class TestEDANode:
    def test_node_result_appended_on_success(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-node-ok",
            "node_results": [],
            "errors": [],
        }
        result = eda_agent(state)
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["node_name"] == "eda"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-dur",
            "node_results": [],
        }
        result = eda_agent(state)
        assert result["node_results"][0]["duration_seconds"] >= 0

    def test_output_summary_has_chart_count(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-chartcount",
            "node_results": [],
        }
        result = eda_agent(state)
        summary = result["node_results"][0]["output_summary"]
        assert "num_charts" in summary
        assert summary["num_charts"] > 0

    def test_node_result_error_on_bad_path(self, chart_output_dir):
        state: WorkflowState = {
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
            "experiment_id": "exp-node-err",
            "node_results": [],
            "errors": [],
        }
        result = eda_agent(state)
        nr = result["node_results"][0]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr

    def test_errors_list_populated_on_failure(self, chart_output_dir):
        state: WorkflowState = {
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
            "experiment_id": "exp-errlist",
            "node_results": [],
            "errors": [],
        }
        result = eda_agent(state)
        assert len(result["errors"]) > 0

    def test_existing_node_results_preserved(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        prior: Any = [{"node_name": "planner", "status": WorkflowNodeStatus.DONE}]
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-preserve",
            "node_results": prior,
        }
        result = eda_agent(state)
        assert len(result["node_results"]) == 2
        assert result["node_results"][0]["node_name"] == "planner"
        assert result["node_results"][1]["node_name"] == "eda"

    def test_eda_summary_written_to_state(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-summary",
            "node_results": [],
        }
        result = eda_agent(state)
        assert "eda_summary" in result
        assert result["eda_summary"]["shape"]["rows"] == 80

    def test_chart_paths_written_to_state(self, chart_output_dir):
        path = _write_csv(_mixed_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-chartpaths",
            "node_results": [],
        }
        result = eda_agent(state)
        assert isinstance(result["chart_paths"], list)
        assert len(result["chart_paths"]) > 0


# ── 3. Chart files on disk ────────────────────────────────────────────────────

class TestChartFiles:
    def _run(self, chart_output_dir, experiment_id: str, df=None, missing=False):
        """Helper: run eda_agent and return result."""
        if df is None:
            df = _df_with_missing() if missing else _mixed_df()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": experiment_id,
            "node_results": [],
        }
        return eda_agent(state)

    def test_distribution_charts_created(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-dist")
        dist_paths = [p for p in result["chart_paths"] if "distributions" in p]
        assert len(dist_paths) > 0
        for p in dist_paths:
            assert Path(p).exists(), f"Distribution chart missing: {p}"

    def test_boxplot_charts_created(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-box")
        box_paths = [p for p in result["chart_paths"] if "boxplots" in p]
        assert len(box_paths) > 0
        for p in box_paths:
            assert Path(p).exists(), f"Boxplot chart missing: {p}"

    def test_categorical_bar_charts_created(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-catbar")
        cat_paths = [p for p in result["chart_paths"] if "categorical" in p]
        assert len(cat_paths) > 0
        for p in cat_paths:
            assert Path(p).exists(), f"Categorical chart missing: {p}"

    def test_correlation_heatmap_created(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-heatmap")
        heatmap_paths = [p for p in result["chart_paths"]
                         if Path(p).name == "correlation_heatmap.png"]
        assert len(heatmap_paths) == 1
        assert Path(heatmap_paths[0]).exists()

    def test_target_distribution_chart_created(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-tgtdist")
        tgt_paths = [p for p in result["chart_paths"]
                     if Path(p).name == "target_distribution.png"]
        assert len(tgt_paths) == 1
        assert Path(tgt_paths[0]).exists()

    def test_missing_value_chart_created(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-missplot", missing=True)
        miss_paths = [p for p in result["chart_paths"] if "missing_values" in p]
        assert len(miss_paths) == 1
        assert Path(miss_paths[0]).exists()

    def test_all_chart_paths_exist_on_disk(self, chart_output_dir):
        result = self._run(chart_output_dir, "exp-allpaths")
        for chart_path in result["chart_paths"]:
            assert Path(chart_path).exists(), f"Chart not found: {chart_path}"

    def test_no_missing_chart_when_no_missing_data(self, chart_output_dir):
        """Missing-value chart should NOT be created when data is complete."""
        df = _mixed_df()
        assert df.isna().sum().sum() == 0
        result = self._run(chart_output_dir, "exp-nomiss", df=df)
        miss_paths = [p for p in result["chart_paths"] if "missing_values" in p]
        assert len(miss_paths) == 0


# ── 4. Full graph integration ─────────────────────────────────────────────────

class TestGraphEDA:
    def test_eda_node_runs_in_graph(self, chart_output_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_mixed_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eda-001",
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
        assert "eda" in node_names

        eda_nr = next(nr for nr in final["node_results"] if nr["node_name"] == "eda")
        assert eda_nr["status"] == WorkflowNodeStatus.DONE

    def test_eda_summary_in_final_state(self, chart_output_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_mixed_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eda-002",
            "dataset_id": "ds-002",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)

        assert "eda_summary" in final
        assert final["eda_summary"]["shape"]["rows"] == 80
        assert len(final.get("chart_paths", [])) > 0

    def test_pipeline_order_planner_then_eda(self, chart_output_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_mixed_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eda-003",
            "dataset_id": "ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert node_names.index("planner") < node_names.index("eda")
