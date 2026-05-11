"""
Phase 3 tests — Planner Agent and LangGraph graph compilation.

Tests are split into three groups:
  1. Unit tests for the planner's pure planning logic (_run_planner).
  2. Node-level tests for the planner_agent() LangGraph callable.
  3. Graph compilation and end-to-end smoke test (no real DB needed).
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agents.planner_agent import planner_agent, _run_planner, PlannerError
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.state.workflow_state import WorkflowState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temp CSV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _classification_df(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "age": range(rows),
        "salary": [float(i) * 1000 for i in range(rows)],
        "department": ["eng" if i % 3 == 0 else "sales" if i % 3 == 1 else "hr"
                       for i in range(rows)],
        "target": [i % 2 for i in range(rows)],
    })


def _regression_df(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": range(rows),
        "feature_b": [float(i) * 0.5 for i in range(rows)],
        "target": [float(i) * 2.5 + 1.0 for i in range(rows)],
    })


def _df_with_missing(rows: int = 50) -> pd.DataFrame:
    import numpy as np
    df = _classification_df(rows)
    df.loc[::3, "salary"] = np.nan   # ~33% missing
    return df


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Ensure temp CSV files created during tests are removed."""
    yield
    # tempfile.NamedTemporaryFile files are cleaned up by the OS on Windows
    # but we explicitly remove them here for safety
    for f in Path(tempfile.gettempdir()).glob("tmp*.csv"):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


# ── 1. Pure planning logic ────────────────────────────────────────────────────

class TestRunPlanner:
    def test_classification_task_type_inferred(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
        }
        result = _run_planner(state)
        assert result["plan"]["task_type"] == "classification"
        assert result["task_type"] == TaskType.CLASSIFICATION

    def test_regression_task_type_inferred(self):
        path = _write_csv(_regression_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
        }
        result = _run_planner(state)
        assert result["plan"]["task_type"] == "regression"
        assert result["task_type"] == TaskType.REGRESSION

    def test_explicit_task_type_honoured(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.REGRESSION,  # override heuristic
        }
        result = _run_planner(state)
        assert result["plan"]["task_type"] == "regression"

    def test_plan_contains_all_required_keys(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
        }
        result = _run_planner(state)
        plan = result["plan"]
        for key in (
            "task_type", "target_column", "feature_columns",
            "numeric_features", "categorical_features", "models",
            "dataset_shape", "data_quality", "column_stats", "steps",
        ):
            assert key in plan, f"Missing plan key: {key}"

    def test_plan_steps_are_ordered(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        orders = [s["order"] for s in result["plan"]["steps"]]
        assert orders == sorted(orders)

    def test_plan_steps_cover_full_pipeline(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        agents = {s["agent"] for s in result["plan"]["steps"]}
        expected = {"eda", "cleaning", "feature_engineering", "training", "evaluation", "report"}
        assert agents == expected

    def test_feature_columns_exclude_target(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        assert "target" not in result["plan"]["feature_columns"]

    def test_dataset_shape_correct(self):
        df = _classification_df(rows=42)
        path = _write_csv(df)
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        assert result["plan"]["dataset_shape"]["rows"] == 42
        assert result["plan"]["dataset_shape"]["columns"] == 4

    def test_missing_columns_detected(self):
        path = _write_csv(_df_with_missing())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        assert "salary" in result["plan"]["data_quality"]["missing_columns"]

    def test_categorical_features_detected(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        assert "department" in result["plan"]["categorical_features"]

    def test_pipeline_config_model_override(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {"models": ["xgboost"]},
        }
        result = _run_planner(state)
        assert result["plan"]["models"] == ["xgboost"]

    def test_default_classification_models(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = _run_planner(state)
        assert "random_forest" in result["plan"]["models"]
        assert "xgboost" in result["plan"]["models"]

    def test_raises_if_dataset_not_found(self):
        state: WorkflowState = {
            "dataset_path": "/nonexistent/path/data.csv",
            "target_column": "target",
        }
        with pytest.raises(PlannerError, match="not found"):
            _run_planner(state)

    def test_raises_if_target_column_missing(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "nonexistent_column",
        }
        with pytest.raises(PlannerError, match="not found in dataset"):
            _run_planner(state)

    def test_raises_if_target_column_empty(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": ""}
        with pytest.raises(PlannerError, match="required"):
            _run_planner(state)


# ── 2. Node-level (planner_agent callable) ────────────────────────────────────

class TestPlannerNode:
    def test_successful_run_returns_plan(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = planner_agent(state)
        assert "plan" in result
        assert result["plan"]["task_type"] == "classification"

    def test_node_result_appended_on_success(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = planner_agent(state)
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["node_name"] == "planner"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self):
        path = _write_csv(_classification_df())
        state: WorkflowState = {"dataset_path": path, "target_column": "target"}
        result = planner_agent(state)
        assert result["node_results"][0]["duration_seconds"] >= 0

    def test_node_result_appended_on_failure(self):
        state: WorkflowState = {
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
        }
        result = planner_agent(state)
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr

    def test_errors_list_populated_on_failure(self):
        state: WorkflowState = {
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
        }
        result = planner_agent(state)
        assert len(result["errors"]) > 0

    def test_existing_node_results_preserved(self):
        """Planner should append to, not replace, existing node_results."""
        path = _write_csv(_classification_df())
        from app.enums.workflow_status import WorkflowNodeStatus
        prior: Any = [{"node_name": "init", "status": WorkflowNodeStatus.DONE}]
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "node_results": prior,
        }
        result = planner_agent(state)
        assert len(result["node_results"]) == 2
        assert result["node_results"][0]["node_name"] == "init"
        assert result["node_results"][1]["node_name"] == "planner"


# ── 3. Graph compilation and smoke test ───────────────────────────────────────

class TestGraph:
    def test_graph_compiles_without_error(self):
        """Importing the graph module should not raise."""
        from app.workflows.graph import pipeline_graph
        assert pipeline_graph is not None

    def test_graph_invoke_success_path(self):
        """Full graph invocation with a valid dataset should reach END."""
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "test-exp-001",
            "dataset_id": "test-ds-001",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)

        # Planner ran and produced a plan
        assert "plan" in final
        assert final["plan"]["task_type"] == "classification"

        # All nodes ran (planner + 6 stubs)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert "planner" in node_names
        assert "eda" in node_names
        assert "cleaning" in node_names

    def test_graph_invoke_error_path(self):
        """Graph should route to handle_error when planner fails."""
        from app.workflows.graph import pipeline_graph
        initial: WorkflowState = {
            "experiment_id": "test-exp-002",
            "dataset_id": "test-ds-002",
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)

        # Errors should be populated
        assert len(final.get("errors", [])) > 0

        # Should NOT have proceeded to EDA
        node_names = [nr["node_name"] for nr in final.get("node_results", [])]
        assert "eda" not in node_names

    def test_graph_node_results_ordered(self):
        """Node results should appear in pipeline order."""
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "test-exp-003",
            "dataset_id": "test-ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        expected_order = ["planner", "eda", "cleaning", "feature_engineering",
                          "training", "evaluation", "report"]
        assert node_names == expected_order
