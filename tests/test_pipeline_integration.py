"""
Phase 10 — Full LangGraph pipeline integration tests.

These tests exercise the complete pipeline end-to-end:
  planner → eda → cleaning → feature_engineering → training → evaluation → report

Groups:
  1. TestGraphStructure      — graph schema, node registration, edge topology
  2. TestFullPipelineRun     — end-to-end invocation with real data
  3. TestErrorPropagation    — error routing and state integrity
  4. TestStateContract       — WorkflowState fields populated correctly
  5. TestPipelineService     — PipelineService orchestration (mocked DB)
  6. TestGraphSchema         — get_graph_schema() output
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus, ExperimentStatus
from app.state.workflow_state import WorkflowState
from app.workflows.graph import build_pipeline_graph, get_graph_schema, pipeline_graph


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
        "age":    rng.uniform(18, 70, rows),
        "salary": rng.uniform(30_000, 120_000, rows),
        "score":  rng.uniform(0, 100, rows),
        "dept":   rng.choice(["eng", "sales", "hr"], rows),
        "target": rng.integers(0, 2, rows),
    })


def _regression_df(rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, rows),
        "x2": rng.uniform(-5, 5, rows),
        "x3": rng.uniform(0, 1, rows),
        "cat": rng.choice(["a", "b", "c"], rows),
        "target": rng.uniform(0, 100, rows),
    })


@pytest.fixture()
def all_dirs(tmp_path):
    """Redirect all output directories to tmp_path for isolation."""
    import app.utils.file_manager as fm
    orig = {
        "models":    fm.OUTPUTS_MODELS_DIR,
        "artifacts": fm.OUTPUTS_ARTIFACTS_DIR,
        "charts":    fm.OUTPUTS_CHARTS_DIR,
        "md":        fm.REPORTS_MARKDOWN_DIR,
        "html":      fm.REPORTS_HTML_DIR,
    }
    fm.OUTPUTS_MODELS_DIR    = tmp_path / "models"
    fm.OUTPUTS_ARTIFACTS_DIR = tmp_path / "artifacts"
    fm.OUTPUTS_CHARTS_DIR    = tmp_path / "charts"
    fm.REPORTS_MARKDOWN_DIR  = tmp_path / "markdown"
    fm.REPORTS_HTML_DIR      = tmp_path / "html"
    yield tmp_path
    for key, val in orig.items():
        setattr(fm, {
            "models": "OUTPUTS_MODELS_DIR",
            "artifacts": "OUTPUTS_ARTIFACTS_DIR",
            "charts": "OUTPUTS_CHARTS_DIR",
            "md": "REPORTS_MARKDOWN_DIR",
            "html": "REPORTS_HTML_DIR",
        }[key], val)


def _initial_state(path: str, task_type=TaskType.CLASSIFICATION,
                   exp_id: str = "integ-001") -> WorkflowState:
    return {
        "experiment_id": exp_id,
        "dataset_id": "ds-001",
        "dataset_path": path,
        "target_column": "target",
        "task_type": task_type,
        "pipeline_config": {},
        "node_results": [],
        "errors": [],
    }


# ── 1. Graph structure ────────────────────────────────────────────────────────

class TestGraphStructure:

    def test_graph_compiles_without_error(self):
        g = build_pipeline_graph()
        assert g is not None

    def test_module_level_singleton_exists(self):
        assert pipeline_graph is not None

    def test_graph_schema_has_all_nodes(self):
        schema = get_graph_schema()
        node_names = {n["name"] for n in schema["nodes"]}
        expected = {
            "planner", "eda", "cleaning", "feature_engineering",
            "training", "evaluation", "report", "handle_error",
        }
        assert node_names == expected

    def test_graph_schema_has_edges(self):
        schema = get_graph_schema()
        assert len(schema["edges"]) >= 9

    def test_graph_schema_entry_point(self):
        schema = get_graph_schema()
        assert schema["entry_point"] == "planner"

    def test_graph_schema_terminal_nodes(self):
        schema = get_graph_schema()
        assert "report" in schema["terminal_nodes"]
        assert "handle_error" in schema["terminal_nodes"]

    def test_graph_schema_parallel_nodes(self):
        schema = get_graph_schema()
        assert "training" in schema["parallel_nodes"]

    def test_graph_schema_phases_assigned(self):
        schema = get_graph_schema()
        phase_map = {n["name"]: n["phase"] for n in schema["nodes"]}
        assert phase_map["planner"] == 3
        assert phase_map["eda"] == 4
        assert phase_map["cleaning"] == 5
        assert phase_map["feature_engineering"] == 6
        assert phase_map["training"] == 7
        assert phase_map["evaluation"] == 8
        assert phase_map["report"] == 9

    def test_conditional_edge_after_planner(self):
        schema = get_graph_schema()
        planner_edges = [
            e for e in schema["edges"] if e["from"] == "planner"
        ]
        conditions = {e["condition"] for e in planner_edges}
        assert "no errors" in conditions
        assert "errors present" in conditions


# ── 2. Full pipeline run ──────────────────────────────────────────────────────

class TestFullPipelineRun:

    def test_classification_pipeline_completes(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path))
        assert final.get("errors", []) == []

    def test_regression_pipeline_completes(self, all_dirs):
        path = _write_csv(_regression_df())
        state = _initial_state(path, task_type=TaskType.REGRESSION, exp_id="integ-reg")
        final = pipeline_graph.invoke(state)
        assert final.get("errors", []) == []

    def test_all_seven_nodes_execute(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-7nodes"))
        node_names = [nr["node_name"] for nr in final["node_results"]]
        expected = [
            "planner", "eda", "cleaning",
            "feature_engineering", "training", "evaluation", "report",
        ]
        assert node_names == expected

    def test_all_nodes_have_done_status(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-status"))
        for nr in final["node_results"]:
            assert nr["status"] == WorkflowNodeStatus.DONE, (
                f"Node '{nr['node_name']}' has status {nr['status']}"
            )

    def test_report_is_last_node(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-last"))
        assert final["node_results"][-1]["node_name"] == "report"

    def test_planner_is_first_node(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-first"))
        assert final["node_results"][0]["node_name"] == "planner"

    def test_custom_model_list_respected(self, all_dirs):
        path = _write_csv(_classification_df())
        state = _initial_state(path, exp_id="integ-custom-models")
        state["pipeline_config"] = {"training": {"models": ["random_forest"]}}
        final = pipeline_graph.invoke(state)
        assert "random_forest" in final["trained_model_paths"]
        assert "xgboost" not in final["trained_model_paths"]

    def test_regression_uses_correct_models(self, all_dirs):
        path = _write_csv(_regression_df())
        state = _initial_state(path, task_type=TaskType.REGRESSION, exp_id="integ-reg-models")
        final = pipeline_graph.invoke(state)
        trained = set(final["trained_model_paths"].keys())
        assert "random_forest" in trained
        assert "linear_regression" in trained
        assert "logistic_regression" not in trained


# ── 3. Error propagation ──────────────────────────────────────────────────────

class TestErrorPropagation:

    def test_bad_dataset_path_routes_to_handle_error(self, all_dirs):
        state: WorkflowState = {
            "experiment_id": "integ-err-001",
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(state)
        assert len(final.get("errors", [])) > 0

    def test_bad_path_skips_all_downstream_nodes(self, all_dirs):
        state: WorkflowState = {
            "experiment_id": "integ-err-002",
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(state)
        node_names = [nr["node_name"] for nr in final.get("node_results", [])]
        # EDA and beyond must NOT have run
        assert "eda" not in node_names
        assert "training" not in node_names
        assert "report" not in node_names

    def test_missing_target_column_routes_to_handle_error(self, all_dirs):
        path = _write_csv(_classification_df())
        state: WorkflowState = {
            "experiment_id": "integ-err-003",
            "dataset_path": path,
            "target_column": "nonexistent_column",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(state)
        assert len(final.get("errors", [])) > 0

    def test_errors_list_contains_descriptive_message(self, all_dirs):
        state: WorkflowState = {
            "experiment_id": "integ-err-004",
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(state)
        errors = final.get("errors", [])
        assert len(errors) > 0
        assert isinstance(errors[0], str)
        assert len(errors[0]) > 5   # not an empty string

    def test_planner_error_node_result_has_error_status(self, all_dirs):
        state: WorkflowState = {
            "experiment_id": "integ-err-005",
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(state)
        planner_nr = next(
            (nr for nr in final.get("node_results", [])
             if nr["node_name"] == "planner"),
            None,
        )
        assert planner_nr is not None
        assert planner_nr["status"] == WorkflowNodeStatus.ERROR


# ── 4. State contract ─────────────────────────────────────────────────────────

class TestStateContract:

    def test_plan_populated_after_planner(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-plan"))
        assert "plan" in final
        assert "task_type" in final["plan"]
        assert "steps" in final["plan"]

    def test_eda_summary_populated(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-eda"))
        assert "eda_summary" in final
        assert final["eda_summary"]["shape"]["rows"] == 150

    def test_cleaned_dataset_path_is_valid_file(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-clean"))
        cleaned = final.get("cleaned_dataset_path", "")
        assert cleaned.endswith("cleaned_dataset.csv")
        assert Path(cleaned).exists()

    def test_feature_dataset_path_is_valid_file(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-feat"))
        feat = final.get("feature_dataset_path", "")
        assert feat.endswith("feature_dataset.csv")
        assert Path(feat).exists()

    def test_selected_features_is_non_empty_list(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-sel"))
        assert isinstance(final.get("selected_features"), list)
        assert len(final["selected_features"]) > 0

    def test_trained_model_paths_all_exist(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-models"))
        for name, model_path in final["trained_model_paths"].items():
            assert Path(model_path).exists(), f"{name} pkl missing"

    def test_training_metrics_all_models_present(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-metrics"))
        metrics = final.get("training_metrics", {})
        assert len(metrics) == 3
        for name, m in metrics.items():
            assert "accuracy" in m or "rmse" in m, f"{name} missing primary metric"

    def test_best_model_name_is_valid(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-best"))
        best = final.get("best_model_name", "")
        assert best in {"random_forest", "logistic_regression", "xgboost"}

    def test_best_model_path_exists(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-bestpath"))
        best_path = final.get("best_model_path", "")
        assert best_path != ""
        assert Path(best_path).exists()

    def test_evaluation_results_has_ranked_models(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-eval"))
        ev = final.get("evaluation_results", {})
        assert "ranked_models" in ev
        assert len(ev["ranked_models"]) == 3

    def test_report_markdown_path_exists(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-md"))
        md = final.get("report_markdown_path", "")
        assert md.endswith(".md")
        assert Path(md).exists()

    def test_report_html_path_exists(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-html"))
        html = final.get("report_html_path", "")
        assert html.endswith(".html")
        assert Path(html).exists()

    def test_chart_paths_all_exist(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-charts"))
        for cp in final.get("chart_paths", []):
            assert Path(cp).exists(), f"Chart missing: {cp}"

    def test_node_results_count_is_seven(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-count"))
        assert len(final["node_results"]) == 7

    def test_each_node_result_has_duration(self, all_dirs):
        path = _write_csv(_classification_df())
        final = pipeline_graph.invoke(_initial_state(path, exp_id="integ-dur"))
        for nr in final["node_results"]:
            assert "duration_seconds" in nr
            assert nr["duration_seconds"] >= 0


# ── 5. PipelineService orchestration ─────────────────────────────────────────

class TestPipelineService:
    """
    Tests PipelineService with a mocked DB session.
    The graph itself is also mocked to keep these tests fast and focused
    on the service's orchestration logic.
    """

    def _make_service(self):
        """Build a PipelineService with a fully mocked DB session."""
        from app.services.pipeline_service import PipelineService
        db = MagicMock()
        service = PipelineService(db)
        return service

    def _mock_dataset(self, file_path: str = "/tmp/data.csv"):
        ds = MagicMock()
        ds.file_path = file_path
        return ds

    def _mock_experiment(self, exp_id: str = "exp-svc-001"):
        exp = MagicMock()
        exp.id = exp_id
        exp.status = ExperimentStatus.COMPLETED
        exp.best_model = "xgboost"
        exp.metrics = {"best_score": 0.85}
        exp.report_path = "/reports/markdown/exp-svc-001.md"
        exp.started_at = None
        exp.completed_at = None
        exp.error_message = None
        return exp

    def test_run_creates_experiment_and_returns_response(self, all_dirs):
        from app.services.pipeline_service import PipelineService
        from app.schemas.workflow import WorkflowRunRequest

        service = self._make_service()
        mock_ds = self._mock_dataset()
        mock_exp = self._mock_experiment()

        service._ds_service.get_by_id = MagicMock(return_value=mock_ds)
        service._exp_service.create = MagicMock(return_value=mock_exp)
        service._exp_service.mark_running = MagicMock(return_value=mock_exp)
        service._exp_service.mark_completed = MagicMock(return_value=mock_exp)
        service._exp_service.get_by_id = MagicMock(return_value=mock_exp)

        mock_final_state = {
            "errors": [],
            "node_results": [
                {"node_name": "planner", "status": WorkflowNodeStatus.DONE,
                 "duration_seconds": 0.1},
            ],
            "best_model_name": "xgboost",
            "evaluation_results": {"best_score": 0.85},
            "report_markdown_path": "/reports/markdown/exp-svc-001.md",
        }

        with patch.object(PipelineService, "_invoke_graph",
                          return_value=mock_final_state):
            request = WorkflowRunRequest(
                dataset_id="ds-001",
                target_column="target",
                task_type=TaskType.CLASSIFICATION,
                experiment_name="test_run",
            )
            response = service.run(request)

        assert response.experiment_id == mock_exp.id
        assert response.status == ExperimentStatus.COMPLETED

    def test_run_marks_failed_on_graph_exception(self, all_dirs):
        from app.services.pipeline_service import PipelineService, PipelineServiceError
        from app.schemas.workflow import WorkflowRunRequest

        service = self._make_service()
        mock_ds = self._mock_dataset()
        mock_exp = self._mock_experiment()

        service._ds_service.get_by_id = MagicMock(return_value=mock_ds)
        service._exp_service.create = MagicMock(return_value=mock_exp)
        service._exp_service.mark_running = MagicMock(return_value=mock_exp)
        service._exp_service.mark_failed = MagicMock(return_value=mock_exp)

        with patch.object(PipelineService, "_invoke_graph",
                          side_effect=RuntimeError("Graph crashed")):
            request = WorkflowRunRequest(
                dataset_id="ds-001",
                target_column="target",
                task_type=TaskType.CLASSIFICATION,
                experiment_name="test_fail",
            )
            with pytest.raises(PipelineServiceError, match="Graph crashed"):
                service.run(request)

        service._exp_service.mark_failed.assert_called_once()

    def test_run_marks_failed_on_pipeline_errors(self, all_dirs):
        from app.services.pipeline_service import PipelineService
        from app.schemas.workflow import WorkflowRunRequest

        service = self._make_service()
        mock_ds = self._mock_dataset()
        mock_exp = self._mock_experiment()
        mock_exp.status = ExperimentStatus.FAILED

        service._ds_service.get_by_id = MagicMock(return_value=mock_ds)
        service._exp_service.create = MagicMock(return_value=mock_exp)
        service._exp_service.mark_running = MagicMock(return_value=mock_exp)
        service._exp_service.mark_failed = MagicMock(return_value=mock_exp)

        mock_final_state = {
            "errors": ["Planner failed: dataset not found"],
            "node_results": [
                {"node_name": "planner", "status": WorkflowNodeStatus.ERROR,
                 "duration_seconds": 0.05, "error": "dataset not found"},
            ],
        }

        with patch.object(PipelineService, "_invoke_graph",
                          return_value=mock_final_state):
            request = WorkflowRunRequest(
                dataset_id="ds-001",
                target_column="target",
                task_type=TaskType.CLASSIFICATION,
                experiment_name="test_fail_soft",
            )
            response = service.run(request)

        service._exp_service.mark_failed.assert_called_once()
        assert response.status == ExperimentStatus.FAILED

    def test_dataset_not_found_raises_pipeline_error(self):
        from app.services.pipeline_service import PipelineService, PipelineServiceError
        from app.schemas.workflow import WorkflowRunRequest
        from app.services.dataset_service import DatasetNotFoundError

        service = self._make_service()
        service._ds_service.get_by_id = MagicMock(
            side_effect=DatasetNotFoundError("Dataset not found")
        )

        request = WorkflowRunRequest(
            dataset_id="bad-id",
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            experiment_name="test",
        )
        with pytest.raises(PipelineServiceError, match="Dataset not found"):
            service.run(request)

    def test_get_experiment_status_returns_dict(self):
        from app.services.pipeline_service import PipelineService

        service = self._make_service()
        mock_exp = self._mock_experiment()
        service._exp_service.get_by_id = MagicMock(return_value=mock_exp)

        status = service.get_experiment_status("exp-svc-001")
        assert "experiment_id" in status
        assert "status" in status
        assert "best_model" in status


# ── 6. Graph schema ───────────────────────────────────────────────────────────

class TestGraphSchema:

    def test_schema_is_dict(self):
        schema = get_graph_schema()
        assert isinstance(schema, dict)

    def test_schema_has_required_top_level_keys(self):
        schema = get_graph_schema()
        for key in ("nodes", "edges", "parallel_nodes",
                    "entry_point", "terminal_nodes"):
            assert key in schema, f"Missing key: {key}"

    def test_each_node_has_name_phase_description(self):
        schema = get_graph_schema()
        for node in schema["nodes"]:
            assert "name" in node
            assert "phase" in node
            assert "description" in node

    def test_each_edge_has_from_to_condition(self):
        schema = get_graph_schema()
        for edge in schema["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert "condition" in edge

    def test_schema_is_json_serialisable(self):
        import json
        schema = get_graph_schema()
        # Should not raise
        serialised = json.dumps(schema)
        assert len(serialised) > 100
