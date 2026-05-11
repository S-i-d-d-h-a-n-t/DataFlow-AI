"""
Phase 7 tests — Training Agent.

Groups:
  1. TestRunTraining        — pure _run_training() logic
  2. TestTrainSingleModel   — train_single_model() helper per model/task
  3. TestTrainingNode       — training_agent() node wrapper (state, errors)
  4. TestModelArtifacts     — .pkl files written correctly and loadable
  5. TestParallelExecution  — multiple models trained, all results collected
  6. TestGraphTraining      — full graph confirms training node runs in order
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agents.training_agent import (
    training_agent,
    _run_training,
    train_single_model,
    TrainingError,
)
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


def _regression_df(rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "f1": rng.uniform(0, 10, rows),
        "f2": rng.uniform(-5, 5, rows),
        "f3": rng.uniform(0, 1, rows),
        "target": rng.uniform(0, 100, rows),
    })


def _multiclass_df(rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "f1": rng.uniform(0, 1, rows),
        "f2": rng.uniform(0, 1, rows),
        "target": rng.integers(0, 3, rows),   # 3 classes
    })


@pytest.fixture()
def model_dir(tmp_path):
    """Patch OUTPUTS_MODELS_DIR to a temp directory."""
    import app.utils.file_manager as fm
    original = fm.OUTPUTS_MODELS_DIR
    fm.OUTPUTS_MODELS_DIR = tmp_path
    yield tmp_path
    fm.OUTPUTS_MODELS_DIR = original


def _base_state(
    path: str,
    task_type: TaskType = TaskType.CLASSIFICATION,
    exp_id: str = "exp-001",
    models: list[str] | None = None,
) -> WorkflowState:
    state: WorkflowState = {
        "feature_dataset_path": path,
        "target_column": "target",
        "task_type": task_type,
        "experiment_id": exp_id,
        "selected_features": ["f1", "f2", "f3", "f4"]
        if task_type == TaskType.CLASSIFICATION
        else ["f1", "f2", "f3"],
        "node_results": [],
        "errors": [],
    }
    if models is not None:
        state["pipeline_config"] = {"training": {"models": models}}
    return state


# ── 1. Pure training logic ────────────────────────────────────────────────────

class TestRunTraining:

    def test_returns_required_keys(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(_base_state(path))
        assert "trained_model_paths" in result
        assert "training_metrics" in result

    def test_all_default_classification_models_trained(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(_base_state(path))
        trained = result["trained_model_paths"]
        assert "random_forest" in trained
        assert "logistic_regression" in trained
        assert "xgboost" in trained

    def test_all_default_regression_models_trained(self, model_dir):
        path = _write_csv(_regression_df())
        result = _run_training(
            _base_state(path, task_type=TaskType.REGRESSION, exp_id="exp-reg")
        )
        trained = result["trained_model_paths"]
        assert "random_forest" in trained
        assert "linear_regression" in trained
        assert "xgboost" in trained

    def test_classification_metrics_present(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(_base_state(path))
        for name, metrics in result["training_metrics"].items():
            assert "accuracy" in metrics, f"{name} missing accuracy"
            assert "f1_weighted" in metrics, f"{name} missing f1_weighted"

    def test_regression_metrics_present(self, model_dir):
        path = _write_csv(_regression_df())
        result = _run_training(
            _base_state(path, task_type=TaskType.REGRESSION, exp_id="exp-reg-m")
        )
        for name, metrics in result["training_metrics"].items():
            assert "rmse" in metrics, f"{name} missing rmse"
            assert "r2" in metrics, f"{name} missing r2"

    def test_metrics_include_timing(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(_base_state(path, exp_id="exp-timing"))
        for name, metrics in result["training_metrics"].items():
            assert "train_duration_seconds" in metrics, f"{name} missing timing"
            assert metrics["train_duration_seconds"] >= 0

    def test_metrics_include_split_sizes(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(_base_state(path, exp_id="exp-split"))
        for name, metrics in result["training_metrics"].items():
            assert "n_train" in metrics
            assert "n_test" in metrics
            assert metrics["n_train"] > metrics["n_test"]

    def test_custom_model_list_via_config(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(
            _base_state(path, exp_id="exp-custom", models=["random_forest"])
        )
        assert "random_forest" in result["trained_model_paths"]
        assert "xgboost" not in result["trained_model_paths"]

    def test_logistic_regression_skipped_for_regression(self, model_dir):
        """logistic_regression is classification-only — must be filtered out."""
        path = _write_csv(_regression_df())
        state = _base_state(
            path, task_type=TaskType.REGRESSION, exp_id="exp-lr-skip",
            models=["logistic_regression", "random_forest"],
        )
        result = _run_training(state)
        assert "logistic_regression" not in result["trained_model_paths"]
        assert "random_forest" in result["trained_model_paths"]

    def test_multiclass_classification_trains(self, model_dir):
        path = _write_csv(_multiclass_df())
        state = _base_state(path, exp_id="exp-multi")
        state["selected_features"] = ["f1", "f2"]
        result = _run_training(state)
        assert len(result["trained_model_paths"]) > 0

    def test_raises_if_feature_dataset_not_found(self, model_dir):
        state = _base_state("/nonexistent/features.csv")
        with pytest.raises(TrainingError, match="not found"):
            _run_training(state)

    def test_raises_if_target_column_missing(self, model_dir):
        path = _write_csv(_classification_df())
        state = _base_state(path)
        state["target_column"] = "nonexistent"
        with pytest.raises(TrainingError, match="not found in feature dataset"):
            _run_training(state)

    def test_raises_if_all_models_invalid_for_task(self, model_dir):
        path = _write_csv(_regression_df())
        state = _base_state(
            path, task_type=TaskType.REGRESSION, exp_id="exp-invalid",
            models=["logistic_regression"],  # classification-only
        )
        with pytest.raises(TrainingError, match="No valid models"):
            _run_training(state)

    def test_accuracy_in_valid_range(self, model_dir):
        path = _write_csv(_classification_df())
        result = _run_training(_base_state(path, exp_id="exp-acc"))
        for name, metrics in result["training_metrics"].items():
            acc = metrics["accuracy"]
            assert 0.0 <= acc <= 1.0, f"{name} accuracy out of range: {acc}"

    def test_r2_in_valid_range(self, model_dir):
        path = _write_csv(_regression_df())
        result = _run_training(
            _base_state(path, task_type=TaskType.REGRESSION, exp_id="exp-r2")
        )
        for name, metrics in result["training_metrics"].items():
            r2 = metrics["r2"]
            assert r2 <= 1.0, f"{name} r2 > 1.0: {r2}"


# ── 2. train_single_model helper ─────────────────────────────────────────────

class TestTrainSingleModel:

    def _make_data(self, rows=100, n_features=4, task=TaskType.CLASSIFICATION):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (rows, n_features))
        y = rng.integers(0, 2, rows) if task == TaskType.CLASSIFICATION \
            else rng.uniform(0, 100, rows)
        split = int(rows * 0.8)
        return (X[:split], y[:split], X[split:], y[split:],
                [f"f{i}" for i in range(n_features)])

    def test_random_forest_classification(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data()
        result = train_single_model(
            "random_forest", X_tr, y_tr, X_te, y_te,
            TaskType.CLASSIFICATION, cols, tmp_path
        )
        assert "accuracy" in result["metrics"]
        assert Path(result["model_path"]).exists()

    def test_logistic_regression_classification(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data()
        result = train_single_model(
            "logistic_regression", X_tr, y_tr, X_te, y_te,
            TaskType.CLASSIFICATION, cols, tmp_path
        )
        assert "f1_weighted" in result["metrics"]

    def test_xgboost_classification(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data()
        result = train_single_model(
            "xgboost", X_tr, y_tr, X_te, y_te,
            TaskType.CLASSIFICATION, cols, tmp_path
        )
        assert "accuracy" in result["metrics"]

    def test_random_forest_regression(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data(task=TaskType.REGRESSION)
        result = train_single_model(
            "random_forest", X_tr, y_tr, X_te, y_te,
            TaskType.REGRESSION, cols, tmp_path
        )
        assert "rmse" in result["metrics"]
        assert "r2" in result["metrics"]

    def test_linear_regression(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data(task=TaskType.REGRESSION)
        result = train_single_model(
            "linear_regression", X_tr, y_tr, X_te, y_te,
            TaskType.REGRESSION, cols, tmp_path
        )
        assert "mae" in result["metrics"]

    def test_xgboost_regression(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data(task=TaskType.REGRESSION)
        result = train_single_model(
            "xgboost", X_tr, y_tr, X_te, y_te,
            TaskType.REGRESSION, cols, tmp_path
        )
        assert "rmse" in result["metrics"]

    def test_raises_on_unknown_model(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data()
        with pytest.raises(TrainingError, match="Unknown model"):
            train_single_model(
                "unknown_model", X_tr, y_tr, X_te, y_te,
                TaskType.CLASSIFICATION, cols, tmp_path
            )

    def test_raises_on_wrong_task_type(self, tmp_path):
        X_tr, y_tr, X_te, y_te, cols = self._make_data()
        with pytest.raises(TrainingError, match="does not support"):
            train_single_model(
                "logistic_regression", X_tr, y_tr, X_te, y_te,
                TaskType.REGRESSION, cols, tmp_path   # LR is classification-only
            )


# ── 3. Node wrapper ───────────────────────────────────────────────────────────

class TestTrainingNode:

    def test_node_result_appended_on_success(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path))
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["node_name"] == "training"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-dur"))
        assert result["node_results"][0]["duration_seconds"] >= 0

    def test_output_summary_has_model_list(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-summary"))
        summary = result["node_results"][0]["output_summary"]
        assert "models_trained" in summary
        assert len(summary["models_trained"]) > 0

    def test_node_result_error_on_bad_path(self, model_dir):
        state = _base_state("/bad/path.csv")
        result = training_agent(state)
        nr = result["node_results"][0]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr

    def test_errors_list_populated_on_failure(self, model_dir):
        state = _base_state("/bad/path.csv")
        result = training_agent(state)
        assert len(result["errors"]) > 0

    def test_existing_node_results_preserved(self, model_dir):
        path = _write_csv(_classification_df())
        prior: Any = [{"node_name": "feature_engineering",
                       "status": WorkflowNodeStatus.DONE}]
        state = _base_state(path, exp_id="exp-preserve")
        state["node_results"] = prior
        result = training_agent(state)
        assert len(result["node_results"]) == 2
        assert result["node_results"][0]["node_name"] == "feature_engineering"
        assert result["node_results"][1]["node_name"] == "training"

    def test_trained_model_paths_written_to_state(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-paths"))
        assert "trained_model_paths" in result
        assert len(result["trained_model_paths"]) > 0

    def test_training_metrics_written_to_state(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-metrics"))
        assert "training_metrics" in result
        assert len(result["training_metrics"]) > 0


# ── 4. Model artifacts on disk ────────────────────────────────────────────────

class TestModelArtifacts:

    def test_model_pkl_files_exist(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-pkl"))
        for name, model_path in result["trained_model_paths"].items():
            assert Path(model_path).exists(), f"{name} pkl not found"

    def test_model_pkl_files_loadable(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-load"))
        for name, model_path in result["trained_model_paths"].items():
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
            assert "model" in artifact, f"{name} pkl missing 'model' key"
            assert "feature_cols" in artifact

    def test_model_can_predict(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-predict"))
        for name, model_path in result["trained_model_paths"].items():
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
            model = artifact["model"]
            X_dummy = np.random.default_rng(0).uniform(0, 1, (5, 4))
            preds = model.predict(X_dummy)
            assert len(preds) == 5, f"{name} predict returned wrong shape"

    def test_models_saved_in_experiment_subdir(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="my-exp-789"))
        for name, model_path in result["trained_model_paths"].items():
            assert "my-exp-789" in model_path, f"{name} not in exp subdir"

    def test_model_pkl_named_correctly(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(
            _base_state(path, exp_id="exp-name", models=["random_forest"])
        )
        model_path = result["trained_model_paths"]["random_forest"]
        assert Path(model_path).name == "random_forest.pkl"


# ── 5. Parallel execution ─────────────────────────────────────────────────────

class TestParallelExecution:

    def test_all_three_models_trained_concurrently(self, model_dir):
        """All 3 models must be present in results — parallel or not."""
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-parallel"))
        trained = result["trained_model_paths"]
        assert len(trained) == 3
        assert set(trained.keys()) == {
            "random_forest", "logistic_regression", "xgboost"
        }

    def test_each_model_has_independent_metrics(self, model_dir):
        """Each model's metrics dict must be independent (not shared reference)."""
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-indep"))
        metrics = result["training_metrics"]
        assert len(metrics) == 3
        # Verify they're not all the same object
        metric_ids = [id(m) for m in metrics.values()]
        assert len(set(metric_ids)) == 3

    def test_models_failed_list_empty_on_success(self, model_dir):
        path = _write_csv(_classification_df())
        result = training_agent(_base_state(path, exp_id="exp-no-fail"))
        assert result.get("models_failed", []) == []

    def test_regression_parallel_training(self, model_dir):
        path = _write_csv(_regression_df())
        state = _base_state(
            path, task_type=TaskType.REGRESSION, exp_id="exp-reg-par"
        )
        state["selected_features"] = ["f1", "f2", "f3"]
        result = training_agent(state)
        assert len(result["trained_model_paths"]) == 3
        assert set(result["trained_model_paths"].keys()) == {
            "random_forest", "linear_regression", "xgboost"
        }


# ── 6. Full graph integration ─────────────────────────────────────────────────

class TestGraphTraining:

    def test_training_node_runs_in_graph(self, model_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-train-001",
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
        assert "training" in node_names
        train_nr = next(
            nr for nr in final["node_results"] if nr["node_name"] == "training"
        )
        assert train_nr["status"] == WorkflowNodeStatus.DONE

    def test_trained_model_paths_in_final_state(self, model_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-train-002",
            "dataset_id": "ds-002",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "trained_model_paths" in final
        assert len(final["trained_model_paths"]) > 0
        for model_path in final["trained_model_paths"].values():
            assert Path(model_path).exists()

    def test_training_metrics_in_final_state(self, model_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-train-003",
            "dataset_id": "ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "training_metrics" in final
        assert len(final["training_metrics"]) > 0

    def test_pipeline_order_feature_then_training(self, model_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-train-004",
            "dataset_id": "ds-004",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert node_names.index("feature_engineering") < node_names.index("training")
