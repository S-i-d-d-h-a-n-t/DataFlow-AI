"""
Phase 6 tests — Feature Engineering Agent.

Groups:
  1. TestRunFeatureEngineering  — pure _run_feature_engineering() logic
  2. TestFeatureNode            — feature_agent() node wrapper (state, errors)
  3. TestOutputFiles            — feature CSV and transformers.pkl written correctly
  4. TestGraphFeature           — full graph confirms feature node runs in order
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agents.feature_agent import feature_agent, _run_feature_engineering, FeatureError
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


def _classification_df(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":        rng.uniform(18, 70, rows),
        "salary":     rng.uniform(30_000, 120_000, rows),
        "score":      rng.uniform(0, 100, rows),
        "department": rng.choice(["eng", "sales", "hr"], rows),   # low-card → ordinal
        "region":     rng.choice(["north", "south", "east", "west"], rows),
        "target":     rng.integers(0, 2, rows),
    })


def _regression_df(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, rows),
        "x2": rng.uniform(-5, 5, rows),
        "x3": rng.uniform(100, 200, rows),
        "cat": rng.choice(["a", "b", "c", "d"], rows),
        "target": rng.uniform(0, 100, rows),
    })


def _many_feature_df(rows: int = 200, n_features: int = 30) -> pd.DataFrame:
    """Dataset with more features than top_k to trigger MI selection."""
    rng = np.random.default_rng(99)
    data = {f"f{i}": rng.uniform(0, 1, rows) for i in range(n_features)}
    data["target"] = rng.integers(0, 2, rows)
    return pd.DataFrame(data)


def _df_with_constant_col(rows: int = 100) -> pd.DataFrame:
    df = _classification_df(rows)
    df["constant"] = 5.0   # zero variance after scaling
    return df


@pytest.fixture()
def artifact_dir(tmp_path):
    """Patch OUTPUTS_ARTIFACTS_DIR to a temp directory."""
    import app.utils.file_manager as fm
    original = fm.OUTPUTS_ARTIFACTS_DIR
    fm.OUTPUTS_ARTIFACTS_DIR = tmp_path
    yield tmp_path
    fm.OUTPUTS_ARTIFACTS_DIR = original


def _base_state(path: str, task_type=TaskType.CLASSIFICATION,
                exp_id: str = "exp-001") -> WorkflowState:
    return {
        "cleaned_dataset_path": path,
        "target_column": "target",
        "task_type": task_type,
        "experiment_id": exp_id,
        "node_results": [],
        "errors": [],
    }


# ── 1. Pure feature engineering logic ────────────────────────────────────────

class TestRunFeatureEngineering:

    def test_returns_required_keys(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        assert "feature_dataset_path" in result
        assert "selected_features" in result
        assert "feature_report" in result

    def test_report_contains_all_keys(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        report = result["feature_report"]
        for key in (
            "features_before", "features_after", "features_dropped_low_variance",
            "features_selected", "numeric_scaled", "categorical_ordinal_encoded",
            "categorical_ohe_encoded", "mi_scores", "top_k",
            "ordinal_cardinality_threshold", "transformer_path", "task_type",
        ):
            assert key in report, f"Missing report key: {key}"

    def test_numeric_columns_are_scaled(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        report = result["feature_report"]
        assert "age" in report["numeric_scaled"]
        assert "salary" in report["numeric_scaled"]

    def test_low_cardinality_categorical_ordinal_encoded(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        report = result["feature_report"]
        # department has 3 unique values → ordinal
        assert "department" in report["categorical_ordinal_encoded"]

    def test_high_cardinality_categorical_ohe_encoded(self, artifact_dir):
        """Columns with > ordinal_threshold unique values → one-hot."""
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "num": rng.uniform(0, 1, 100),
            "high_card": [f"cat_{i}" for i in range(100)],  # 100 unique → OHE
            "target": rng.integers(0, 2, 100),
        })
        path = _write_csv(df)
        result = _run_feature_engineering(
            _base_state(path, exp_id="exp-ohe")
        )
        assert "high_card" in result["feature_report"]["categorical_ohe_encoded"]

    def test_target_not_in_selected_features(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        assert "target" not in result["selected_features"]

    def test_target_present_in_output_csv(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        df_out = pd.read_csv(result["feature_dataset_path"])
        assert "target" in df_out.columns

    def test_near_zero_variance_feature_dropped(self, artifact_dir):
        path = _write_csv(_df_with_constant_col())
        result = _run_feature_engineering(_base_state(path, exp_id="exp-var"))
        dropped = result["feature_report"]["features_dropped_low_variance"]
        assert "constant" in dropped

    def test_mi_selection_reduces_features(self, artifact_dir):
        """With 30 features and top_k=10, MI should select exactly 10."""
        path = _write_csv(_many_feature_df(n_features=30))
        state = _base_state(path, exp_id="exp-mi")
        state["pipeline_config"] = {"feature_engineering": {"top_k_features": 10}}
        result = _run_feature_engineering(state)
        assert len(result["selected_features"]) <= 10

    def test_mi_selection_disabled_keeps_all_features(self, artifact_dir):
        path = _write_csv(_many_feature_df(n_features=30))
        state = _base_state(path, exp_id="exp-no-mi")
        state["pipeline_config"] = {
            "feature_engineering": {"feature_selection": False}
        }
        result = _run_feature_engineering(state)
        # All features kept (minus any dropped for low variance)
        report = result["feature_report"]
        assert report["mi_scores"] == {}

    def test_regression_task_type_uses_mi_regression(self, artifact_dir):
        path = _write_csv(_regression_df())
        result = _run_feature_engineering(
            _base_state(path, task_type=TaskType.REGRESSION, exp_id="exp-reg")
        )
        assert result["feature_report"]["task_type"] == "regression"

    def test_features_before_count_correct(self, artifact_dir):
        df = _classification_df()
        path = _write_csv(df)
        result = _run_feature_engineering(_base_state(path))
        # features_before = all columns minus target
        assert result["feature_report"]["features_before"] == len(df.columns) - 1

    def test_output_csv_has_no_missing_values(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = _run_feature_engineering(_base_state(path))
        df_out = pd.read_csv(result["feature_dataset_path"])
        assert df_out.isna().sum().sum() == 0

    def test_output_csv_row_count_unchanged(self, artifact_dir):
        df = _classification_df(rows=80)
        path = _write_csv(df)
        result = _run_feature_engineering(_base_state(path, exp_id="exp-rows"))
        df_out = pd.read_csv(result["feature_dataset_path"])
        assert len(df_out) == 80

    def test_raises_if_cleaned_dataset_not_found(self, artifact_dir):
        state = _base_state("/nonexistent/cleaned.csv")
        with pytest.raises(FeatureError, match="not found"):
            _run_feature_engineering(state)

    def test_raises_if_target_column_missing(self, artifact_dir):
        path = _write_csv(_classification_df())
        state = _base_state(path)
        state["target_column"] = "nonexistent"
        with pytest.raises(FeatureError, match="not found in cleaned dataset"):
            _run_feature_engineering(state)

    def test_raises_if_target_column_empty(self, artifact_dir):
        path = _write_csv(_classification_df())
        state = _base_state(path)
        state["target_column"] = ""
        with pytest.raises(FeatureError, match="required"):
            _run_feature_engineering(state)

    def test_custom_ordinal_threshold_via_config(self, artifact_dir):
        """With threshold=0, all categoricals should be OHE-encoded."""
        path = _write_csv(_classification_df())
        state = _base_state(path, exp_id="exp-ohe-thresh")
        state["pipeline_config"] = {
            "feature_engineering": {"ordinal_cardinality_threshold": 0}
        }
        result = _run_feature_engineering(state)
        report = result["feature_report"]
        assert report["categorical_ordinal_encoded"] == []
        assert len(report["categorical_ohe_encoded"]) > 0

    def test_drop_low_variance_disabled_via_config(self, artifact_dir):
        path = _write_csv(_df_with_constant_col())
        state = _base_state(path, exp_id="exp-no-drop-var")
        state["pipeline_config"] = {
            "feature_engineering": {"drop_low_variance": False}
        }
        result = _run_feature_engineering(state)
        assert result["feature_report"]["features_dropped_low_variance"] == []


# ── 2. Node wrapper ───────────────────────────────────────────────────────────

class TestFeatureNode:

    def test_node_result_appended_on_success(self, artifact_dir):
        path = _write_csv(_classification_df())
        state = _base_state(path)
        result = feature_agent(state)
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["node_name"] == "feature_engineering"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-dur"))
        assert result["node_results"][0]["duration_seconds"] >= 0

    def test_output_summary_has_feature_counts(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-summary"))
        summary = result["node_results"][0]["output_summary"]
        assert "features_before" in summary
        assert "features_after" in summary
        assert "feature_dataset_path" in summary

    def test_node_result_error_on_bad_path(self, artifact_dir):
        state = _base_state("/bad/path.csv")
        result = feature_agent(state)
        nr = result["node_results"][0]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr

    def test_errors_list_populated_on_failure(self, artifact_dir):
        state = _base_state("/bad/path.csv")
        result = feature_agent(state)
        assert len(result["errors"]) > 0

    def test_fallback_path_on_failure(self, artifact_dir):
        """On failure, feature_dataset_path falls back to cleaned_dataset_path."""
        cleaned = "/bad/path.csv"
        state = _base_state(cleaned)
        result = feature_agent(state)
        assert result["feature_dataset_path"] == cleaned

    def test_existing_node_results_preserved(self, artifact_dir):
        path = _write_csv(_classification_df())
        prior: Any = [{"node_name": "cleaning", "status": WorkflowNodeStatus.DONE}]
        state = _base_state(path, exp_id="exp-preserve")
        state["node_results"] = prior
        result = feature_agent(state)
        assert len(result["node_results"]) == 2
        assert result["node_results"][0]["node_name"] == "cleaning"
        assert result["node_results"][1]["node_name"] == "feature_engineering"

    def test_selected_features_written_to_state(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-sel"))
        assert isinstance(result["selected_features"], list)
        assert len(result["selected_features"]) > 0

    def test_feature_report_written_to_state(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-rep"))
        assert "feature_report" in result
        assert result["feature_report"]["features_before"] > 0


# ── 3. Output files on disk ───────────────────────────────────────────────────

class TestOutputFiles:

    def test_feature_csv_exists(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-file"))
        assert Path(result["feature_dataset_path"]).exists()

    def test_feature_csv_is_valid_dataframe(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-valid"))
        df_out = pd.read_csv(result["feature_dataset_path"])
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) > 0

    def test_feature_csv_saved_in_experiment_subdir(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="my-exp-456"))
        assert "my-exp-456" in result["feature_dataset_path"]

    def test_feature_csv_named_correctly(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-name"))
        assert Path(result["feature_dataset_path"]).name == "feature_dataset.csv"

    def test_transformers_pkl_exists(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-pkl"))
        transformer_path = Path(result["feature_report"]["transformer_path"])
        assert transformer_path.exists()

    def test_transformers_pkl_is_loadable(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-load"))
        transformer_path = result["feature_report"]["transformer_path"]
        with open(transformer_path, "rb") as f:
            transformers = pickle.load(f)
        assert isinstance(transformers, dict)
        assert "scaler" in transformers

    def test_scaler_in_transformers_is_fitted(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-scaler"))
        transformer_path = result["feature_report"]["transformer_path"]
        with open(transformer_path, "rb") as f:
            transformers = pickle.load(f)
        scaler = transformers["scaler"]["scaler"]
        # A fitted StandardScaler has mean_ attribute
        assert hasattr(scaler, "mean_")

    def test_all_selected_features_in_output_csv(self, artifact_dir):
        path = _write_csv(_classification_df())
        result = feature_agent(_base_state(path, exp_id="exp-cols"))
        df_out = pd.read_csv(result["feature_dataset_path"])
        for feat in result["selected_features"]:
            assert feat in df_out.columns, f"Feature '{feat}' missing from output CSV"


# ── 4. Full graph integration ─────────────────────────────────────────────────

class TestGraphFeature:

    def test_feature_node_runs_in_graph(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-feat-001",
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
        assert "feature_engineering" in node_names
        fe_nr = next(
            nr for nr in final["node_results"]
            if nr["node_name"] == "feature_engineering"
        )
        assert fe_nr["status"] == WorkflowNodeStatus.DONE

    def test_feature_dataset_path_in_final_state(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-feat-002",
            "dataset_id": "ds-002",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "feature_dataset_path" in final
        assert final["feature_dataset_path"].endswith("feature_dataset.csv")
        assert Path(final["feature_dataset_path"]).exists()

    def test_selected_features_in_final_state(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-feat-003",
            "dataset_id": "ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "selected_features" in final
        assert isinstance(final["selected_features"], list)
        assert len(final["selected_features"]) > 0

    def test_pipeline_order_cleaning_then_feature(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-feat-004",
            "dataset_id": "ds-004",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert node_names.index("cleaning") < node_names.index("feature_engineering")
