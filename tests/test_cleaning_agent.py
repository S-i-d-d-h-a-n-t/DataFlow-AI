"""
Phase 5 tests — Cleaning Agent.

Groups:
  1. TestRunCleaning      — pure _run_cleaning() logic
  2. TestCleaningNode     — cleaning_agent() node wrapper (state, errors)
  3. TestCleanedFile      — output CSV is written correctly to disk
  4. TestGraphCleaning    — full graph confirms cleaning node runs in order
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agents.cleaning_agent import cleaning_agent, _run_cleaning, CleaningError
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


def _clean_df(rows: int = 100) -> pd.DataFrame:
    """Fully clean dataset — no missing, no duplicates."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age":        rng.integers(18, 70, rows).astype(float),
        "salary":     rng.uniform(30_000, 120_000, rows),
        "department": rng.choice(["eng", "sales", "hr"], rows),
        "target":     rng.integers(0, 2, rows),
    })


def _df_with_missing(rows: int = 100) -> pd.DataFrame:
    df = _clean_df(rows)
    rng = np.random.default_rng(1)
    df.loc[rng.choice(rows, 20, replace=False), "salary"] = np.nan
    df.loc[rng.choice(rows, 10, replace=False), "department"] = np.nan
    return df


def _df_with_high_missing(rows: int = 100) -> pd.DataFrame:
    """Column 'junk' has 70% missing — should be dropped."""
    df = _clean_df(rows)
    df["junk"] = np.nan
    df.loc[:29, "junk"] = 1.0   # only 30% filled → 70% missing
    return df


def _df_with_duplicates(rows: int = 80) -> pd.DataFrame:
    df = _clean_df(rows // 2)
    return pd.concat([df, df], ignore_index=True)   # 100% duplicates


def _df_with_outliers(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    df = _clean_df(rows)
    # Inject extreme outliers into salary
    df.loc[0, "salary"] = 1_000_000_000.0
    df.loc[1, "salary"] = -999_999.0
    return df


def _df_missing_target(rows: int = 80) -> pd.DataFrame:
    df = _clean_df(rows)
    # Cast target to float so NaN can be stored (int columns coerce NaN to 0)
    df["target"] = df["target"].astype(float)
    df.loc[:9, "target"] = np.nan
    return df


@pytest.fixture()
def artifact_dir(tmp_path):
    """Patch OUTPUTS_ARTIFACTS_DIR to a temp directory."""
    import app.utils.file_manager as fm
    original = fm.OUTPUTS_ARTIFACTS_DIR
    fm.OUTPUTS_ARTIFACTS_DIR = tmp_path
    yield tmp_path
    fm.OUTPUTS_ARTIFACTS_DIR = original


# ── 1. Pure cleaning logic ────────────────────────────────────────────────────

class TestRunCleaning:

    def test_returns_required_keys(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-keys",
        }
        result = _run_cleaning(state)
        assert "cleaned_dataset_path" in result
        assert "cleaning_report" in result

    def test_clean_data_passes_through_unchanged_rows(self, artifact_dir):
        df = _clean_df(rows=50)
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-passthrough",
        }
        result = _run_cleaning(state)
        report = result["cleaning_report"]
        assert report["rows_before"] == 50
        assert report["rows_after"] == 50
        assert report["rows_dropped"] == 0

    def test_numeric_missing_imputed_with_median(self, artifact_dir):
        df = _df_with_missing()
        original_median = df["salary"].median()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-impute-num",
        }
        result = _run_cleaning(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert cleaned["salary"].isna().sum() == 0
        assert "salary" in result["cleaning_report"]["columns_imputed_numeric"]

    def test_categorical_missing_imputed_with_mode(self, artifact_dir):
        df = _df_with_missing()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-impute-cat",
        }
        result = _run_cleaning(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert cleaned["department"].isna().sum() == 0
        assert "department" in result["cleaning_report"]["columns_imputed_categorical"]

    def test_high_missing_column_dropped(self, artifact_dir):
        df = _df_with_high_missing()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-drop-col",
        }
        result = _run_cleaning(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert "junk" not in cleaned.columns
        assert "junk" in result["cleaning_report"]["columns_dropped"]

    def test_target_column_never_dropped(self, artifact_dir):
        """Even if target has high missing, it must not be dropped."""
        df = _clean_df(100)
        df["target"] = np.nan          # 100% missing target
        df.loc[:9, "target"] = 1.0     # only 10% filled
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-keep-target",
        }
        result = _run_cleaning(state)
        assert "target" not in result["cleaning_report"]["columns_dropped"]

    def test_duplicate_rows_removed(self, artifact_dir):
        df = _df_with_duplicates()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-dedup",
        }
        result = _run_cleaning(state)
        report = result["cleaning_report"]
        assert report["duplicates_removed"] > 0
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert len(cleaned) < len(df)

    def test_outliers_clipped(self, artifact_dir):
        df = _df_with_outliers()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-clip",
        }
        result = _run_cleaning(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert cleaned["salary"].max() < 1_000_000_000.0
        assert cleaned["salary"].min() > -999_999.0
        assert "salary" in result["cleaning_report"]["columns_clipped"]

    def test_target_column_not_clipped(self, artifact_dir):
        """Target column must never be clipped."""
        df = _df_with_outliers()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-clip-target",
        }
        result = _run_cleaning(state)
        assert "target" not in result["cleaning_report"]["columns_clipped"]

    def test_rows_with_missing_target_dropped(self, artifact_dir):
        df = _df_missing_target()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-drop-target-rows",
        }
        result = _run_cleaning(state)
        report = result["cleaning_report"]
        assert report["target_missing_dropped"] == 10
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert cleaned["target"].isna().sum() == 0

    def test_clip_outliers_disabled_via_config(self, artifact_dir):
        df = _df_with_outliers()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-clip",
            "pipeline_config": {"cleaning": {"clip_outliers": False}},
        }
        result = _run_cleaning(state)
        assert result["cleaning_report"]["columns_clipped"] == []

    def test_imputation_disabled_via_config(self, artifact_dir):
        df = _df_with_missing()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-impute",
            "pipeline_config": {"cleaning": {"impute_missing": False}},
        }
        result = _run_cleaning(state)
        report = result["cleaning_report"]
        assert report["columns_imputed_numeric"] == []
        assert report["columns_imputed_categorical"] == []

    def test_custom_drop_threshold_via_config(self, artifact_dir):
        """With threshold=0.0, any column with any missing should be dropped."""
        df = _df_with_missing()
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-low-threshold",
            "pipeline_config": {
                "cleaning": {
                    "drop_missing_threshold": 0.0,
                    "impute_missing": False,
                }
            },
        }
        result = _run_cleaning(state)
        dropped = result["cleaning_report"]["columns_dropped"]
        assert "salary" in dropped
        assert "department" in dropped

    def test_report_contains_all_required_keys(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-report-keys",
        }
        result = _run_cleaning(state)
        report = result["cleaning_report"]
        for key in (
            "rows_before", "rows_after", "rows_dropped",
            "duplicates_removed", "target_missing_dropped",
            "columns_before", "columns_after",
            "columns_dropped", "columns_imputed_numeric",
            "columns_imputed_categorical", "columns_clipped",
            "clip_bounds", "drop_missing_threshold",
            "iqr_multiplier", "imputation_strategy",
        ):
            assert key in report, f"Missing report key: {key}"

    def test_raises_if_dataset_not_found(self, artifact_dir):
        state: WorkflowState = {
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "experiment_id": "exp-notfound",
        }
        with pytest.raises(CleaningError, match="not found"):
            _run_cleaning(state)

    def test_no_data_loss_on_clean_dataset(self, artifact_dir):
        """A perfectly clean dataset should come out identical (modulo index)."""
        df = _clean_df(rows=60)
        path = _write_csv(df)
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-loss",
            "pipeline_config": {"cleaning": {"clip_outliers": False}},
        }
        result = _run_cleaning(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert len(cleaned) == 60
        assert set(cleaned.columns) == set(df.columns)


# ── 2. Node wrapper ───────────────────────────────────────────────────────────

class TestCleaningNode:

    def test_node_result_appended_on_success(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-node-ok",
            "node_results": [],
            "errors": [],
        }
        result = cleaning_agent(state)
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["node_name"] == "cleaning"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-dur",
            "node_results": [],
        }
        result = cleaning_agent(state)
        assert result["node_results"][0]["duration_seconds"] >= 0

    def test_output_summary_has_row_counts(self, artifact_dir):
        path = _write_csv(_clean_df(50))
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-summary",
            "node_results": [],
        }
        result = cleaning_agent(state)
        summary = result["node_results"][0]["output_summary"]
        assert summary["rows_before"] == 50
        assert "rows_after" in summary
        assert "cleaned_path" in summary

    def test_node_result_error_on_bad_path(self, artifact_dir):
        state: WorkflowState = {
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
            "experiment_id": "exp-err",
            "node_results": [],
            "errors": [],
        }
        result = cleaning_agent(state)
        nr = result["node_results"][0]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr

    def test_errors_list_populated_on_failure(self, artifact_dir):
        state: WorkflowState = {
            "dataset_path": "/bad/path.csv",
            "target_column": "target",
            "experiment_id": "exp-errlist",
            "node_results": [],
            "errors": [],
        }
        result = cleaning_agent(state)
        assert len(result["errors"]) > 0

    def test_fallback_path_on_failure(self, artifact_dir):
        """On failure, cleaned_dataset_path should fall back to the raw path."""
        raw_path = "/bad/path.csv"
        state: WorkflowState = {
            "dataset_path": raw_path,
            "target_column": "target",
            "experiment_id": "exp-fallback",
            "node_results": [],
            "errors": [],
        }
        result = cleaning_agent(state)
        assert result["cleaned_dataset_path"] == raw_path

    def test_existing_node_results_preserved(self, artifact_dir):
        path = _write_csv(_clean_df())
        prior: Any = [{"node_name": "eda", "status": WorkflowNodeStatus.DONE}]
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-preserve",
            "node_results": prior,
        }
        result = cleaning_agent(state)
        assert len(result["node_results"]) == 2
        assert result["node_results"][0]["node_name"] == "eda"
        assert result["node_results"][1]["node_name"] == "cleaning"

    def test_cleaned_path_written_to_state(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-path",
            "node_results": [],
        }
        result = cleaning_agent(state)
        assert "cleaned_dataset_path" in result
        assert result["cleaned_dataset_path"].endswith("cleaned_dataset.csv")

    def test_cleaning_report_written_to_state(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-report",
            "node_results": [],
        }
        result = cleaning_agent(state)
        assert "cleaning_report" in result
        assert result["cleaning_report"]["rows_before"] == 100


# ── 3. Output CSV on disk ─────────────────────────────────────────────────────

class TestCleanedFile:

    def test_cleaned_csv_exists_on_disk(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-file",
            "node_results": [],
        }
        result = cleaning_agent(state)
        assert Path(result["cleaned_dataset_path"]).exists()

    def test_cleaned_csv_is_valid_dataframe(self, artifact_dir):
        path = _write_csv(_df_with_missing())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-valid",
            "node_results": [],
        }
        result = cleaning_agent(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0

    def test_cleaned_csv_has_no_missing_values(self, artifact_dir):
        path = _write_csv(_df_with_missing())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-missing",
            "node_results": [],
        }
        result = cleaning_agent(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert cleaned.isna().sum().sum() == 0

    def test_cleaned_csv_has_no_duplicates(self, artifact_dir):
        path = _write_csv(_df_with_duplicates())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-dups",
            "node_results": [],
        }
        result = cleaning_agent(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert cleaned.duplicated().sum() == 0

    def test_cleaned_csv_saved_in_experiment_subdir(self, artifact_dir):
        path = _write_csv(_clean_df())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "my-exp-123",
            "node_results": [],
        }
        result = cleaning_agent(state)
        cleaned_path = Path(result["cleaned_dataset_path"])
        assert "my-exp-123" in str(cleaned_path)

    def test_high_missing_column_absent_in_cleaned_csv(self, artifact_dir):
        path = _write_csv(_df_with_high_missing())
        state: WorkflowState = {
            "dataset_path": path,
            "target_column": "target",
            "experiment_id": "exp-no-junk",
            "node_results": [],
        }
        result = cleaning_agent(state)
        cleaned = pd.read_csv(result["cleaned_dataset_path"])
        assert "junk" not in cleaned.columns


# ── 4. Full graph integration ─────────────────────────────────────────────────

class TestGraphCleaning:

    def test_cleaning_node_runs_in_graph(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_df_with_missing())
        initial: WorkflowState = {
            "experiment_id": "graph-clean-001",
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
        assert "cleaning" in node_names
        cleaning_nr = next(
            nr for nr in final["node_results"] if nr["node_name"] == "cleaning"
        )
        assert cleaning_nr["status"] == WorkflowNodeStatus.DONE

    def test_cleaned_path_in_final_state(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_df_with_missing())
        initial: WorkflowState = {
            "experiment_id": "graph-clean-002",
            "dataset_id": "ds-002",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "cleaned_dataset_path" in final
        assert final["cleaned_dataset_path"].endswith("cleaned_dataset.csv")
        assert Path(final["cleaned_dataset_path"]).exists()

    def test_pipeline_order_eda_then_cleaning(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_clean_df())
        initial: WorkflowState = {
            "experiment_id": "graph-clean-003",
            "dataset_id": "ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert node_names.index("eda") < node_names.index("cleaning")

    def test_cleaning_report_in_final_state(self, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_df_with_missing())
        initial: WorkflowState = {
            "experiment_id": "graph-clean-004",
            "dataset_id": "ds-004",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "cleaning_report" in final
        assert "rows_before" in final["cleaning_report"]
