"""
Phase 8 tests — Evaluation Agent.

Groups:
  1. TestRunEvaluation     — pure _run_evaluation() logic
  2. TestEvaluationNode    — evaluation_agent() node wrapper (state, errors)
  3. TestRankingLogic      — ranking correctness for classification & regression
  4. TestGraphEvaluation   — full graph confirms evaluation node runs in order
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agents.evaluation_agent import (
    evaluation_agent,
    _run_evaluation,
    EvaluationError,
)
from app.enums.task_type import TaskType
from app.enums.workflow_status import WorkflowNodeStatus
from app.state.workflow_state import WorkflowState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cls_metrics(accuracy: float, f1: float) -> dict[str, Any]:
    """Build a realistic classification metrics dict."""
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "precision_weighted": round(f1 * 0.98, 4),
        "recall_weighted": round(f1 * 0.97, 4),
        "roc_auc": round(f1 * 1.02, 4),
        "train_duration_seconds": 0.5,
        "n_train": 120,
        "n_test": 30,
    }


def _reg_metrics(rmse: float, r2: float) -> dict[str, Any]:
    """Build a realistic regression metrics dict."""
    return {
        "mae": round(rmse * 0.7, 4),
        "mse": round(rmse ** 2, 4),
        "rmse": rmse,
        "r2": r2,
        "train_duration_seconds": 0.3,
        "n_train": 120,
        "n_test": 30,
    }


def _cls_state(
    rf_f1: float = 0.82,
    lr_f1: float = 0.78,
    xgb_f1: float = 0.85,
    model_paths: dict | None = None,
    exp_id: str = "exp-001",
) -> WorkflowState:
    paths = model_paths or {
        "random_forest": "/tmp/rf.pkl",
        "logistic_regression": "/tmp/lr.pkl",
        "xgboost": "/tmp/xgb.pkl",
    }
    return {
        "training_metrics": {
            "random_forest": _cls_metrics(0.80, rf_f1),
            "logistic_regression": _cls_metrics(0.76, lr_f1),
            "xgboost": _cls_metrics(0.83, xgb_f1),
        },
        "trained_model_paths": paths,
        "task_type": TaskType.CLASSIFICATION,
        "experiment_id": exp_id,
        "node_results": [],
        "errors": [],
    }


def _reg_state(
    rf_rmse: float = 12.5,
    lr_rmse: float = 15.0,
    xgb_rmse: float = 10.2,
    exp_id: str = "exp-reg",
) -> WorkflowState:
    return {
        "training_metrics": {
            "random_forest": _reg_metrics(rf_rmse, 0.88),
            "linear_regression": _reg_metrics(lr_rmse, 0.82),
            "xgboost": _reg_metrics(xgb_rmse, 0.92),
        },
        "trained_model_paths": {
            "random_forest": "/tmp/rf.pkl",
            "linear_regression": "/tmp/lr.pkl",
            "xgboost": "/tmp/xgb.pkl",
        },
        "task_type": TaskType.REGRESSION,
        "experiment_id": exp_id,
        "node_results": [],
        "errors": [],
    }


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


@pytest.fixture()
def model_dir(tmp_path):
    import app.utils.file_manager as fm
    original = fm.OUTPUTS_MODELS_DIR
    fm.OUTPUTS_MODELS_DIR = tmp_path
    yield tmp_path
    fm.OUTPUTS_MODELS_DIR = original


@pytest.fixture()
def artifact_dir(tmp_path):
    import app.utils.file_manager as fm
    original = fm.OUTPUTS_ARTIFACTS_DIR
    fm.OUTPUTS_ARTIFACTS_DIR = tmp_path
    yield tmp_path
    fm.OUTPUTS_ARTIFACTS_DIR = original


# ── 1. Pure evaluation logic ──────────────────────────────────────────────────

class TestRunEvaluation:

    def test_returns_required_keys(self):
        result = _run_evaluation(_cls_state())
        assert "evaluation_results" in result
        assert "best_model_name" in result
        assert "best_model_path" in result

    def test_evaluation_results_has_all_keys(self):
        result = _run_evaluation(_cls_state())
        ev = result["evaluation_results"]
        for key in (
            "models_evaluated", "task_type", "primary_metric",
            "higher_is_better", "best_model", "best_score",
            "ranked_models", "comparison_table",
        ):
            assert key in ev, f"Missing key: {key}"

    def test_classification_primary_metric_is_f1(self):
        result = _run_evaluation(_cls_state())
        assert result["evaluation_results"]["primary_metric"] == "f1_weighted"

    def test_regression_primary_metric_is_rmse(self):
        result = _run_evaluation(_reg_state())
        assert result["evaluation_results"]["primary_metric"] == "rmse"

    def test_classification_higher_is_better_true(self):
        result = _run_evaluation(_cls_state())
        assert result["evaluation_results"]["higher_is_better"] is True

    def test_regression_higher_is_better_false(self):
        result = _run_evaluation(_reg_state())
        assert result["evaluation_results"]["higher_is_better"] is False

    def test_best_classification_model_has_highest_f1(self):
        # xgboost has f1=0.85 — should be best
        result = _run_evaluation(_cls_state(rf_f1=0.82, lr_f1=0.78, xgb_f1=0.85))
        assert result["best_model_name"] == "xgboost"

    def test_best_regression_model_has_lowest_rmse(self):
        # xgboost has rmse=10.2 — should be best (lowest)
        result = _run_evaluation(_reg_state(rf_rmse=12.5, lr_rmse=15.0, xgb_rmse=10.2))
        assert result["best_model_name"] == "xgboost"

    def test_ranked_models_sorted_best_first_classification(self):
        result = _run_evaluation(_cls_state(rf_f1=0.82, lr_f1=0.78, xgb_f1=0.85))
        ranked = result["evaluation_results"]["ranked_models"]
        scores = [r["primary_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranked_models_sorted_best_first_regression(self):
        result = _run_evaluation(_reg_state(rf_rmse=12.5, lr_rmse=15.0, xgb_rmse=10.2))
        ranked = result["evaluation_results"]["ranked_models"]
        scores = [r["primary_score"] for r in ranked]
        assert scores == sorted(scores)   # ascending for regression (lower=better)

    def test_ranked_models_have_rank_field(self):
        result = _run_evaluation(_cls_state())
        for i, entry in enumerate(result["evaluation_results"]["ranked_models"]):
            assert entry["rank"] == i + 1

    def test_all_models_appear_in_ranked_list(self):
        result = _run_evaluation(_cls_state())
        model_names = {r["model"] for r in result["evaluation_results"]["ranked_models"]}
        assert model_names == {"random_forest", "logistic_regression", "xgboost"}

    def test_comparison_table_has_all_models(self):
        result = _run_evaluation(_cls_state())
        table_models = {row["model"] for row in result["evaluation_results"]["comparison_table"]}
        assert table_models == {"random_forest", "logistic_regression", "xgboost"}

    def test_comparison_table_includes_bookkeeping(self):
        result = _run_evaluation(_cls_state())
        for row in result["evaluation_results"]["comparison_table"]:
            assert "train_duration_seconds" in row
            assert "n_train" in row
            assert "n_test" in row

    def test_best_score_matches_best_model_metric(self):
        result = _run_evaluation(_cls_state(rf_f1=0.82, lr_f1=0.78, xgb_f1=0.85))
        ev = result["evaluation_results"]
        assert ev["best_score"] == 0.85

    def test_models_evaluated_list_correct(self):
        result = _run_evaluation(_cls_state())
        ev = result["evaluation_results"]
        assert set(ev["models_evaluated"]) == {
            "random_forest", "logistic_regression", "xgboost"
        }

    def test_raises_if_no_training_metrics(self):
        state: WorkflowState = {
            "training_metrics": {},
            "trained_model_paths": {"rf": "/tmp/rf.pkl"},
            "task_type": TaskType.CLASSIFICATION,
        }
        with pytest.raises(EvaluationError, match="No training_metrics"):
            _run_evaluation(state)

    def test_raises_if_no_trained_model_paths(self):
        state: WorkflowState = {
            "training_metrics": {"rf": _cls_metrics(0.8, 0.8)},
            "trained_model_paths": {},
            "task_type": TaskType.CLASSIFICATION,
        }
        with pytest.raises(EvaluationError, match="No trained_model_paths"):
            _run_evaluation(state)

    def test_single_model_still_evaluates(self):
        state: WorkflowState = {
            "training_metrics": {"random_forest": _cls_metrics(0.80, 0.82)},
            "trained_model_paths": {"random_forest": "/tmp/rf.pkl"},
            "task_type": TaskType.CLASSIFICATION,
        }
        result = _run_evaluation(state)
        assert result["best_model_name"] == "random_forest"
        assert len(result["evaluation_results"]["ranked_models"]) == 1

    def test_best_model_path_matches_trained_paths(self):
        paths = {
            "random_forest": "/tmp/rf.pkl",
            "logistic_regression": "/tmp/lr.pkl",
            "xgboost": "/tmp/xgb.pkl",
        }
        result = _run_evaluation(_cls_state(xgb_f1=0.90, model_paths=paths))
        assert result["best_model_path"] == "/tmp/xgb.pkl"

    def test_task_type_recorded_in_results(self):
        result = _run_evaluation(_cls_state())
        assert result["evaluation_results"]["task_type"] == "classification"

        result_reg = _run_evaluation(_reg_state())
        assert result_reg["evaluation_results"]["task_type"] == "regression"


# ── 2. Node wrapper ───────────────────────────────────────────────────────────

class TestEvaluationNode:

    def test_node_result_appended_on_success(self):
        result = evaluation_agent(_cls_state())
        assert len(result["node_results"]) == 1
        nr = result["node_results"][0]
        assert nr["node_name"] == "evaluation"
        assert nr["status"] == WorkflowNodeStatus.DONE

    def test_node_result_has_duration(self):
        result = evaluation_agent(_cls_state(exp_id="exp-dur"))
        assert result["node_results"][0]["duration_seconds"] >= 0

    def test_output_summary_has_best_model(self):
        result = evaluation_agent(_cls_state(exp_id="exp-summary"))
        summary = result["node_results"][0]["output_summary"]
        assert "best_model" in summary
        assert "primary_metric" in summary
        assert "best_score" in summary
        assert "ranking" in summary

    def test_output_summary_ranking_is_list(self):
        result = evaluation_agent(_cls_state(exp_id="exp-rank"))
        ranking = result["node_results"][0]["output_summary"]["ranking"]
        assert isinstance(ranking, list)
        assert len(ranking) == 3

    def test_node_result_error_on_empty_metrics(self):
        state: WorkflowState = {
            "training_metrics": {},
            "trained_model_paths": {},
            "task_type": TaskType.CLASSIFICATION,
            "node_results": [],
            "errors": [],
        }
        result = evaluation_agent(state)
        nr = result["node_results"][0]
        assert nr["status"] == WorkflowNodeStatus.ERROR
        assert "error" in nr

    def test_errors_list_populated_on_failure(self):
        state: WorkflowState = {
            "training_metrics": {},
            "trained_model_paths": {},
            "task_type": TaskType.CLASSIFICATION,
            "node_results": [],
            "errors": [],
        }
        result = evaluation_agent(state)
        assert len(result["errors"]) > 0

    def test_existing_node_results_preserved(self):
        prior: Any = [{"node_name": "training", "status": WorkflowNodeStatus.DONE}]
        state = _cls_state(exp_id="exp-preserve")
        state["node_results"] = prior
        result = evaluation_agent(state)
        assert len(result["node_results"]) == 2
        assert result["node_results"][0]["node_name"] == "training"
        assert result["node_results"][1]["node_name"] == "evaluation"

    def test_evaluation_results_written_to_state(self):
        result = evaluation_agent(_cls_state(exp_id="exp-ev"))
        assert "evaluation_results" in result
        assert result["evaluation_results"]["best_model"] == "xgboost"

    def test_best_model_name_written_to_state(self):
        result = evaluation_agent(_cls_state(xgb_f1=0.90, exp_id="exp-best"))
        assert result["best_model_name"] == "xgboost"

    def test_best_model_path_written_to_state(self):
        result = evaluation_agent(_cls_state(exp_id="exp-path"))
        assert "best_model_path" in result
        assert result["best_model_path"] == "/tmp/xgb.pkl"


# ── 3. Ranking logic ──────────────────────────────────────────────────────────

class TestRankingLogic:

    def test_tie_broken_deterministically(self):
        """When two models have identical scores, ranking must be stable."""
        state: WorkflowState = {
            "training_metrics": {
                "model_a": _cls_metrics(0.80, 0.80),
                "model_b": _cls_metrics(0.80, 0.80),
            },
            "trained_model_paths": {
                "model_a": "/tmp/a.pkl",
                "model_b": "/tmp/b.pkl",
            },
            "task_type": TaskType.CLASSIFICATION,
        }
        r1 = _run_evaluation(state)
        r2 = _run_evaluation(state)
        assert r1["best_model_name"] == r2["best_model_name"]

    def test_regression_best_is_lowest_rmse(self):
        state = _reg_state(rf_rmse=5.0, lr_rmse=20.0, xgb_rmse=3.0)
        result = _run_evaluation(state)
        assert result["best_model_name"] == "xgboost"
        assert result["evaluation_results"]["best_score"] == 3.0

    def test_classification_best_is_highest_f1(self):
        state = _cls_state(rf_f1=0.95, lr_f1=0.70, xgb_f1=0.88)
        result = _run_evaluation(state)
        assert result["best_model_name"] == "random_forest"
        assert result["evaluation_results"]["best_score"] == 0.95

    def test_rank_1_is_always_best(self):
        result = _run_evaluation(_cls_state(rf_f1=0.82, lr_f1=0.78, xgb_f1=0.85))
        ranked = result["evaluation_results"]["ranked_models"]
        rank1 = next(r for r in ranked if r["rank"] == 1)
        assert rank1["model"] == "xgboost"

    def test_bookkeeping_keys_excluded_from_ranking_score(self):
        """n_train, n_test, duration must not affect the primary score."""
        state: WorkflowState = {
            "training_metrics": {
                "model_a": {
                    "f1_weighted": 0.80,
                    "accuracy": 0.80,
                    "train_duration_seconds": 999.0,  # large — must not affect rank
                    "n_train": 10000,
                    "n_test": 2000,
                },
                "model_b": {
                    "f1_weighted": 0.75,
                    "accuracy": 0.75,
                    "train_duration_seconds": 0.001,
                    "n_train": 10000,
                    "n_test": 2000,
                },
            },
            "trained_model_paths": {
                "model_a": "/tmp/a.pkl",
                "model_b": "/tmp/b.pkl",
            },
            "task_type": TaskType.CLASSIFICATION,
        }
        result = _run_evaluation(state)
        assert result["best_model_name"] == "model_a"


# ── 4. Full graph integration ─────────────────────────────────────────────────

class TestGraphEvaluation:

    def test_evaluation_node_runs_in_graph(self, model_dir, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eval-001",
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
        assert "evaluation" in node_names
        eval_nr = next(
            nr for nr in final["node_results"] if nr["node_name"] == "evaluation"
        )
        assert eval_nr["status"] == WorkflowNodeStatus.DONE

    def test_best_model_name_in_final_state(self, model_dir, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eval-002",
            "dataset_id": "ds-002",
            "dataset_path": path,
            "target_column": "target",
            "task_type": TaskType.CLASSIFICATION,
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "best_model_name" in final
        assert final["best_model_name"] in {
            "random_forest", "logistic_regression", "xgboost"
        }

    def test_evaluation_results_in_final_state(self, model_dir, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eval-003",
            "dataset_id": "ds-003",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        assert "evaluation_results" in final
        ev = final["evaluation_results"]
        assert "ranked_models" in ev
        assert "comparison_table" in ev
        assert len(ev["ranked_models"]) == 3

    def test_pipeline_order_training_then_evaluation(self, model_dir, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eval-004",
            "dataset_id": "ds-004",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        node_names = [nr["node_name"] for nr in final["node_results"]]
        assert node_names.index("training") < node_names.index("evaluation")

    def test_best_model_path_points_to_existing_file(self, model_dir, artifact_dir):
        from app.workflows.graph import pipeline_graph
        path = _write_csv(_classification_df())
        initial: WorkflowState = {
            "experiment_id": "graph-eval-005",
            "dataset_id": "ds-005",
            "dataset_path": path,
            "target_column": "target",
            "pipeline_config": {},
            "node_results": [],
            "errors": [],
        }
        final = pipeline_graph.invoke(initial)
        best_path = final.get("best_model_path", "")
        assert best_path != ""
        assert Path(best_path).exists(), f"Best model file not found: {best_path}"
