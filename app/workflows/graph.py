"""
LangGraph StateGraph — full ML pipeline (Phase 10: hardened integration).

Graph topology:

    START
      │
      ▼
  [planner] ──(error)──► [handle_error] ──► END
      │(ok)
      ▼
    [eda]
      │
      ▼
  [cleaning]
      │
      ▼
  [feature_engineering]
      │
      ▼
  [training]   ← parallel RF / LR / XGB via ThreadPoolExecutor
      │
      ▼
  [evaluation]
      │
      ▼
   [report]
      │
      ▼
     END

Error handling:
  - After planner: hard error → handle_error (skips all downstream nodes).
  - All other nodes: errors are recorded in state["errors"] but execution
    continues so partial results are always persisted.
  - handle_error_node is a terminal sink — it logs and passes errors through.

State contract:
  - Every node reads from WorkflowState and returns a dict patch.
  - node_results is append-only — each agent adds exactly one NodeResult.
  - errors is append-only — agents add to it on failure, never clear it.

The graph is compiled once at module import and reused across all requests.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END, START

from app.state.workflow_state import WorkflowState
from app.workflows.nodes import (
    planner_node,
    eda_node,
    cleaning_node,
    feature_node,
    training_node,
    evaluation_node,
    report_node,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Error handler node ────────────────────────────────────────────────────────

def handle_error_node(state: WorkflowState) -> dict[str, Any]:
    """
    Terminal error sink — reached only when the Planner fails hard.
    Logs the failure and passes errors through so the orchestrator can
    persist the failure reason to the database.
    """
    errors = state.get("errors") or ["Unknown error"]
    logger.error(f"[handle_error] Pipeline aborted: {errors}")
    return {"errors": errors}


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_planner(state: WorkflowState) -> str:
    """
    Hard gate after the Planner.
    If the Planner produced errors (e.g. bad dataset path, missing target),
    abort the pipeline immediately — there is nothing useful to do downstream.
    """
    errors = state.get("errors") or []
    if errors:
        logger.warning(
            f"[router] Planner failed — aborting pipeline. errors={errors}"
        )
        return "handle_error"
    return "eda"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_pipeline_graph() -> Any:
    """
    Construct and compile the full ML pipeline StateGraph.

    Returns a CompiledStateGraph ready to be invoked with an initial WorkflowState.
    """
    graph = StateGraph(WorkflowState)

    # ── Register all nodes ────────────────────────────────────────────────────
    graph.add_node("planner",             planner_node)
    graph.add_node("eda",                 eda_node)
    graph.add_node("cleaning",            cleaning_node)
    graph.add_node("feature_engineering", feature_node)
    graph.add_node("training",            training_node)
    graph.add_node("evaluation",          evaluation_node)
    graph.add_node("report",              report_node)
    graph.add_node("handle_error",        handle_error_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "planner")

    # ── Conditional routing after planner ────────────────────────────────────
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "eda":          "eda",
            "handle_error": "handle_error",
        },
    )

    # ── Sequential pipeline (errors are soft — execution continues) ───────────
    graph.add_edge("eda",                 "cleaning")
    graph.add_edge("cleaning",            "feature_engineering")
    graph.add_edge("feature_engineering", "training")
    graph.add_edge("training",            "evaluation")
    graph.add_edge("evaluation",          "report")
    graph.add_edge("report",              END)
    graph.add_edge("handle_error",        END)

    compiled = graph.compile()
    logger.info("Pipeline graph compiled successfully.")
    return compiled


def get_graph_schema() -> dict[str, Any]:
    """
    Return a JSON-serialisable description of the pipeline graph.
    Used by the /api/v1/workflow/schema endpoint.
    """
    return {
        "nodes": [
            {"name": "planner",             "phase": 3, "description": "Analyse dataset, infer task type, build execution plan"},
            {"name": "eda",                 "phase": 4, "description": "Exploratory data analysis — statistics and charts"},
            {"name": "cleaning",            "phase": 5, "description": "Imputation, deduplication, outlier clipping"},
            {"name": "feature_engineering", "phase": 6, "description": "Encoding, scaling, MI feature selection"},
            {"name": "training",            "phase": 7, "description": "Parallel model training (RF, LR/LinReg, XGBoost)"},
            {"name": "evaluation",          "phase": 8, "description": "Model ranking by primary metric"},
            {"name": "report",              "phase": 9, "description": "Markdown + HTML report generation"},
            {"name": "handle_error",        "phase": None, "description": "Terminal error sink"},
        ],
        "edges": [
            {"from": "START",               "to": "planner",             "condition": None},
            {"from": "planner",             "to": "eda",                 "condition": "no errors"},
            {"from": "planner",             "to": "handle_error",        "condition": "errors present"},
            {"from": "eda",                 "to": "cleaning",            "condition": None},
            {"from": "cleaning",            "to": "feature_engineering", "condition": None},
            {"from": "feature_engineering", "to": "training",            "condition": None},
            {"from": "training",            "to": "evaluation",          "condition": None},
            {"from": "evaluation",          "to": "report",              "condition": None},
            {"from": "report",              "to": "END",                 "condition": None},
            {"from": "handle_error",        "to": "END",                 "condition": None},
        ],
        "parallel_nodes": ["training"],
        "entry_point": "planner",
        "terminal_nodes": ["report", "handle_error"],
    }


# ── Module-level singleton ────────────────────────────────────────────────────
# Compiled once at import time; reused for every pipeline invocation.
pipeline_graph = build_pipeline_graph()
