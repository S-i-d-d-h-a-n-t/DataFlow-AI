"""
LangGraph node wrappers.

Each function here is a thin adapter that:
  1. Calls the corresponding agent's pure function.
  2. Is registered as a named node in the StateGraph.

Keeping agent logic in app/agents/ and graph wiring in app/workflows/
means agents are independently testable without importing LangGraph.

Nodes added per phase:
  Phase 3  → planner_node          ✅
  Phase 4  → eda_node              ✅
  Phase 5  → cleaning_node         ✅
  Phase 6  → feature_node          ✅
  Phase 7  → training_node         ✅
  Phase 8  → evaluation_node       ✅
  Phase 9  → report_node           ✅
"""

from __future__ import annotations

from typing import Any

from app.state.workflow_state import WorkflowState
from app.agents.planner_agent import planner_agent
from app.agents.eda_agent import eda_agent
from app.agents.cleaning_agent import cleaning_agent
from app.agents.feature_agent import feature_agent
from app.agents.training_agent import training_agent
from app.agents.evaluation_agent import evaluation_agent
from app.agents.report_agent import report_agent
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def planner_node(state: WorkflowState) -> dict[str, Any]:
    """Wrapper for the Planner Agent."""
    return planner_agent(state)


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def eda_node(state: WorkflowState) -> dict[str, Any]:
    """Wrapper for the EDA Agent."""
    return eda_agent(state)


# ── Phase 5 ───────────────────────────────────────────────────────────────────

def cleaning_node(state: WorkflowState) -> dict[str, Any]:
    """Wrapper for the Cleaning Agent."""
    return cleaning_agent(state)


# ── Phase 6 ───────────────────────────────────────────────────────────────────

def feature_node(state: WorkflowState) -> dict[str, Any]:
    """Wrapper for the Feature Engineering Agent."""
    return feature_agent(state)


# ── Phase 7 ───────────────────────────────────────────────────────────────────

def training_node(state: WorkflowState) -> dict[str, Any]:
    """
    Wrapper for the Training Agent.
    Parallel model training is handled internally via ThreadPoolExecutor.
    """
    return training_agent(state)


# ── Phase 8 ───────────────────────────────────────────────────────────────────

def evaluation_node(state: WorkflowState) -> dict[str, Any]:
    """Wrapper for the Evaluation Agent."""
    return evaluation_agent(state)


# ── Phase 9 ───────────────────────────────────────────────────────────────────

def report_node(state: WorkflowState) -> dict[str, Any]:
    """Wrapper for the Report Agent."""
    return report_agent(state)
