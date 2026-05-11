"""
Central API router.
All versioned sub-routers are registered here and mounted in main.py.

Complete route map:
  /health                              → liveness probe
  /ready                               → readiness probe (DB check)
  /api/v1/datasets/upload              → CSV upload
  /api/v1/datasets                     → list / get / patch / delete datasets
  /api/v1/datasets/{id}/preview        → row preview
  /api/v1/datasets/{id}/structure      → structural analysis
  /api/v1/workflow/run                 → trigger pipeline run
  /api/v1/workflow/schema              → graph topology
  /api/v1/workflow/{id}/status         → poll experiment status
  /api/v1/workflow/{id}                → full experiment detail
  /api/v1/workflow                     → list experiments for a dataset
  /api/v1/reports/{id}                 → report metadata
  /api/v1/reports/{id}/markdown        → raw Markdown report
  /api/v1/reports/{id}/html            → rendered HTML report
"""

from fastapi import APIRouter

from app.api import health, upload, workflow, reports
from app.core.config import settings

api_router = APIRouter()

# ── Infrastructure (no version prefix) ───────────────────────────────────────
api_router.include_router(health.router)

# ── Phase 2 — Dataset management ─────────────────────────────────────────────
api_router.include_router(
    upload.router,
    prefix=f"{settings.API_V1_PREFIX}/datasets",
    tags=["Datasets"],
)

# ── Phase 11 — Workflow trigger & monitoring ──────────────────────────────────
api_router.include_router(
    workflow.router,
    prefix=f"{settings.API_V1_PREFIX}/workflow",
    tags=["Workflow"],
)

# ── Phase 11 — Report retrieval ───────────────────────────────────────────────
api_router.include_router(
    reports.router,
    prefix=f"{settings.API_V1_PREFIX}/reports",
    tags=["Reports"],
)
