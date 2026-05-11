"""
Shared pytest fixtures for the test suite.

Strategy:
  - Use a file-based SQLite database (test.db) so tables persist across
    the session-scoped TestClient and function-scoped reset_db fixture.
  - Patch app.core.database.engine so the lifespan's create_all_tables()
    targets SQLite instead of the Docker 'db' PostgreSQL host.
  - Override the get_db FastAPI dependency so every request uses SQLite.
  - Tables are dropped and recreated before each test for full isolation.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Import models BEFORE Base.metadata is used so all tables are registered
from app.core.database import Base, get_db
from app.models import dataset as _ds   # noqa: F401
from app.models import experiment as _exp  # noqa: F401

# ── Test engine (file-based SQLite so it survives across connections) ─────────
TEST_DB_PATH = "test_temp.db"
TEST_DB_URL = f"sqlite:///{TEST_DB_PATH}"

_test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)
_TestingSessionLocal = sessionmaker(
    bind=_test_engine, autocommit=False, autoflush=False
)

# Create tables once at import time
Base.metadata.create_all(bind=_test_engine)


def _override_get_db():
    """FastAPI dependency override — yields a SQLite test session."""
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Per-test table reset ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_db():
    """Drop and recreate all tables before each test for full isolation."""
    Base.metadata.drop_all(bind=_test_engine)
    Base.metadata.create_all(bind=_test_engine)
    yield


# ── Session-scoped TestClient ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client():
    """
    Build a TestClient that:
      1. Patches the SQLAlchemy engine used by create_all_tables() → SQLite
      2. Overrides the get_db dependency → SQLite sessions
    """
    from app.main import app
    import app.core.database as db_module

    app.dependency_overrides[get_db] = _override_get_db

    with patch.object(db_module, "engine", _test_engine):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    app.dependency_overrides.clear()


# ── Cleanup test DB file after session ───────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Remove the SQLite test database file after the test run."""
    _test_engine.dispose()
    try:
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
    except OSError:
        pass  # Windows may hold a lock briefly — not a real failure
