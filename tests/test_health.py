"""
Phase 1 smoke tests — health endpoints.
"""

from unittest.mock import patch, MagicMock


class TestHealth:
    def test_liveness_check(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "app" in body

    def test_readiness_check_db_ok(self, test_client):
        response = test_client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        # SQLite test DB is always reachable
        assert body["status"] == "ok"
        assert body["database"] == "ok"
