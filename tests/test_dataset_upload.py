"""
Phase 2 tests — Dataset upload, parsing, and management endpoints.
All tests use the in-memory SQLite database configured in conftest.py.
"""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_csv(rows: int = 10) -> bytes:
    df = pd.DataFrame({
        "age": range(rows),
        "salary": [float(i) * 1000 for i in range(rows)],
        "department": ["eng" if i % 2 == 0 else "sales" for i in range(rows)],
        "target": [i % 2 for i in range(rows)],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _upload(client: TestClient, filename: str = "test.csv",
            content: bytes | None = None, name: str | None = None):
    data = {}
    if name:
        data["name"] = name
    return client.post(
        "/api/v1/datasets/upload",
        files={"file": (filename, content or _make_csv(), "text/csv")},
        data=data,
    )


def _mock_path(path_str: str = "/tmp/test.csv"):
    """Return a MagicMock that behaves like a Path for file saving."""
    tmp = MagicMock()
    tmp.__str__ = lambda s: path_str
    tmp.parent.mkdir = MagicMock()
    tmp.write_bytes = MagicMock()
    return tmp


# ── Upload tests ──────────────────────────────────────────────────────────────

class TestUpload:
    def test_upload_valid_csv_returns_201(self, test_client):
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=1024):
            resp = _upload(test_client)

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["num_rows"] == 10
        assert body["num_columns"] == 4
        assert "age" in body["column_names"]
        assert body["column_dtypes"]["age"] == "numeric"
        assert body["column_dtypes"]["department"] == "categorical"

    def test_upload_sets_custom_name(self, test_client):
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=512):
            resp = _upload(test_client, name="my_custom_name")

        assert resp.status_code == 201
        assert resp.json()["name"] == "my_custom_name"

    def test_upload_defaults_name_to_filename_stem(self, test_client):
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=512):
            resp = _upload(test_client, filename="my_data.csv")

        assert resp.status_code == 201
        assert resp.json()["name"] == "my_data"

    def test_upload_rejects_non_csv(self, test_client):
        resp = test_client.post(
            "/api/v1/datasets/upload",
            files={"file": ("data.xlsx", b"fake", "application/vnd.ms-excel")},
        )
        assert resp.status_code == 415

    def test_upload_rejects_empty_csv(self, test_client):
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()):
            resp = test_client.post(
                "/api/v1/datasets/upload",
                files={"file": ("empty.csv", b"col1,col2\n", "text/csv")},
            )
        assert resp.status_code == 422

    def test_upload_rejects_oversized_file(self, test_client):
        big = b"a,b\n" + b"1,2\n" * 1000
        with patch("app.services.dataset_service._MAX_FILE_SIZE_BYTES", 10):
            resp = test_client.post(
                "/api/v1/datasets/upload",
                files={"file": ("big.csv", big, "text/csv")},
            )
        assert resp.status_code == 413

    def test_upload_stores_column_dtypes(self, test_client):
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=200):
            resp = _upload(test_client)

        dtypes = resp.json()["column_dtypes"]
        assert dtypes["salary"] == "numeric"
        assert dtypes["department"] == "categorical"


# ── List tests ────────────────────────────────────────────────────────────────

class TestList:
    def test_list_empty(self, test_client):
        resp = test_client.get("/api/v1/datasets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_uploaded_datasets(self, test_client):
        for i in range(3):
            with patch("app.services.dataset_service.dataset_upload_path",
                       return_value=_mock_path(f"/tmp/ds{i}.csv")), \
                 patch("app.services.dataset_service.get_file_size", return_value=100):
                _upload(test_client, name=f"dataset_{i}")

        resp = test_client.get("/api/v1/datasets")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_list_pagination_limit(self, test_client):
        for i in range(5):
            with patch("app.services.dataset_service.dataset_upload_path",
                       return_value=_mock_path(f"/tmp/ds{i}.csv")), \
                 patch("app.services.dataset_service.get_file_size", return_value=100):
                _upload(test_client, name=f"ds_{i}")

        resp = test_client.get("/api/v1/datasets?skip=0&limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_summary_fields(self, test_client):
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=100):
            _upload(test_client)

        body = test_client.get("/api/v1/datasets").json()
        assert "id" in body[0]
        assert "name" in body[0]
        assert "num_rows" in body[0]
        assert "num_columns" in body[0]
        # file_path should NOT appear in summary
        assert "file_path" not in body[0]


# ── Get / Preview / Structure tests ──────────────────────────────────────────

class TestGetPreviewStructure:
    def _create(self, test_client) -> str:
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=200):
            resp = _upload(test_client)
        assert resp.status_code == 201
        return resp.json()["id"]

    def test_get_existing_dataset(self, test_client):
        dataset_id = self._create(test_client)
        resp = test_client.get(f"/api/v1/datasets/{dataset_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == dataset_id

    def test_get_nonexistent_returns_404(self, test_client):
        resp = test_client.get("/api/v1/datasets/does-not-exist")
        assert resp.status_code == 404

    def test_preview_returns_correct_row_count(self, test_client):
        dataset_id = self._create(test_client)
        with patch("app.services.dataset_service.DatasetService._load_from_disk",
                   return_value=pd.read_csv(io.BytesIO(_make_csv(20)))):
            resp = test_client.get(f"/api/v1/datasets/{dataset_id}/preview?n_rows=3")

        assert resp.status_code == 200
        body = resp.json()
        assert body["num_rows_shown"] == 3
        assert len(body["rows"]) == 3

    def test_preview_includes_column_names(self, test_client):
        dataset_id = self._create(test_client)
        with patch("app.services.dataset_service.DatasetService._load_from_disk",
                   return_value=pd.read_csv(io.BytesIO(_make_csv(10)))):
            resp = test_client.get(f"/api/v1/datasets/{dataset_id}/preview")

        assert "columns" in resp.json()

    def test_structure_returns_all_keys(self, test_client):
        dataset_id = self._create(test_client)
        with patch("app.services.dataset_service.DatasetService._load_from_disk",
                   return_value=pd.read_csv(io.BytesIO(_make_csv(50)))):
            resp = test_client.get(f"/api/v1/datasets/{dataset_id}/structure")

        assert resp.status_code == 200
        body = resp.json()
        for key in ("column_dtypes", "missing_values", "cardinality",
                    "column_stats", "memory_usage_mb", "num_rows", "num_columns"):
            assert key in body, f"Missing key: {key}"

    def test_structure_nonexistent_returns_404(self, test_client):
        resp = test_client.get("/api/v1/datasets/ghost/structure")
        assert resp.status_code == 404


# ── Update / Delete tests ─────────────────────────────────────────────────────

class TestUpdateDelete:
    def _create(self, test_client) -> str:
        with patch("app.services.dataset_service.dataset_upload_path",
                   return_value=_mock_path()), \
             patch("app.services.dataset_service.get_file_size", return_value=200):
            resp = _upload(test_client)
        return resp.json()["id"]

    def test_patch_name(self, test_client):
        dataset_id = self._create(test_client)
        resp = test_client.patch(
            f"/api/v1/datasets/{dataset_id}",
            json={"name": "renamed_dataset"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "renamed_dataset"

    def test_patch_description(self, test_client):
        dataset_id = self._create(test_client)
        resp = test_client.patch(
            f"/api/v1/datasets/{dataset_id}",
            json={"description": "A test dataset"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "A test dataset"

    def test_patch_nonexistent_returns_404(self, test_client):
        resp = test_client.patch("/api/v1/datasets/ghost-id", json={"name": "x"})
        assert resp.status_code == 404

    def test_delete_returns_204(self, test_client):
        dataset_id = self._create(test_client)
        with patch("app.services.dataset_service.delete_file"):
            resp = test_client.delete(f"/api/v1/datasets/{dataset_id}")
        assert resp.status_code == 204

    def test_delete_then_get_returns_404(self, test_client):
        dataset_id = self._create(test_client)
        with patch("app.services.dataset_service.delete_file"):
            test_client.delete(f"/api/v1/datasets/{dataset_id}")
        resp = test_client.get(f"/api/v1/datasets/{dataset_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, test_client):
        resp = test_client.delete("/api/v1/datasets/ghost-id")
        assert resp.status_code == 404
