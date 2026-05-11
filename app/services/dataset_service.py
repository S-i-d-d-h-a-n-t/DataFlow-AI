"""
Dataset Service — business logic for dataset upload and management.

Responsibilities:
  - Validate the uploaded file (extension, size, parseability)
  - Save the file to disk via file_manager
  - Parse the CSV with pandas and extract structural metadata
  - Persist a Dataset record via DatasetRepository
  - Coordinate delete (file + DB row)

Rules:
  - No SQLAlchemy imports — all DB access goes through DatasetRepository.
  - No FastAPI imports — this layer is framework-agnostic.
  - Raises domain-level exceptions that the API layer translates to HTTP errors.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import pandas as pd

from app.models.dataset import Dataset
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import DatasetUpdate
from app.utils.file_manager import (
    dataset_upload_path,
    get_file_size,
    delete_file,
    ensure_directories,
)
from app.utils.dataframe_utils import infer_column_dtypes
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_ALLOWED_EXTENSIONS = {".csv"}
_MAX_FILE_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
_CSV_READ_KWARGS: dict = {
    "low_memory": False,
    "on_bad_lines": "warn",
}


# ── Domain exceptions ─────────────────────────────────────────────────────────

class DatasetServiceError(Exception):
    """Base class for dataset service errors."""


class InvalidFileTypeError(DatasetServiceError):
    """Raised when the uploaded file is not a supported type."""


class FileTooLargeError(DatasetServiceError):
    """Raised when the uploaded file exceeds the size limit."""


class ParseError(DatasetServiceError):
    """Raised when pandas cannot parse the uploaded file."""


class DatasetNotFoundError(DatasetServiceError):
    """Raised when a requested dataset does not exist."""


# ── Service ───────────────────────────────────────────────────────────────────

class DatasetService:
    """Orchestrates all dataset-related business operations."""

    def __init__(self, repo: DatasetRepository) -> None:
        self._repo = repo

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload(
        self,
        filename: str,
        file_content: bytes,
        name: str | None = None,
        description: str | None = None,
    ) -> Dataset:
        """
        Validate, save, parse, and register a new dataset.

        Args:
            filename:     Original filename from the upload (used for extension check).
            file_content: Raw bytes of the uploaded file.
            name:         Human-readable name; defaults to the filename stem.
            description:  Optional free-text description.

        Returns:
            The persisted Dataset ORM instance.

        Raises:
            InvalidFileTypeError: File extension is not .csv
            FileTooLargeError:    File exceeds _MAX_FILE_SIZE_BYTES
            ParseError:           pandas cannot read the file as CSV
        """
        ensure_directories()

        # 1. Validate extension
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError(
                f"Unsupported file type '{suffix}'. Only CSV files are accepted."
            )

        # 2. Validate size
        if len(file_content) > _MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size {len(file_content) / 1_048_576:.1f} MB exceeds the "
                f"{settings.MAX_UPLOAD_SIZE_MB} MB limit."
            )

        # 3. Parse with pandas (in-memory — no disk write yet)
        df = self._parse_csv(file_content, filename)

        # 4. Save to disk
        dest_path = dataset_upload_path(filename)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(file_content)
        logger.info(f"Dataset file saved: {dest_path}")

        # 5. Extract metadata
        column_dtypes = infer_column_dtypes(df)
        dataset_name = name or Path(filename).stem

        # 6. Persist to DB
        dataset = Dataset(
            name=dataset_name,
            file_path=str(dest_path),
            file_size_bytes=get_file_size(dest_path),
            num_rows=len(df),
            num_columns=len(df.columns),
            column_names=list(df.columns),
            column_dtypes=column_dtypes,
            description=description,
        )
        return self._repo.create(dataset)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, dataset_id: str) -> Dataset:
        """Return a Dataset or raise DatasetNotFoundError."""
        dataset = self._repo.get_by_id(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")
        return dataset

    def list_all(self, skip: int = 0, limit: int = 100) -> Sequence[Dataset]:
        return self._repo.list_all(skip=skip, limit=limit)

    def count(self) -> int:
        return self._repo.count()

    # ── Preview ───────────────────────────────────────────────────────────────

    def get_preview(self, dataset_id: str, n_rows: int = 5) -> dict:
        """
        Return a JSON-safe preview of the first n_rows of a dataset.
        Reads the file from disk on demand.
        """
        dataset = self.get_by_id(dataset_id)
        df = self._load_from_disk(dataset.file_path)
        sample = df.head(n_rows)
        # Replace NaN/Inf for JSON safety
        sample = sample.where(pd.notnull(sample), other=None)
        return {
            "dataset_id": dataset_id,
            "num_rows_shown": len(sample),
            "total_rows": dataset.num_rows,
            "columns": dataset.column_names,
            "rows": sample.to_dict(orient="records"),
        }

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dataset_id: str, payload: DatasetUpdate) -> Dataset:
        """Patch mutable metadata fields on an existing dataset."""
        dataset = self.get_by_id(dataset_id)
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            return dataset
        return self._repo.update(dataset, updates)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, dataset_id: str) -> None:
        """Remove the dataset file from disk and its DB record."""
        dataset = self.get_by_id(dataset_id)
        file_path = dataset.file_path
        self._repo.delete(dataset)
        delete_file(file_path)
        logger.info(f"Dataset {dataset_id} deleted (file + DB row).")

    # ── Structure detection ───────────────────────────────────────────────────

    def detect_structure(self, dataset_id: str) -> dict:
        """
        Return a detailed structural analysis of the dataset.
        Used by the Planner Agent in Phase 3.
        """
        from app.utils.dataframe_utils import (
            missing_value_report,
            cardinality_report,
            column_stats,
        )

        dataset = self.get_by_id(dataset_id)
        df = self._load_from_disk(dataset.file_path)

        return {
            "dataset_id": dataset_id,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_dtypes": infer_column_dtypes(df),
            "missing_values": missing_value_report(df),
            "cardinality": cardinality_report(df),
            "column_stats": column_stats(df),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_csv(content: bytes, filename: str) -> pd.DataFrame:
        """
        Attempt to parse CSV bytes with multiple encoding fallbacks.
        Raises ParseError with a descriptive message on failure.
        """
        encodings = ["utf-8", "latin-1", "cp1252"]
        last_exc: Exception | None = None

        for enc in encodings:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=enc,
                    **_CSV_READ_KWARGS,
                )
                if df.empty:
                    raise ParseError(f"'{filename}' parsed to an empty DataFrame.")
                logger.debug(
                    f"Parsed '{filename}' with encoding={enc} "
                    f"shape=({len(df)}, {len(df.columns)})"
                )
                return df
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_exc = exc
                continue

        raise ParseError(
            f"Could not parse '{filename}' as CSV. "
            f"Tried encodings: {encodings}. Last error: {last_exc}"
        )

    @staticmethod
    def _load_from_disk(file_path: str) -> pd.DataFrame:
        """Load a previously saved CSV from disk."""
        path = Path(file_path)
        if not path.exists():
            raise DatasetServiceError(
                f"Dataset file not found on disk: {file_path}"
            )
        return pd.read_csv(path, **_CSV_READ_KWARGS)
