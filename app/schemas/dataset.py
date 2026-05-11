"""
Pydantic schemas for the Dataset resource.

Separation of concerns:
  - DatasetCreate  → validates incoming request bodies
  - DatasetRead    → serialises ORM rows for API responses
  - DatasetSummary → lightweight list-view projection
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class DatasetCreate(BaseModel):
    """Fields accepted when registering a new dataset."""

    name: str = Field(..., min_length=1, max_length=255, examples=["titanic_train"])
    description: str | None = Field(None, max_length=1000)


class DatasetRead(BaseModel):
    """Full dataset representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    file_path: str
    file_size_bytes: int
    num_rows: int | None
    num_columns: int | None
    column_names: list[str] | None
    column_dtypes: dict[str, Any] | None
    description: str | None
    created_at: datetime
    updated_at: datetime


class DatasetSummary(BaseModel):
    """Lightweight projection used in list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    num_rows: int | None
    num_columns: int | None
    created_at: datetime


class DatasetUpdate(BaseModel):
    """Fields that can be patched after upload."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
