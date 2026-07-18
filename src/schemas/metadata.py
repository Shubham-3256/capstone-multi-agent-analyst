"""Schemas for tracking dataset profiling metadata and column maps."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.schemas.dataset import ColumnInfo


class DatasetMetadata(BaseModel):
    """Complete metadata profile for a dataset file inside the workspace."""

    dataset_id: str = Field(
        ...,
        description="UUID identifier for the dataset database record",
        examples=["4a4f7a9a-7a9a-4f7a-9a7a-9a7a9a7a9a7a"]
    )
    filename: str = Field(
        ...,
        description="Sanitized dataset filename",
        examples=["boston_housing.csv"]
    )
    filepath: str = Field(
        ...,
        description="Absolute or workspace-relative path to the file location",
        examples=["/app/workspace/uploads/boston_housing.csv"]
    )
    file_hash: str = Field(
        ...,
        description="SHA-256 checksum of the file content",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"]
    )
    file_size_bytes: int = Field(
        ...,
        description="Physical dataset storage size on disk in bytes",
        examples=[254109]
    )
    row_count: int = Field(
        ...,
        description="Total row records in the dataset table",
        examples=[506]
    )
    column_count: int = Field(
        ...,
        description="Total columns in the dataset table",
        examples=[14]
    )
    columns: Dict[str, ColumnInfo] = Field(
        ...,
        description="Key-value mapping of column name to its profile metadata"
    )
    missing_value_count: int = Field(
        ...,
        description="Total count of all missing cells in the dataset",
        examples=[12]
    )
    duplicate_rows_count: int = Field(
        ...,
        description="Total count of duplicate rows in the dataset",
        examples=[0]
    )
    memory_usage_bytes: int = Field(
        ...,
        description="Estimated pandas RAM footprint in bytes",
        examples=[56800]
    )
    status: str = Field(
        default="uploaded",
        description="Current processing status of the dataset in the workflow",
        examples=["uploaded"]
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp indicating when the metadata profile was extracted"
    )
