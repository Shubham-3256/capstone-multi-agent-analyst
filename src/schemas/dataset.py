"""Schemas representing column profiles and dataset summaries."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """Schema representing structural profiles and statistics for a single column."""

    name: str = Field(
        ...,
        description="Name of the target column",
        examples=["age"]
    )
    dtype: str = Field(
        ...,
        description="Data type inferred for the column values",
        examples=["int64"]
    )
    non_null_count: int = Field(
        ...,
        description="Count of non-null records in the column",
        examples=[1000]
    )
    null_count: int = Field(
        ...,
        description="Count of null records in the column",
        examples=[0]
    )
    null_percentage: float = Field(
        ...,
        description="Percentage of records that are null, ranging from 0.0 to 100.0",
        examples=[0.0]
    )
    unique_count: int = Field(
        ...,
        description="Number of distinct unique values in the column",
        examples=[45]
    )
    sample_values: list[Any] = Field(
        default=[],
        description="Small list of representative sample values from the column",
        examples=[[25, 42, 19, 64]]
    )
    statistics: dict[str, Any] = Field(
        default={},
        description="Descriptive statistics calculated for the column (e.g. mean, std, min, max, etc.)",
        examples=[{"mean": 38.5, "min": 18, "max": 90}]
    )


class TargetInfo(BaseModel):
    """Schema tracking label target specifications for modeling activities."""

    column_name: str = Field(
        ...,
        description="Name of the selected target label column",
        examples=["target"]
    )
    task_type: str = Field(
        ...,
        description="Inferred modeling task type (e.g. classification, regression)",
        examples=["classification"]
    )
    classes: list[Any] | None = Field(
        default=None,
        description="List of distinct classes for classification tasks",
        examples=[[0, 1]]
    )
    class_distribution: dict[str, float] | None = Field(
        default=None,
        description="Ratio details of target values (e.g. class proportions)",
        examples=[{"0": 0.82, "1": 0.18}]
    )


class DatasetSummary(BaseModel):
    """Schema outlining dimensions and column breakdowns for a dataset."""

    filename: str = Field(
        ...,
        description="Sanitized name of the dataset file",
        examples=["dataset.csv"]
    )
    row_count: int = Field(
        ...,
        description="Total number of rows in the dataset",
        examples=[1000]
    )
    column_count: int = Field(
        ...,
        description="Total number of columns in the dataset",
        examples=[10]
    )
    file_size_bytes: int = Field(
        ...,
        description="Physical file size in bytes",
        examples=[1048576]
    )
    columns: list[ColumnInfo] = Field(
        ...,
        description="Detailed profiling metrics for each column",
    )
    duplicate_rows_count: int = Field(
        default=0,
        description="Number of identical row duplicates in the dataset",
        examples=[0]
    )
    memory_usage_bytes: int = Field(
        ...,
        description="Estimated RAM consumption for loading this dataset into memory",
        examples=[412056]
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Profiling evaluation timestamp"
    )
