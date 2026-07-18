"""Data schemas for the Data Intelligence Agent output parameters."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """Schema representing a single issue identified during dataset validation checks."""

    check_name: str = Field(
        ...,
        description="Name of the validation check executed",
        examples=["mixed_datatypes"]
    )
    severity: str = Field(
        ...,
        description="Severity classification of the issue (warning or error)",
        examples=["warning"]
    )
    column: Optional[str] = Field(
        default=None,
        description="Name of the column associated with the validation issue",
        examples=["age"]
    )
    message: str = Field(
        ...,
        description="Detailed description of the validation issue",
        examples=["Column 'age' contains mixed data types: float and string."]
    )


class ValidationReport(BaseModel):
    """Schema summarizing the results of dataset validation checks."""

    is_valid: bool = Field(
        ...,
        description="True if the dataset passes validation tests without errors",
        examples=[True]
    )
    issues: List[ValidationIssue] = Field(
        default=[],
        description="List of all warnings or errors detected in the dataset"
    )
    summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Consolidated counts of total issues, warnings, and errors",
        examples=[{"total_issues": 3, "warnings": 3, "errors": 0}]
    )


class CleaningAction(BaseModel):
    """Schema representing a single cleaning step applied to a dataset."""

    column: Optional[str] = Field(
        default=None,
        description="Name of the target column cleaned",
        examples=["age"]
    )
    action_type: str = Field(
        ...,
        description="Type of cleaning action executed (e.g. impute_missing, remove_outliers, change_dtype, trim_whitespace, drop_duplicates, drop_columns)",
        examples=["impute_missing"]
    )
    details: str = Field(
        ...,
        description="Descriptive details of the modification made",
        examples=["Imputed 12 missing values in column 'age' using the column's median value (38.5)."]
    )


class CleaningReport(BaseModel):
    """Schema compiling the changes applied during dataset cleaning operations."""

    transformations: List[CleaningAction] = Field(
        default=[],
        description="Chronological log list of all transformations applied"
    )
    initial_shape: List[int] = Field(
        ...,
        description="Data dimensions before cleaning [rows, columns]",
        examples=[[1000, 10]]
    )
    final_shape: List[int] = Field(
        ...,
        description="Data dimensions after cleaning [rows, columns]",
        examples=[[988, 10]]
    )
    duration_seconds: float = Field(
        ...,
        description="Time taken to perform the cleaning tasks in seconds",
        examples=[1.42]
    )


class Recommendation(BaseModel):
    """Schema representing modeling recommendations based on dataset properties."""

    title: str = Field(
        ...,
        description="Summary of the recommendation",
        examples=["Highly imbalanced target label detected"]
    )
    description: str = Field(
        ...,
        description="Detailed explanation and action plan for modeling",
        examples=["Target variable 'churn' shows an 82:18 skew. Consider using stratification splits or SMOTE during model training."]
    )
    severity: str = Field(
        ...,
        description="Severity classification level (info, warning, critical)",
        examples=["warning"]
    )
    column: Optional[str] = Field(
        default=None,
        description="Associated column name target",
        examples=["churn"]
    )


class ColumnProfile(BaseModel):
    """Schema for complete statistics and profile details of a single column."""

    name: str = Field(
        ...,
        description="Name of the column",
        examples=["age"]
    )
    dtype: str = Field(
        ...,
        description="Final clean data type of the column",
        examples=["int64"]
    )
    unique_count: int = Field(
        ...,
        description="Distinct count of unique values in the column",
        examples=[45]
    )
    null_count: int = Field(
        ...,
        description="Number of null records in the column",
        examples=[0]
    )
    null_percentage: float = Field(
        ...,
        description="Percentage of null values in the column",
        examples=[0.0]
    )
    numeric_summary: Optional[Dict[str, float]] = Field(
        default=None,
        description="Descriptive statistics summary for numerical columns",
        examples=[{"mean": 38.5, "min": 18.0, "max": 90.0}]
    )
    categorical_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Descriptive statistics summary for categorical columns",
        examples=[{"top": "Chicago", "freq": 15}]
    )
    date_summary: Optional[Dict[str, str]] = Field(
        default=None,
        description="Descriptive statistics summary for date columns",
        examples=[{"min": "2020-01-01", "max": "2025-12-31"}]
    )


class DatasetProfile(BaseModel):
    """Schema representing structural, statistical, and ML task profiles for a dataset."""

    row_count: int = Field(
        ...,
        description="Total row records in the dataset",
        examples=[1000]
    )
    column_count: int = Field(
        ...,
        description="Total column count in the dataset",
        examples=[10]
    )
    memory_usage_bytes: int = Field(
        ...,
        description="Estimated RAM consumption for loading this dataset in bytes",
        examples=[450122]
    )
    columns: Dict[str, ColumnProfile] = Field(
        ...,
        description="Descriptive profile structures for each column in the dataset"
    )
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Pearson correlation matrix mapping numeric columns",
        examples=[{"age": {"age": 1.0, "tenure": 0.42}}]
    )
    target_distribution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Distributions details of class targets",
        examples=[{"0": 820, "1": 180}]
    )
    recommended_ml_task: str = Field(
        ...,
        description="Recommended ML objective category (classification, regression, clustering, unknown)",
        examples=["classification"]
    )
    recommendations: List[Recommendation] = Field(
        default=[],
        description="List of calculated modeling suggestions"
    )


class DataIntelligenceResult(BaseModel):
    """Payload representing output parameters returned by the Data Intelligence Agent."""

    dataset_id: str = Field(
        ...,
        description="Reference database ID of the registered dataset",
        examples=["4a4f7a9a-7a9a-4f7a-9a7a-9a7a9a7a9a7a"]
    )
    is_valid: bool = Field(
        ...,
        description="True if validation checks succeeded without blocker failures",
        examples=[True]
    )
    validation_report: ValidationReport = Field(
        ...,
        description="Complete structural validation diagnostic parameters"
    )
    cleaning_report: Optional[CleaningReport] = Field(
        default=None,
        description="Dataset transformations summary details"
    )
    profile: Optional[DatasetProfile] = Field(
        default=None,
        description="Dataset statistical summaries and model suggestions"
    )
    cleaned_filepath: Optional[str] = Field(
        default=None,
        description="Workspace filepath location of the generated clean dataset file",
        examples=["/app/workspace/cleaned/customer_churn.csv"]
    )
    duration_seconds: float = Field(
        ...,
        description="Total duration taken to execute the pipeline operations in seconds",
        examples=[3.41]
    )
