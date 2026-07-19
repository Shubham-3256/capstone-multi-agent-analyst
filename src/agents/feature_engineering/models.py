"""Data schemas for the Feature Engineering Agent outputs."""

from typing import Any

from pydantic import BaseModel, Field


class FeatureInfo(BaseModel):
    """Schema tracking classification and basic metrics for a single feature column."""

    name: str = Field(..., description="Column header of the feature", examples=["age"])
    inferred_type: str = Field(
        ...,
        description="Detected type ('numeric', 'categorical', 'ordinal', 'boolean', 'datetime', 'identifier', 'text')",
        examples=["numeric"],
    )
    missing_count: int = Field(..., description="Missing count of values", examples=[0])
    variance: float = Field(
        ..., description="Calculated sample variance of column", examples=[45.2]
    )
    is_constant: bool = Field(
        ..., description="True if column is constant or near constant", examples=[False]
    )


class EncodingReport(BaseModel):
    """Schema compiling encoding strategies and category category keys."""

    strategy_used: dict[str, str] = Field(
        default_factory=dict,
        description="Mappings of column names to encoder strategy chosen ('onehot', 'ordinal', 'frequency', 'label')",
        examples=[{"city": "onehot", "occupation": "frequency"}],
    )
    mappings: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Fitted mapping lookup values (e.g. for label or ordinal categories)",
        examples=[{"city": {"New York": 0, "Chicago": 1}}],
    )


class ScalingReport(BaseModel):
    """Schema compiling scaling settings and fitted coefficients."""

    scaler_type: dict[str, str] = Field(
        default_factory=dict,
        description="Scaling methods applied to numeric features ('standard', 'robust', 'minmax', 'maxabs', 'none')",
        examples=[{"age": "standard", "salary": "robust"}],
    )
    scaling_parameters: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Fitted scaling parameters (e.g. means, stds, minimums, maximums)",
        examples=[{"age": {"mean": 38.5, "scale": 12.3}}],
    )


class SelectionReport(BaseModel):
    """Schema compiling feature selector configurations and prunings."""

    method: str = Field(
        ..., description="Selection strategy executed", examples=["mutual_info"]
    )
    original_count: int = Field(
        ..., description="Input feature dimensions", examples=[25]
    )
    selected_count: int = Field(
        ..., description="Final output selected dimensions", examples=[10]
    )
    selected_features: list[str] = Field(
        default=[], description="Selected column names", examples=[["age", "salary"]]
    )
    feature_importances: dict[str, float] = Field(
        default_factory=dict,
        description="Calculated feature score or coefficient importance",
        examples=[{"age": 0.42, "salary": 0.35}],
    )


class LeakageReport(BaseModel):
    """Schema summarizing potential target or identifier data leakages."""

    has_leakage: bool = Field(
        ...,
        description="True if any target leak or ID leaks are flagged",
        examples=[False],
    )
    leakage_issues: list[dict[str, str]] = Field(
        default=[],
        description="List of detected leakage warnings and details",
        examples=[
            [
                {
                    "column": "customer_id",
                    "issue": "Identifier column included as numerical input",
                }
            ]
        ],
    )


class SplitReport(BaseModel):
    """Schema compiling data split configurations and shapes."""

    train_shape: list[int] = Field(
        ...,
        description="Dimensions of training set [rows, columns]",
        examples=[[700, 10]],
    )
    val_shape: list[int] = Field(
        ...,
        description="Dimensions of validation set [rows, columns]",
        examples=[[150, 10]],
    )
    test_shape: list[int] = Field(
        ..., description="Dimensions of test set [rows, columns]", examples=[[150, 10]]
    )
    strategy: str = Field(
        ..., description="Splitting technique chosen", examples=["stratified"]
    )


class PipelineReport(BaseModel):
    """Schema tracking preprocessing pipeline files and pipelines components."""

    pipeline_filepath: str = Field(
        ...,
        description="Physical path location of joblib pipeline",
        examples=["/app/workspace/processed/pipeline.joblib"],
    )
    components: list[str] = Field(
        default=[],
        description="Components included in the pipeline",
        examples=[["encoding", "scaling", "selection"]],
    )


class FeatureEngineeringResult(BaseModel):
    """Payload representing output parameters returned by the Feature Engineering Agent."""

    dataset_id: str = Field(
        ...,
        description="Reference database ID of the registered clean/engineered dataset",
        examples=["4a4f7a9a-7a9a-4f7a-9a7a-9a7a9a7a9a7a"],
    )
    is_success: bool = Field(
        ...,
        description="True if pipeline completes without fatal failures",
        examples=[True],
    )
    feature_types: dict[str, str] = Field(
        ...,
        description="Column category types mapped by the detector",
        examples=[{"age": "numeric", "city": "categorical"}],
    )
    encoding_report: EncodingReport = Field(
        ..., description="Summary details of encodings"
    )
    scaling_report: ScalingReport = Field(
        ..., description="Summary details of scalings"
    )
    selection_report: SelectionReport = Field(
        ..., description="Summary details of selections"
    )
    leakage_report: LeakageReport = Field(
        ..., description="Summary details of leakage checks"
    )
    split_report: SplitReport = Field(..., description="Summary details of splits")
    pipeline_report: PipelineReport = Field(
        ..., description="Summary details of serialized pipelines"
    )
    train_filepath: str = Field(
        ...,
        description="Physical path of training segment",
        examples=["/app/workspace/processed/train.csv"],
    )
    val_filepath: str = Field(
        ...,
        description="Physical path of validation segment",
        examples=["/app/workspace/processed/val.csv"],
    )
    test_filepath: str = Field(
        ...,
        description="Physical path of test segment",
        examples=["/app/workspace/processed/test.csv"],
    )
    duration_seconds: float = Field(
        ..., description="Pipeline execution duration in seconds", examples=[3.42]
    )
