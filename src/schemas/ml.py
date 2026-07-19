"""Schemas for Machine Learning task definitions, configurations, and evaluation metrics."""

from typing import Any

from pydantic import BaseModel, Field


class MLTask(BaseModel):
    """Schema defining a Machine Learning analytical objective."""

    task_id: str = Field(
        ...,
        description="Unique identifier for the machine learning task",
        examples=["ml_task_90e3f1ac"],
    )
    task_type: str = Field(
        ...,
        description="Machine learning task class (e.g., classification, regression)",
        examples=["classification"],
    )
    target_column: str = Field(
        ...,
        description="Column target label representing the prediction variable",
        examples=["target"],
    )
    feature_columns: list[str] = Field(
        ...,
        description="List of selected input feature columns",
        examples=[["feature_1", "feature_2", "feature_3"]],
    )


class TrainingConfig(BaseModel):
    """Schema representing model hyperparameter configurations for training pipelines."""

    model_type: str = Field(
        ...,
        description="Target algorithm model class (e.g. xgboost, lightgbm, random_forest)",
        examples=["xgboost"],
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value mapping of model hyperparameters to configure",
        examples=[{"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100}],
    )
    test_size: float = Field(
        default=0.2,
        description="Validation test size fraction split (ranging from 0.05 to 0.95)",
        examples=[0.2],
    )
    random_seed: int = Field(
        default=42,
        description="Random state seed value for deterministic model evaluations",
        examples=[42],
    )


class EvaluationMetrics(BaseModel):
    """Schema encapsulating performance metrics for regression or classification models."""

    # --- Classification Metrics ---
    accuracy: float | None = Field(
        default=None,
        description="Accuracy fraction (classification tasks)",
        examples=[0.875],
    )
    precision: float | None = Field(
        default=None,
        description="Precision rating (classification tasks)",
        examples=[0.812],
    )
    recall: float | None = Field(
        default=None,
        description="Recall rating (classification tasks)",
        examples=[0.764],
    )
    f1_score: float | None = Field(
        default=None,
        description="F1 rating harmonic average (classification tasks)",
        examples=[0.787],
    )
    roc_auc: float | None = Field(
        default=None,
        description="Receiver Operating Characteristic Area Under Curve (classification)",
        examples=[0.923],
    )

    # --- Regression Metrics ---
    mae: float | None = Field(
        default=None,
        description="Mean Absolute Error (regression tasks)",
        examples=[4.21],
    )
    mse: float | None = Field(
        default=None,
        description="Mean Squared Error (regression tasks)",
        examples=[24.56],
    )
    rmse: float | None = Field(
        default=None,
        description="Root Mean Squared Error (regression tasks)",
        examples=[4.95],
    )
    r2_score: float | None = Field(
        default=None,
        description="R-squared coefficient of determination (regression tasks)",
        examples=[0.812],
    )


class TrainingResult(BaseModel):
    """Schema detailing output properties from model training executions."""

    model_id: str = Field(
        ...,
        description="UUID reference of the generated model",
        examples=["model_4f7a-9a7a9a7a9a7a"],
    )
    status: str = Field(
        ...,
        description="Outcome state of the model training (e.g. success, failed)",
        examples=["success"],
    )
    model_path: str = Field(
        ...,
        description="Absolute path to the serialized model file on disk",
        examples=["/app/workspace/models/model_4f7a.joblib"],
    )
    metrics: EvaluationMetrics = Field(
        ..., description="Calculated validation test performance evaluation metrics"
    )
    feature_importances: dict[str, float] | None = Field(
        default=None,
        description="Key-value mapping of feature names to importance scores",
        examples=[{"feature_1": 0.42, "feature_2": 0.38, "feature_3": 0.20}],
    )
