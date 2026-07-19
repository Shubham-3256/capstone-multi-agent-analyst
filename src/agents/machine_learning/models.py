"""Data schemas for the Machine Learning Agent outputs."""

from typing import Any

from pydantic import BaseModel, Field


class TaskReport(BaseModel):
    """Schema tracking type detection and binary details of the target variable."""

    task_type: str = Field(..., description="Target modeling task type ('classification' or 'regression')", examples=["classification"])
    classes: list[Any] | None = Field(default=None, description="Unique values of target classes if classification", examples=[[0, 1]])
    is_binary: bool | None = Field(default=None, description="True if classification target contains exactly 2 unique values", examples=[True])


class TrainingResult(BaseModel):
    """Schema tracking fitted hyperparameters, cross-validation limits, and validation runs."""

    model_name: str = Field(..., description="Target candidate algorithm name", examples=["RandomForestClassifier"])
    best_params: dict[str, Any] = Field(default_factory=dict, description="Optimal tuned hyperparameters", examples=[{"n_estimators": 100, "max_depth": 5}])
    cv_score: float = Field(..., description="Mean cross validation score under CV", examples=[0.82])
    duration_seconds: float = Field(..., description="Execution time in seconds", examples=[2.14])
    error_message: str | None = Field(default=None, description="Captured model training errors", examples=[None])


class CrossValidationResult(BaseModel):
    """Schema detailing cross-validation splits scores."""

    fold_scores: list[float] = Field(..., description="Performance score vector for each fold run", examples=[[0.82, 0.84, 0.79, 0.81, 0.83]])
    mean_score: float = Field(..., description="Average CV score", examples=[0.82])
    std_score: float = Field(..., description="Standard deviation of CV scores", examples=[0.02])


class EvaluationReport(BaseModel):
    """Schema consolidating test evaluation metrics and diagnostic shapes."""

    metrics: dict[str, float] = Field(
        ...,
        description="Fitted score parameters (Accuracy, F1, Recall, Precision, ROC-AUC, MAE, RMSE, R²)",
        examples=[{"accuracy": 0.85, "f1": 0.83}]
    )
    confusion_matrix: list[list[int]] | None = Field(
        default=None,
        description="Nested confusion matrix weights if classification",
        examples=[[[120, 15], [20, 95]]]
    )
    residuals_summary: dict[str, float] | None = Field(
        default=None,
        description="Summary statistics of prediction residuals if regression",
        examples=[{"mean": -0.01, "std": 1.25}]
    )


class LeaderboardEntry(BaseModel):
    """Schema mapping a single candidate run entry in the leaderboard."""

    model_name: str = Field(..., description="Name of the model candidate", examples=["RandomForestClassifier"])
    rank: int = Field(..., description="Leaderboard placement rank index", examples=[1])
    score: float = Field(..., description="Primary sort key metric score (F1 or RMSE)", examples=[0.83])
    metrics: dict[str, float] = Field(..., description="Complete metrics evaluated", examples=[{"accuracy": 0.85, "f1": 0.83}])


class Leaderboard(BaseModel):
    """Schema consolidating leaderboard lists."""

    entries: list[LeaderboardEntry] = Field(default=[], description="Ranked list of trained candidate models")


class FeatureImportance(BaseModel):
    """Schema tracking feature names and numeric importance coefficients."""

    column: str = Field(..., description="Feature name", examples=["age"])
    importance: float = Field(..., description="Calculated importance score", examples=[0.42])


class MachineLearningResult(BaseModel):
    """Payload representing output parameters returned by the Machine Learning Agent."""

    task_report: TaskReport = Field(..., description="Inferred modeling task profile")
    best_model_name: str = Field(..., description="Trained candidate algorithm chosen as best", examples=["RandomForestClassifier"])
    best_model_path: str = Field(..., description="Physical path location of the saved best model joblib", examples=["/app/artifacts/models/best_model.joblib"])
    leaderboard: Leaderboard = Field(..., description="Performance leaderboard compiled across candidates")
    best_metrics: dict[str, float] = Field(..., description="Evaluation metrics scores achieved on validation/test sets")
    feature_importances: list[FeatureImportance] = Field(default=[], description="List of feature importances")
    duration_seconds: float = Field(..., description="Total duration of agent modeling sweeps", examples=[12.54])
