"""Strongly typed state and result schemas for workflow orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NodeExecution(BaseModel):
    """Audit record for one graph node execution."""

    node_name: str
    status: str
    duration_seconds: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    retries: int = 0


class WorkflowMetadata(BaseModel):
    """Workflow-level provenance and execution metadata."""

    workflow_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    target_column: str | None = None
    template_type: str = "executive"


from src.agents.business_insights.models import BusinessInsightResult
from src.agents.data_intelligence.models import DataIntelligenceResult, DatasetProfile
from src.agents.feature_engineering.models import FeatureEngineeringResult
from src.agents.machine_learning.models import MachineLearningResult
from src.agents.report_generation.models import ReportResult
from src.agents.visualization.models import VisualizationResult


class WorkflowState(BaseModel):
    """Single typed payload passed between LangGraph nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_path: str
    dataframe: Any | None = None
    data_intelligence_result: Any | None = None
    dataset_profile: Any | None = None
    feature_result: Any | None = None
    ml_result: Any | None = None
    visualization_result: Any | None = None
    business_result: Any | None = None
    report_result: Any | None = None
    execution_history: list[NodeExecution] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)
    metadata: WorkflowMetadata

    @model_validator(mode="before")
    @classmethod
    def convert_dicts_to_models(cls, data: Any) -> Any:
        """Convert serialized dicts inside the state payload back to concrete models if possible."""
        if isinstance(data, dict):
            # 1. data_intelligence_result
            val = data.get("data_intelligence_result")
            if isinstance(val, dict):
                try:
                    data["data_intelligence_result"] = (
                        DataIntelligenceResult.model_validate(val)
                    )
                except Exception:
                    pass
            # 2. dataset_profile
            val = data.get("dataset_profile")
            if isinstance(val, dict):
                try:
                    data["dataset_profile"] = DatasetProfile.model_validate(val)
                except Exception:
                    pass
            # 3. feature_result
            val = data.get("feature_result")
            if isinstance(val, dict):
                try:
                    data["feature_result"] = FeatureEngineeringResult.model_validate(
                        val
                    )
                except Exception:
                    pass
            # 4. ml_result
            val = data.get("ml_result")
            if isinstance(val, dict):
                try:
                    data["ml_result"] = MachineLearningResult.model_validate(val)
                except Exception:
                    pass
            # 5. visualization_result
            val = data.get("visualization_result")
            if isinstance(val, dict):
                try:
                    data["visualization_result"] = VisualizationResult.model_validate(
                        val
                    )
                except Exception:
                    pass
            # 6. business_result
            val = data.get("business_result")
            if isinstance(val, dict):
                try:
                    data["business_result"] = BusinessInsightResult.model_validate(val)
                except Exception:
                    pass
            # 7. report_result
            val = data.get("report_result")
            if isinstance(val, dict):
                try:
                    data["report_result"] = ReportResult.model_validate(val)
                except Exception:
                    pass
        return data


class WorkflowResult(BaseModel):
    """Final orchestration result returned to workflow callers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_success: bool
    state: WorkflowState
    output_paths: dict[str, str] = Field(default_factory=dict)
