"""Pydantic data schemas representing generated business insight report sections."""

from pydantic import BaseModel, Field


class ExecutiveSummary(BaseModel):
    """Schema tracking operations optimizations summary findings."""

    headline: str = Field(..., description="High-level punchy business summary statement")
    key_takeaways: list[str] = Field(..., description="List of primary operational summary points")
    impact_statement: str = Field(..., description="Details regarding business value or cost reductions")


class DatasetInsight(BaseModel):
    """Schema tracking data quality anomalies audits and cleaning actions."""

    completeness_score: float = Field(..., description="Value representing completeness ratios")
    anomalies_detected: list[str] = Field(..., description="Issues discovered in data schema formats")
    recommendation: str = Field(..., description="Suggested operations cleaning actions")


class ModelInsight(BaseModel):
    """Schema detailing model performance scores and key predictive drivers."""

    algorithm_name: str = Field(..., description="Best performing algorithm selected")
    accuracy: float = Field(..., description="Accuracy metric score")
    f1: float = Field(..., description="F1-macro metric score")
    feature_weights: dict[str, float] = Field(..., description="Coefficients representing feature importances")
    conclusion: str = Field(..., description="Summary explanation of model validity and error bounds")


class Recommendation(BaseModel):
    """Schema outlining actionable business strategies."""

    title: str = Field(..., description="Actionable recommendation title")
    description: str = Field(..., description="Detailed explanation of the strategy")
    actionability: str = Field(..., description="Actionability rating index ('High', 'Medium', 'Low')")


class RiskItem(BaseModel):
    """Schema auditing operational risk parameters."""

    severity: str = Field(..., description="Severity index ('High', 'Medium', 'Low')")
    probability: str = Field(..., description="Likelihood index ('High', 'Medium', 'Low')")
    description: str = Field(..., description="Operational risk description details")


class ConfidenceReport(BaseModel):
    """Schema auditing overall analysis confidence metrics."""

    confidence_score: float = Field(..., description="Audit confidence score between 0.0 and 1.0")
    reliability_rating: str = Field(..., description="Reliability rating ('High', 'Medium', 'Low')")
    justification: str = Field(..., description="Structured audit reasoning statement")


class BusinessInsightResult(BaseModel):
    """Unified payload representing compiled executive reports returned by BusinessInsightAgent."""

    executive_summary: ExecutiveSummary = Field(..., description="Executive summary overview findings")
    dataset_insight: DatasetInsight = Field(..., description="Data quality validation metrics findings")
    model_insight: ModelInsight = Field(..., description="Model validation accuracy interpretations")
    recommendations: list[Recommendation] = Field(default=[], description="Strategic recommendations checklist")
    risks: list[RiskItem] = Field(default=[], description="Identified operational risk items")
    confidence_report: ConfidenceReport = Field(..., description="Analysis confidence verification report")
    token_usage: dict[str, float] = Field(default_factory=dict, description="Estimated token usage summary count")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
