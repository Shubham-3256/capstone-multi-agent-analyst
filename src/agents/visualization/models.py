"""Pydantic data schemas representing generated charts, captions, and export metadata."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ChartCaption(BaseModel):
    """Schema tracking descriptive summaries and explanations of target plots."""

    summary: str = Field(..., description="High-level text summary of the pattern shown", examples=["Strong correlation observed between income and spending."])
    details: str = Field(..., description="Detailed analytical breakdown or secondary insight text", examples=["The correlation coefficient R=0.88 is statistically significant at p<0.01."])


class ChartMetadata(BaseModel):
    """Schema consolidations tracking saved physical plot assets details."""

    chart_id: str = Field(..., description="Unique slug ID of target chart", examples=["correlation_heatmap"])
    title: str = Field(..., description="Chart title string", examples=["Feature Correlation Matrix"])
    chart_type: str = Field(..., description="Chart formatting type category (e.g. heatmap, scatter, bar)", examples=["heatmap"])
    file_path: str = Field(..., description="File path location of saved PNG/SVG chart", examples=["/app/artifacts/plots/correlation_heatmap.png"])
    html_path: Optional[str] = Field(default=None, description="File path location of interactive Plotly HTML if generated", examples=["/app/artifacts/plots/correlation_heatmap.html"])
    caption: ChartCaption = Field(..., description="Auto-generated analytical details description")


class VisualizationMetadata(BaseModel):
    """Schema tracking execution environment details of generated visualizations."""

    dataset_hash: Optional[str] = Field(default=None, description="Sha256 hash reference of the target dataset source")
    created_at: str = Field(..., description="Creation date timestamp iso-string")
    theme: str = Field(..., description="Applied styling design layout theme", examples=["corporate"])


class FigureReference(BaseModel):
    """Schema mapping target chart slugs to JSON-serialized interactive structures."""

    chart_id: str = Field(..., description="Unique slug reference ID of target chart", examples=["correlation_heatmap"])
    plotly_fig_json: Optional[str] = Field(default=None, description="Plotly figure JSON string dictionary for frontend embedding")


class VisualizationReport(BaseModel):
    """Schema compiling lists of generated charts metadata."""

    charts: List[ChartMetadata] = Field(default=[], description="List of generated chart metadata records")
    metadata: Optional[VisualizationMetadata] = Field(default=None, description="Orchestration metadata mapping details")


class VisualizationResult(BaseModel):
    """Payload representing output parameters returned by the Visualization Agent."""

    is_success: bool = Field(..., description="True if agent generated charts successfully")
    report: VisualizationReport = Field(..., description="Compiled metadata report detailing output plots")
    output_directory: str = Field(..., description="Root directory path location where plots are saved")
    duration_seconds: float = Field(..., description="Total execution time in seconds", examples=[3.14])
