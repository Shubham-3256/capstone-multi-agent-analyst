"""Pydantic schemas representing report layouts, contexts, and manifests."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class FigureReference(BaseModel):
    """Schema tracking visual chart placements and descriptions."""

    label: str = Field(..., description="Figure identifier index (e.g. 'Figure 1')")
    file_path: str = Field(..., description="Absolute or relative path to graph image file")
    caption: str = Field(..., description="Fig description context caption")


class TableReference(BaseModel):
    """Schema formatting raw tabular data grids."""

    label: str = Field(..., description="Table identifier index (e.g. 'Table 1')")
    description: str = Field(..., description="Table description context caption")
    headers: List[str] = Field(..., description="Column header labels")
    rows: List[List[Any]] = Field(..., description="Grid cell values matrix")


class ReportSection(BaseModel):
    """Schema documenting a single chapter section in compiled reports."""

    title: str = Field(..., description="Chapter title name")
    section_id: str = Field(..., description="Unique slug key identifier")
    content_markdown: str = Field(..., description="Body section text formatted in markdown")
    figures: List[FigureReference] = Field(default_factory=list, description="Embedded figure references list")
    tables: List[TableReference] = Field(default_factory=list, description="Embedded data table references list")


class ReportContext(BaseModel):
    """Unified layout merging preceding outputs from all pipeline agents."""

    dataset_name: str = Field(default="Analyst Dataset", description="Name descriptor for analyzed data source")
    dataset_profile_str: str = Field(..., description="Overview summaries from Data Intelligence Agent")
    feature_summary_str: str = Field(..., description="Overview summaries from Feature Engineering Agent")
    ml_summary_str: str = Field(..., description="AutoML leaderboard outputs from Machine Learning Agent")
    viz_summary_str: str = Field(..., description="Graph layouts metadata from Visualization Agent")
    insights_str: str = Field(..., description="Corporate summaries and recommendations from Business Insight Agent")
    charts_paths: List[str] = Field(default_factory=list, description="Paths to generated chart visual assets")


class ReportManifest(BaseModel):
    """Schema registering compiled document items, hashes, and charts for provenance audit checks."""

    report_id: str = Field(..., description="Unique UUID report token identifier")
    dataset_hash: str = Field(..., description="SHA256 signature hash of target source data")
    pipeline_version: str = Field(default="1.0.0", description="Code version key")
    model_version: str = Field(default="gpt-4o", description="Underlying core LLM tag")
    charts_included: List[str] = Field(default_factory=list, description="Chart file basenames exported inside document folders")
    sections: List[str] = Field(default_factory=list, description="Ordered list of chapter titles generated")
    generation_timestamp: str = Field(..., description="UTC ISO format creation timestamp")


class ReportMetadata(BaseModel):
    """Schema tracking report catalog records details."""

    title: str = Field(..., description="Report title header")
    author: str = Field(default="Multi-Agent Platform", description="Author signature")
    template_type: str = Field(..., description="Template model selected ('executive', 'technical', 'audit')")
    created_at: str = Field(..., description="Creation date description")


class ReportResult(BaseModel):
    """Unified payload returned by ReportGenerationAgent containing manifest records and output paths."""

    is_success: bool = Field(..., description="Agent run completion validation index")
    output_paths: Dict[str, str] = Field(..., description="Paths mapping exported format keys (markdown, html, pdf, docx)")
    manifest: ReportManifest = Field(..., description="Compiled report manifest index metadata")
    metadata: ReportMetadata = Field(..., description="Report catalog parameters")
    duration_seconds: float = Field(..., description="Total pipeline execution time")
