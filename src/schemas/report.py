"""Schemas for report compilation and PDF generation metadata."""

from datetime import datetime

from pydantic import BaseModel, Field


class ReportMetadata(BaseModel):
    """Schema tracking compile profiles and file details for analytical PDF/HTML reports."""

    report_id: str = Field(
        ...,
        description="Unique identifier reference of the generated report",
        examples=["rep_90e3f1ac-7a9a-4f7a-9a7a"],
    )
    filename: str = Field(
        ...,
        description="Name of the generated report output file",
        examples=["analytics_report.pdf"],
    )
    filepath: str = Field(
        ...,
        description="Absolute path to the generated report location in the workspace",
        examples=["/app/workspace/reports/analytics_report.pdf"],
    )
    file_size_bytes: int = Field(
        ...,
        description="File storage size in bytes on the filesystem",
        examples=[412906],
    )
    dataset_id: str = Field(
        ...,
        description="UUID identifier reference of the related source dataset",
        examples=["4a4f7a9a-7a9a-4f7a-9a7a-9a7a9a7a9a7a"],
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Report compilation timestamp"
    )
