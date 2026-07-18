"""Unit tests for SectionBuilder formatting markdown pages."""

from src.agents.report_generation.models import ReportContext
from src.agents.report_generation.section_builder import SectionBuilder


def test_section_builder_cover_and_executive():
    """Test generating individual chapters cover pages, summaries, risks, and ref sections."""
    context = ReportContext(
        dataset_profile_str="100 columns, highly dense.",
        feature_summary_str="Scaled and Encoded.",
        ml_summary_str="Trained logistic classifier.",
        viz_summary_str="Missing heatmap saved.",
        insights_str="High customer churn."
    )
    
    sections = SectionBuilder.build_sections(context)
    
    assert "cover_page" in sections
    assert "executive_summary" in sections
    assert "risks" in sections
    assert "Analytics Intelligence Report" in sections["cover_page"].content_markdown
