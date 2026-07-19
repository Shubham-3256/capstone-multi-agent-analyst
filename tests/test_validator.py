"""Unit tests for ReportValidator validation checks."""

from src.agents.report_generation.models import ReportManifest, ReportSection
from src.agents.report_generation.validator import ReportValidator


def test_report_validator_missing_placeholders(tmp_path):
    """Test validating markdown files with unresolved placeholders."""
    sections = {
        "cover_page": ReportSection(
            title="Cover Page",
            section_id="cover_page",
            content_markdown="This has an unresolved template placeholder {{missing_var}}",
        )
    }

    manifest = ReportManifest(
        report_id="test-id",
        dataset_hash="abc",
        generation_timestamp="2026-07-18T12:00:00Z",
    )

    # Assert validation check fails (returns False) due to unresolved placeholder
    is_valid = ReportValidator.validate_report(sections, manifest, tmp_path)
    assert is_valid is False
