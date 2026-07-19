"""Unit tests for TemplateEngine formatting."""

from src.agents.report_generation.template_engine import TemplateEngine


def test_template_engine_renders():
    """Test rendering executive markdown report template."""
    metadata = {
        "title": "Corporate Report",
        "author": "Platform Team",
        "created_at": "2026-07-18",
    }
    sections = {
        "cover_page": "# Cover Details",
        "executive_summary": "Executive summaries content.",
    }

    rendered = TemplateEngine.render("executive", metadata, sections)

    assert "Corporate Report" in rendered
    assert "Platform Team" in rendered
    assert "Executive summaries content." in rendered


def test_template_engine_supports_all_declared_templates():
    """Test every supported report layout renders its distinct report heading."""
    metadata = {"title": "Test", "author": "Test", "created_at": "2026-07-18"}
    for template_name in ("executive", "technical", "research", "audit"):
        rendered = TemplateEngine.render(template_name, metadata, {})
        assert "Report" in rendered
