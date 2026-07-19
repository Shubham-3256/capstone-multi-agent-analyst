"""Unit tests for Exporter layout format output compilations."""

from src.agents.report_generation.exporter import Exporter


def test_exporter_markdown_and_html(tmp_path):
    """Test exporting clean markdown text and converted html files."""
    markdown_content = "# Title Header\nThis is basic test content."

    md_file = tmp_path / "report.md"
    html_file = tmp_path / "report.html"

    Exporter.export_markdown(markdown_content, md_file)
    Exporter.export_html(markdown_content, html_file)

    assert md_file.exists()
    assert html_file.exists()

    # Read HTML content to assert inline CSS block is present
    with open(html_file, encoding="utf-8") as f:
        html_content = f.read()
    assert "<style>" in html_content
    assert "<h1>Title Header</h1>" in html_content


def test_exporter_pdf_and_docx_graceful_run(tmp_path):
    """Test compiling reportlab PDF and python-docx files."""
    markdown_content = "# Title Header\n## Section H2\nThis is basic test content."

    pdf_file = tmp_path / "report.pdf"
    docx_file = tmp_path / "report.docx"

    Exporter.export_pdf(markdown_content, pdf_file)
    Exporter.export_docx(markdown_content, docx_file)

    assert pdf_file.exists()
    assert docx_file.exists()
