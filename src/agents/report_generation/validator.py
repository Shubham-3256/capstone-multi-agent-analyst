"""Validator verifying file structures, chart connections, and placeholder completions."""

import re
from pathlib import Path

from src.agents.report_generation.models import ReportManifest, ReportSection
from src.core.logger import get_logger

logger = get_logger(__name__)


class ReportValidator:
    """Audits report structures, asset links, and identifies unresolved layout placeholders."""

    @classmethod
    def validate_report(
        cls,
        sections: dict[str, ReportSection],
        manifest: ReportManifest,
        target_dir: Path,
        output_paths: dict[str, str] | None = None,
    ) -> bool:
        """Validate generated document assets, sections content, and manifest schemas.

        Args:
            sections: Dictionary of section models.
            manifest: Compiled manifest structure.
            target_dir: Export parent directory.

        Returns:
            bool: True if validation checks succeed, otherwise False.
        """
        logger.info("ReportValidator: Commencing structural document audits...")
        is_valid = True

        # 1. Verify the required sections exist and contain content.
        missing_sections = cls.REQUIRED_SECTIONS.difference(sections)
        if missing_sections:
            logger.error("ReportValidator: Missing sections: %s", sorted(missing_sections))
            is_valid = False
        for sec_id, section in sections.items():
            if not section.content_markdown.strip():
                logger.warning(f"ReportValidator: Section '{sec_id}' is empty.")
                is_valid = False

        # 2. Verify all manifest listed charts and all section references exist.
        for chart_name in manifest.charts_included:
            chart_file = target_dir / chart_name
            if not chart_file.exists():
                logger.error(f"ReportValidator: Asset reference {chart_file} not found on disk.")
                is_valid = False

        manifest_assets = {Path(chart).name for chart in manifest.charts_included}
        for section in sections.values():
            for figure in section.figures:
                if Path(figure.file_path).name not in manifest_assets:
                    logger.error("ReportValidator: %s is not represented in the manifest.", figure.file_path)
                    is_valid = False

        # 3. Check for unresolved template placeholders.
        placeholder_pattern = re.compile(r"\{\{[a-zA-Z0-9_]+\}\}|\{[a-zA-Z_][a-zA-Z0-9_]*\}")
        for sec_id, section in sections.items():
            content = section.content_markdown
            matches = placeholder_pattern.findall(content)
            if matches:
                logger.warning(f"ReportValidator: Found unresolved template placeholders in '{sec_id}': {matches}")
                is_valid = False

        # 4. Verify each requested export was successfully written.
        if output_paths:
            for output_type, output_path in output_paths.items():
                if not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
                    logger.error("ReportValidator: %s export is missing or empty.", output_type)
                    is_valid = False

        if is_valid:
            logger.info("ReportValidator: Document validation checks passed successfully.")
        else:
            logger.error("ReportValidator: Document validation failures discovered. See warnings.")

        return is_valid
    REQUIRED_SECTIONS = {
        "cover_page", "executive_summary", "dataset_overview", "data_quality",
        "feature_engineering", "machine_learning", "visualizations",
        "business_insights", "recommendations", "risks", "technical_appendix",
        "references",
    }
