"""Template engine managing layout options and substituting section markdowns."""

from typing import Dict, Any

from src.core.logger import get_logger

logger = get_logger(__name__)


class TemplateEngine:
    """Stores report layouts (Executive, Technical, Research, Audit) and replaces placeholders."""

    _HEADER = """# {report_heading}\n**Title**: {title}\n**Author**: {author}\n**Created**: {created_at}\n\n---\n\n"""
    _ALL_SECTIONS = """{cover_page}\n\n---\n\n{executive_summary}\n\n---\n\n{dataset_overview}\n\n---\n\n{data_quality}\n\n---\n\n{feature_engineering}\n\n---\n\n{machine_learning}\n\n---\n\n{visualizations}\n\n---\n\n{business_insights}\n\n---\n\n{recommendations}\n\n---\n\n{risks}\n\n---\n\n{technical_appendix}\n\n---\n\n{references}\n"""
    TEMPLATES: Dict[str, str] = {
        "executive": _HEADER.replace("{report_heading}", "Executive Business Analytics Report") + _ALL_SECTIONS,
        "technical": _HEADER.replace("{report_heading}", "Technical AutoML Engineering Report") + _ALL_SECTIONS,
        "research": _HEADER.replace("{report_heading}", "Research Analytics Report") + _ALL_SECTIONS,
        "audit": _HEADER.replace("{report_heading}", "Analytics Audit Report") + _ALL_SECTIONS,
    }

    @classmethod
    def render(cls, template_type: str, metadata: Dict[str, str], sections: Dict[str, str]) -> str:
        """Replace template variables and sections placeholders.

        Args:
            template_type: Layout code key ('executive', 'technical').
            metadata: Basic headers parameters (title, author, created_at).
            sections: String mappings of section contents.

        Returns:
            str: Compiled complete markdown report.
        """
        logger.info(f"TemplateEngine: Rendering layout type: '{template_type}'")
        template_key = template_type.lower()
        if template_key not in cls.TEMPLATES:
            raise ValueError(f"Unsupported report template: {template_type}")
        template = cls.TEMPLATES[template_key]
        
        # Merge metadata and sections
        payload = {**metadata, **sections}

        # Replace any missing section keys with a blank string to prevent formatting errors
        for key in ["cover_page", "executive_summary", "dataset_overview", "data_quality",
                    "feature_engineering", "machine_learning", "visualizations", "business_insights",
                    "recommendations", "risks", "technical_appendix", "references"]:
            if key not in payload:
                payload[key] = f"*Section '{key}' omitted by design.*"

        try:
            return template.format(**payload)
        except KeyError as e:
            logger.error(f"TemplateEngine: Key missing during format evaluation: {e}")
            # Safe recovery: replace curly brackets to avoid format failures
            safe_template = template.replace("{", "[").replace("}", "]")
            return safe_template
