"""Section builder compiling individual markdown chapters from consolidated contexts."""

from pathlib import Path

from src.agents.report_generation.models import (
    FigureReference,
    ReportContext,
    ReportSection,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class SectionBuilder:
    """Builds distinct markdown sections (Cover, Executive Summary, ML, Visualizations, risks, references)."""

    @staticmethod
    def build_sections(context: ReportContext) -> dict[str, ReportSection]:
        """Compile a dictionary of report sections from ReportContext.

        Args:
            context: Consolidated ReportContext.

        Returns:
            Dict[str, ReportSection]: Dict mapping section ID keys to typed ReportSection objects.
        """
        logger.info("SectionBuilder: Building markdown report sections...")
        sections = {}

        # 1. Cover Page
        sections["cover_page"] = ReportSection(
            title="Cover Page",
            section_id="cover_page",
            content_markdown=(
                f"# Analytics Intelligence Report: {context.dataset_name}\n\n"
                f"**System**: Multi-Agent AI Analytics Platform  \n"
                f"**Project**: Capstone Multi-Agent Analyst  \n"
                f"**Security Classification**: CONFIDENTIAL  \n\n"
                f"This document consolidates results from our Data Intelligence, "
                f"Feature Engineering, AutoML Model Search, and Business Insight pipelines."
            ),
        )

        # 2. Executive Summary
        sections["executive_summary"] = ReportSection(
            title="Executive Summary",
            section_id="executive_summary",
            content_markdown=(
                f"## Executive Summary\n\n"
                f"{context.insights_str.split('Business Insights:')[1].strip() if 'Business Insights:' in context.insights_str else context.insights_str}"
            ),
        )

        # 3. Dataset Overview
        sections["dataset_overview"] = ReportSection(
            title="Dataset Overview",
            section_id="dataset_overview",
            content_markdown=(
                f"## Dataset Overview\n\n"
                f"### Base Profile Summary\n"
                f"{context.dataset_profile_str}\n"
            ),
        )

        figures = [
            FigureReference(
                label="",
                file_path=chart_path,
                caption=f"Generated visualization: {Path(chart_path).stem.replace('_', ' ')}.",
            )
            for chart_path in context.charts_paths
        ]

        # 4. Data Quality
        sections["data_quality"] = ReportSection(
            title="Data Quality Assessment",
            section_id="data_quality",
            content_markdown=(
                f"## Data Quality Assessment\n\n"
                f"We audited column completeness, missingness distribution, and collinearity parameters:\n\n"
                f"{context.viz_summary_str}\n"
            ),
        )

        # 5. Feature Engineering
        sections["feature_engineering"] = ReportSection(
            title="Feature Engineering Summary",
            section_id="feature_engineering",
            content_markdown=(
                f"## Feature Engineering & Preprocessing Summary\n\n"
                f"Features were dynamically transformed for optimization prior to model fitting:\n\n"
                f"{context.feature_summary_str}"
            ),
        )

        # 6. Machine Learning
        sections["machine_learning"] = ReportSection(
            title="AutoML Modeling Performance",
            section_id="machine_learning",
            content_markdown=(
                f"## AutoML Modeling Performance\n\n"
                f"Candidate models were trained and ranked based on validation splits performance:\n\n"
                f"{context.ml_summary_str}"
            ),
        )

        # 7. Visualizations
        sections["visualizations"] = ReportSection(
            title="Visualizations Portfolio",
            section_id="visualizations",
            content_markdown=(
                f"## Visualizations Portfolio\n\n"
                f"Below are key performance and distribution charts generated during execution:\n\n"
                f"{context.viz_summary_str}\n"
            ),
            figures=figures,
        )

        # 8. Business Insights
        sections["business_insights"] = ReportSection(
            title="Business Insights",
            section_id="business_insights",
            content_markdown=(
                f"## Strategic Business Insights\n\n"
                f"Based on feature coefficient weights and data profiling findings:\n\n"
                f"{context.insights_str}"
            ),
        )

        # Extract Recommendations from insights_str
        recs_content = ""
        if "Recommendations:" in context.insights_str:
            parts = context.insights_str.split("Recommendations:")
            if len(parts) > 1:
                recs_content = parts[1].split("Risks:")[0].strip()
        if not recs_content:
            recs_content = "1. **Redesign and optimize features**: Continuous profiling of high-importance inputs.\n2. **Review performance boundaries**: Implement standard monitoring to track predictive drift."

        # Extract Risks from insights_str
        risks_content = ""
        if "Risks:" in context.insights_str:
            parts = context.insights_str.split("Risks:")
            if len(parts) > 1:
                risks_content = parts[1].strip()
        if not risks_content:
            risks_content = "- **Feature Drift**: Variations in target variable distribution over time.\n- **Overfitting Risk**: Low sample count limits validation generalization capabilities."

        # 9. Recommendations
        sections["recommendations"] = ReportSection(
            title="Actionable Recommendations",
            section_id="recommendations",
            content_markdown=(f"## Actionable Recommendations\n\n{recs_content}"),
        )

        # 10. Risks
        sections["risks"] = ReportSection(
            title="Operational Risks Assessment",
            section_id="risks",
            content_markdown=(f"## Operational Risks Assessment\n\n{risks_content}"),
        )

        # 11. Technical Appendix
        sections["technical_appendix"] = ReportSection(
            title="Technical Appendix",
            section_id="technical_appendix",
            content_markdown=(
                "## Technical Appendix\n\n"
                "### System Configurations\n"
                "- Python Version: 3.12+  \n"
                "- Database: SQLite (relational logs schema)  \n"
                "- Execution: Scalable multi-agent framework  \n"
            ),
        )

        # 12. References
        sections["references"] = ReportSection(
            title="References",
            section_id="references",
            content_markdown=(
                "## References\n\n"
                "- [1] Capstone Handbooks, Section 8-9 (System Reporting Requirements).\n"
                "- [2] Scikit-Learn Model Selection Documentation (Cross-Validation Standards)."
            ),
        )

        return sections
