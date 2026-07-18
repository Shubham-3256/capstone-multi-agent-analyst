"""Data quality assessment prompt builder."""

from src.core.llm.prompts import PromptTemplates


class DataQualityBuilder:
    """Compiles formatting prompts for dataset quality audits."""

    @staticmethod
    def build_prompt(missing_meta: str, corr_meta: str) -> str:
        """Format prompt with missing values and correlation descriptions.

        Args:
            missing_meta: Heatmap captions context.
            corr_meta: Correlation matrix captions context.

        Returns:
            str: Compiled prompt.
        """
        return PromptTemplates.DATA_QUALITY.format(
            missing_meta=missing_meta,
            corr_meta=corr_meta
        )
