"""Confidence rating prompt builder."""

from src.core.llm.prompts import PromptTemplates


class ConfidenceBuilder:
    """Compiles formatting prompts for audit confidence assessments."""

    @staticmethod
    def build_prompt(best_metrics: str, dataset_summary: str) -> str:
        """Format prompt with model metrics and dataset profiling summary.

        Args:
            best_metrics: Evaluation performance metrics.
            dataset_summary: Dataset profiling summary overview.

        Returns:
            str: Compiled prompt.
        """
        return PromptTemplates.CONFIDENCE.format(
            best_metrics=best_metrics,
            dataset_summary=dataset_summary
        )
