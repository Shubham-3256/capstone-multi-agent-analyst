"""Executive Summary prompt builder."""

from src.core.llm.prompts import PromptTemplates


class ExecutiveSummaryBuilder:
    """Compiles formatting prompts for business executive summaries."""

    @staticmethod
    def build_prompt(dataset_summary: str, leaderboard: str) -> str:
        """Format the Executive Summary prompt with context metrics.

        Args:
            dataset_summary: Dataset profiling metrics overview.
            leaderboard: AutoML leaderboard rankings list.

        Returns:
            str: Compiled prompt.
        """
        return PromptTemplates.EXECUTIVE_SUMMARY.format(
            dataset_summary=dataset_summary, leaderboard=leaderboard
        )
