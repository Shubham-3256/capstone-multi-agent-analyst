"""Strategic business recommendations prompt builder."""

from src.core.llm.prompts import PromptTemplates


class BusinessRecommendationsBuilder:
    """Compiles formatting prompts for corporate strategy recommendations."""

    @staticmethod
    def build_prompt(best_metrics: str, importances: str) -> str:
        """Format prompt with model metrics and feature importance coefficients.

        Args:
            best_metrics: Evaluation performance metrics.
            importances: Feature importances.

        Returns:
            str: Compiled prompt.
        """
        return PromptTemplates.RECOMMENDATIONS.format(
            best_metrics=best_metrics, importances=importances
        )
