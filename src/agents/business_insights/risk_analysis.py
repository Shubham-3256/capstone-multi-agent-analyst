"""Risk assessment prompt builder."""

from src.core.llm.prompts import PromptTemplates


class RiskAnalysisBuilder:
    """Compiles formatting prompts for operations risks assessments."""

    @staticmethod
    def build_prompt(best_metrics: str, importances: str) -> str:
        """Format prompt with AutoML metrics and feature importances.

        Args:
            best_metrics: Evaluation performance metrics.
            importances: Feature importances.

        Returns:
            str: Compiled prompt.
        """
        return PromptTemplates.RISKS.format(
            best_metrics=best_metrics, importances=importances
        )
