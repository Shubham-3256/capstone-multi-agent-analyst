"""Model performance interpretation prompt builder."""

from src.core.llm.prompts import PromptTemplates


class ModelAnalysisBuilder:
    """Compiles formatting prompts for AutoML model metrics explanations."""

    @staticmethod
    def build_prompt(leaderboard: str, best_model_name: str, best_metrics: str) -> str:
        """Format prompt with leaderboard rankings and best estimator metrics.

        Args:
            leaderboard: AutoML leaderboard rankings list.
            best_model_name: Chosen algorithm name.
            best_metrics: Performance evaluation metrics.

        Returns:
            str: Compiled prompt.
        """
        return PromptTemplates.MODEL_ANALYSIS.format(
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            best_metrics=best_metrics,
        )
