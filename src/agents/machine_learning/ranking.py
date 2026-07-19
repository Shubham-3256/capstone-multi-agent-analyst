"""Ranking engine to compile performance leaderboards and identify the best candidate."""

from src.agents.machine_learning.models import Leaderboard, LeaderboardEntry
from src.core.logger import get_logger

logger = get_logger(__name__)


class ModelRanker:
    """Ranks estimator models, returning sorted leaderboards and resolving the best model."""

    @staticmethod
    def rank_models(
        candidate_metrics: dict[str, dict[str, float]], task_type: str
    ) -> tuple[Leaderboard, str]:
        """Rank model candidate runs based on task-specific primary keys.

        Args:
            candidate_metrics: Dict mapping model name key to dict of calculated metric scores.
            task_type: Mapped task type ('classification' or 'regression').

        Returns:
            Tuple[Leaderboard, str]: Ranked leaderboard and the name key of the best model.
        """
        logger.info(
            f"ModelRanker: Sorting trained candidates leaderboard (task={task_type.upper()})"
        )

        # Resolve sorting parameters
        # Classification: sort descending on 'f1'
        # Regression: sort ascending on 'rmse'
        is_classification = task_type == "classification"
        primary_metric = "f1" if is_classification else "rmse"
        reverse_sort = is_classification  # True for desc, False for asc

        # Filter candidates and compile list
        entries_to_sort = []
        for model_name, metrics in candidate_metrics.items():
            score = metrics.get(
                primary_metric, 0.0 if is_classification else float("inf")
            )
            entries_to_sort.append((model_name, score, metrics))

        # Sort entries
        entries_to_sort.sort(key=lambda x: x[1], reverse=reverse_sort)

        # Build Leaderboard entries
        leaderboard_entries: list[LeaderboardEntry] = []
        best_model_name = ""

        for idx, (model_name, score, metrics) in enumerate(entries_to_sort):
            rank = idx + 1
            if rank == 1:
                best_model_name = model_name

            leaderboard_entries.append(
                LeaderboardEntry(
                    model_name=model_name, rank=rank, score=score, metrics=metrics
                )
            )

        leaderboard = Leaderboard(entries=leaderboard_entries)
        logger.info(
            f"ModelRanker: Leaderboard created. Ranked best model candidate: '{best_model_name}' (score: {round(entries_to_sort[0][1], 4) if entries_to_sort else 'N/A'})"
        )
        return leaderboard, best_model_name
