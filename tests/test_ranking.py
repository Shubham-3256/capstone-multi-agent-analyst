"""Unit tests for ModelRanker leaderboard listings."""

from src.agents.machine_learning.ranking import ModelRanker


def test_ranking_classification():
    """Test classification models sorted in descending order of macro F1 scores."""
    metrics = {
        "RandomForest": {"f1": 0.85, "accuracy": 0.88},
        "LogisticRegression": {"f1": 0.72, "accuracy": 0.76},
        "GradientBoosting": {"f1": 0.91, "accuracy": 0.93},
    }

    leaderboard, best = ModelRanker.rank_models(metrics, "classification")

    assert best == "GradientBoosting"
    assert leaderboard.entries[0].model_name == "GradientBoosting"
    assert leaderboard.entries[1].model_name == "RandomForest"
    assert leaderboard.entries[2].model_name == "LogisticRegression"
    assert leaderboard.entries[0].rank == 1


def test_ranking_regression():
    """Test regression models sorted in ascending order of RMSE values."""
    metrics = {
        "Ridge": {"rmse": 1.25, "r2": 0.81},
        "LinearRegression": {"rmse": 1.48, "r2": 0.75},
        "RandomForest": {"rmse": 0.92, "r2": 0.90},
    }

    leaderboard, best = ModelRanker.rank_models(metrics, "regression")

    assert best == "RandomForest"
    assert leaderboard.entries[0].model_name == "RandomForest"
    assert leaderboard.entries[1].model_name == "Ridge"
    assert leaderboard.entries[2].model_name == "LinearRegression"
