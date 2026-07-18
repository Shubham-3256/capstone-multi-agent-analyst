"""Unit tests for CaptionGenerator text explanations."""

import pandas as pd
from src.agents.visualization.caption_generator import CaptionGenerator
from src.agents.machine_learning.models import Leaderboard, LeaderboardEntry, FeatureImportance


def test_caption_generator_missing():
    """Test generating missing values matrix descriptions."""
    df = pd.DataFrame({"age": [20, None, 40]})
    caption = CaptionGenerator.generate_missing_heatmap_caption(df)
    
    assert "missing" in caption.summary
    assert "highest missing" in caption.details


def test_caption_generator_correlation():
    """Test generating correlation matrix descriptions."""
    df = pd.DataFrame({"age": [20, 30, 40], "income": [50000, 60000, 70000]})
    caption = CaptionGenerator.generate_correlation_heatmap_caption(df)
    
    assert "linear association" in caption.summary or "weak correlations" in caption.summary
    assert "Pearson" in caption.details


def test_caption_generator_leaderboard():
    """Test generating leaderboards metric descriptions."""
    leaderboard = Leaderboard(entries=[
        LeaderboardEntry(model_name="RandomForest", rank=1, score=0.91, metrics={"f1": 0.91})
    ])
    caption = CaptionGenerator.generate_leaderboard_caption(leaderboard, "classification")
    
    assert "best performing" in caption.summary
    assert "RandomForest" in caption.summary
    assert "0.91" in caption.details


def test_caption_generator_importance():
    """Test generating feature importance descriptions."""
    importances = [FeatureImportance(column="age", importance=0.85)]
    caption = CaptionGenerator.generate_importance_caption(importances)
    
    assert "predictive value" in caption.summary
    assert "age" in caption.summary
    assert "0.85" in caption.details
