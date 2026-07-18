"""Unit tests for ModelVisualizer and DashboardVisualizer."""

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.agents.visualization.model_visualizer import ModelVisualizer
from src.agents.visualization.dashboard_visualizer import DashboardVisualizer
from src.agents.machine_learning.models import FeatureImportance, Leaderboard


def test_model_visualizer_confusion_matrix():
    """Test generating confusion matrix heatmaps."""
    matrix = [[10, 2], [1, 15]]
    classes = ["negative", "positive"]
    figs = ModelVisualizer.generate_confusion_matrix(matrix, classes)
    
    assert "matplotlib" in figs
    assert "plotly" in figs


def test_model_visualizer_roc_curve():
    """Test generating ROC validation curve plots."""
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    
    model = LogisticRegression()
    model.fit(df, y)
    
    figs = ModelVisualizer.generate_roc_curve(model, df, y)
    
    assert "matplotlib" in figs
    assert "plotly" in figs


def test_model_visualizer_residual_plot():
    """Test generating regression residuals scatter plots."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.1, 1.9, 3.2, 3.8])
    
    model = LinearRegression()
    model.fit(df, y)
    
    figs = ModelVisualizer.generate_residual_plot(model, df, y)
    
    assert "matplotlib" in figs
    assert "plotly" in figs


def test_model_visualizer_feature_importance():
    """Test feature importances horizontal charts."""
    importances = [FeatureImportance(column="age", importance=0.8)]
    figs = ModelVisualizer.generate_feature_importance_plot(importances)
    
    assert "matplotlib" in figs
    assert "plotly" in figs


def test_dashboard_visualizer_leaderboard():
    """Test AutoML leaderboard comparison bar charts."""
    leaderboard = Leaderboard(entries=[])
    figs = DashboardVisualizer.generate_leaderboard_chart(leaderboard, "classification")
    
    assert "matplotlib" in figs
    assert "plotly" in figs
