"""Unit tests for DatasetVisualizer configurations."""

import pandas as pd

from src.agents.visualization.dataset_visualizer import DatasetVisualizer


def test_dataset_visualizer_missing_heatmap():
    """Test generating missingness heatmaps for matplotlib and plotly engines."""
    df = pd.DataFrame({"age": [20, None, 40], "income": [50000, 60000, None]})
    figs = DatasetVisualizer.generate_missing_heatmap(df)

    assert "matplotlib" in figs
    assert "plotly" in figs


def test_dataset_visualizer_correlation_heatmap():
    """Test generating numeric correlation matrices plots."""
    df = pd.DataFrame({"age": [20, 30, 40], "income": [50000, 60000, 70000]})
    figs = DatasetVisualizer.generate_correlation_heatmap(df)

    assert "matplotlib" in figs
    assert "plotly" in figs


def test_dataset_visualizer_distribution_plot():
    """Test generating numerical distributions histograms."""
    df = pd.DataFrame({"age": [20, 30, 40], "income": [50000, 60000, 70000]})
    figs = DatasetVisualizer.generate_distribution_plot(df, "age")

    assert "matplotlib" in figs
    assert "plotly" in figs
