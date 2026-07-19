"""Unit tests for Profiler class metrics and recommendation outputs."""

import pandas as pd

from src.agents.data_intelligence.profiler import Profiler


def test_profiler():
    """Test column summaries, Pearson correlations, target profiles, and multicollinearity warnings."""
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "height": [
                150,
                160,
                170,
                180,
            ],  # Perfectly correlated with age (correlation = 1.0)
            "constant_col": [1, 1, 1, 1],  # Constant
            "churn": [0, 0, 0, 1],  # Skewed target variable
        }
    )

    profile = Profiler.profile(df, target_column="churn")

    assert profile.row_count == 4
    assert profile.column_count == 4
    assert profile.recommended_ml_task == "classification"
    assert profile.target_distribution == {"0": 3, "1": 1}

    # Check recommendations
    rec_titles = {rec.title for rec in profile.recommendations}
    assert "Constant column detected" in rec_titles
    assert "Strong multicollinearity" in rec_titles

    # Verify Pearson correlation calculation
    assert profile.correlation_matrix["age"]["height"] == 1.0
