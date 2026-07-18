"""Unit tests for FeatureSelector custom transformer."""

import pandas as pd
from src.agents.feature_engineering.selector import FeatureSelector


def test_feature_selector():
    """Test selector pruning (low variance and high correlation drops)."""
    df = pd.DataFrame({
        "feature_1": [1.0, 1.0, 1.0, 1.0, 1.001],  # Low variance
        "feature_2": [10, 20, 30, 40, 50],
        "feature_3": [10.1, 20.1, 30.1, 40.1, 50.1]  # Almost perfectly correlated with feature_2
    })

    # Fit correlation filtering method
    selector = FeatureSelector(method="correlation", variance_threshold=0.01, correlation_threshold=0.9)
    selector.fit(df)
    transformed_df = selector.transform(df)

    # feature_1 should be dropped due to low variance
    # feature_3 should be dropped due to high correlation with feature_2
    assert "feature_1" not in transformed_df.columns
    assert "feature_2" in transformed_df.columns
    assert "feature_3" not in transformed_df.columns
