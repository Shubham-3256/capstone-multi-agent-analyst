"""Unit tests for NumericalScaler custom transformer."""

import pandas as pd

from src.agents.feature_engineering.scaler import NumericalScaler


def test_numerical_scaler():
    """Test scaling selections (Standard vs Robust scaling)."""
    df = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],  # No outliers -> Standard/MinMax
            "salary": [50000, 52000, 48000, 51000, 1000000],  # Heavy outlier -> Robust
        }
    )

    scaler = NumericalScaler(outlier_threshold=0.1)
    scaler.fit(df)
    transformed_df = scaler.transform(df)

    # Outlier ratio in salary is 1/5 = 20% (> 10% threshold) -> Robust Scaler
    # Outlier ratio in age is 0% (< 10%) -> Standard Scaler
    assert scaler.scalers_["salary"] == "robust"
    assert scaler.scalers_["age"] in ["standard", "minmax"]
    assert transformed_df.loc[4, "salary"] < 1000.0  # Scaled down from 1,000,000.0
