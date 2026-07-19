"""Unit tests for FeatureGenerator custom transformer."""

import pandas as pd

from src.agents.feature_engineering.generator import FeatureGenerator


def test_feature_generator():
    """Test interaction feature creations and date segment extractions."""
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "tenure": [2, 5, 8],
            "joined_date": pd.to_datetime(["2020-01-15", "2021-06-20", "2022-12-05"]),
        }
    )

    generator = FeatureGenerator(
        polynomial_degree=2, interaction_only=True, date_features=True
    )
    generator.fit(df)
    transformed_df = generator.transform(df)

    # Verify interaction feature generated
    assert "age_x_tenure" in transformed_df.columns
    assert transformed_df.loc[0, "age_x_tenure"] == 40.0  # 20 * 2

    # Verify datetime extractions
    assert "joined_date_year" in transformed_df.columns
    assert "joined_date_month" in transformed_df.columns
    assert transformed_df.loc[0, "joined_date_year"] == 2020
