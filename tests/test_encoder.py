"""Unit tests for CategoricalEncoder custom transformer."""

import pandas as pd
from src.agents.feature_engineering.encoder import CategoricalEncoder


def test_categorical_encoder():
    """Test one-hot and frequency encoding strategies."""
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "green"],  # Low cardinality -> One Hot
        "country": ["USA", "UK", "Canada", "Germany"]  # Low cardinality -> One Hot
    })

    encoder = CategoricalEncoder(low_cardinality_threshold=3)
    encoder.fit(df)
    transformed_df = encoder.transform(df)

    # color has 3 unique values -> onehot encoded.
    # country has 4 unique values -> larger than threshold -> ordinal/frequency encoded.
    assert "color_red" in transformed_df.columns
    assert "color_blue" in transformed_df.columns
    assert "color_green" in transformed_df.columns
    assert "color" not in transformed_df.columns
    
    # country is frequency encoded (since uniques=4 > threshold=3)
    assert "country" in transformed_df.columns
    assert transformed_df["country"].dtype != "object"
