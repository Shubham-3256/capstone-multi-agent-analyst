"""Unit tests for FeatureDetector classification checks."""

import pandas as pd

from src.agents.feature_engineering.detector import FeatureDetector


def test_feature_detector():
    """Test feature categories detection (numeric, categorical, constants, identifiers)."""
    df = pd.DataFrame(
        {
            "customer_id": [f"ID{i:03d}" for i in range(1, 101)],  # Identifier
            "age": list(range(1, 101)),  # Numeric
            "city": ["NY", "London", "Chicago", "Tokyo"] * 25,  # Categorical
            "constant_col": [10.0] * 100,  # Constant
            "near_constant": ["A"] * 99 + ["B"],  # Near constant (99% > 95%)
            "target": [0, 1] * 50,
        }
    )

    report = FeatureDetector.detect(df, target_column="target")
    feature_types = report["feature_types"]

    assert feature_types["customer_id"] == "identifier"
    assert feature_types["age"] == "numeric"
    assert feature_types["city"] == "categorical"
    assert "constant_col" in report["constant_columns"]
    assert "near_constant" in report["near_constant_columns"]
