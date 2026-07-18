"""Unit tests for Cleaner class transformations."""

import numpy as np
import pandas as pd
from src.agents.data_intelligence.cleaner import Cleaner


def test_cleaner_actions():
    """Test column normalization, trimming, missing value imputations, and outlier caps."""
    df = pd.DataFrame({
        "Age": [25, 42, None, 25, 30, 35, 28],
        "Salary ": [50000.0, 1000000.0, 42000.0, 50000.0, 55000.0, 60000.0, 48000.0],  # Outlier in row 1, trailing space in header
        " City": ["  New York  ", "Chicago ", " Boston", "  New York  ", "Chicago", "London", "Boston"]  # Whitespace-only strings
    })
    
    # 1. Test clean execution
    cleaned_df, report = Cleaner.clean(
        df=df,
        imputation_strategies={"age": "mean"},
        outlier_strategies={"salary": "iqr_cap"},
        datatype_conversions={"age": "int"}
    )
    
    # Assert column normalization
    assert "age" in cleaned_df.columns
    assert "salary" in cleaned_df.columns
    assert "city" in cleaned_df.columns
    
    # Assert duplicates dropped (7 rows -> 6 rows)
    assert cleaned_df.shape[0] == 6
    
    # Assert whitespace trimmed
    assert cleaned_df.loc[0, "city"] == "New York"
    assert cleaned_df.loc[1, "city"] == "Chicago"
    
    # Assert imputation
    assert pd.notnull(cleaned_df.loc[2, "age"])
    # Mean of non-null age is (25 + 42 + 30 + 35 + 28) / 5 = 32
    assert cleaned_df.loc[2, "age"] == 32
    
    # Assert outlier capping
    # IQR limits cap 1000000.0 to upper bound limit
    assert cleaned_df.loc[1, "salary"] < 1000000.0
    assert len(report.transformations) > 0
