"""Unit tests for CrossValidator folds validations."""

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.agents.machine_learning.cross_validator import CrossValidator


def test_cross_validator_classification():
    """Test StratifiedKFold cross validation evaluation scoring."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    
    model = LogisticRegression()
    result = CrossValidator.evaluate(model, df, y, "classification", cv_folds=2)
    
    assert len(result.fold_scores) == 2
    assert result.mean_score >= 0.0
    assert result.std_score >= 0.0


def test_cross_validator_regression():
    """Test standard KFold cross validation evaluation scoring for regression."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.1, 1.9, 3.2, 3.8])
    
    model = LinearRegression()
    result = CrossValidator.evaluate(model, df, y, "regression", cv_folds=2)
    
    assert len(result.fold_scores) == 2
    assert result.mean_score >= 0.0
