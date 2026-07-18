"""Unit tests for Trainer fitting checks."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.agents.machine_learning.trainer import Trainer


def test_trainer_success():
    """Test successful model fit and parameter extraction."""
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    
    model = LogisticRegression()
    result = Trainer.train_model("LR", model, df, y)
    
    assert result.error_message is None
    assert "C" in result.best_params


def test_trainer_error():
    """Test trainer gracefully handles fitting failures."""
    df = pd.DataFrame({"x": ["bad_string", "another"]})
    y = pd.Series([0, 1])
    
    model = LogisticRegression()
    result = Trainer.train_model("LR", model, df, y)
    
    assert result.error_message is not None
    assert "ValueError" in result.error_message or "could not convert" in result.error_message
