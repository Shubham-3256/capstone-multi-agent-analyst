"""Unit tests for Evaluator score parameters compilations."""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.agents.machine_learning.evaluator import Evaluator


def test_evaluator_classification():
    """Test detailed metric scoring calculations for classification tasks."""
    X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame({"x": [1, 2]})
    y_test = pd.Series([0, 1])

    model = LogisticRegression()
    model.fit(X_train, y_train)

    report = Evaluator.evaluate_model(
        model, X_test, y_test, "classification", is_binary=True
    )

    assert "accuracy" in report.metrics
    assert "f1" in report.metrics
    assert "roc_auc" in report.metrics
    assert report.confusion_matrix is not None


def test_evaluator_regression():
    """Test metric scoring calculations for regression tasks."""
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])
    X_test = pd.DataFrame({"x": [1.5, 2.5]})
    y_test = pd.Series([1.4, 2.6])

    model = LinearRegression()
    model.fit(X_train, y_train)

    report = Evaluator.evaluate_model(model, X_test, y_test, "regression")

    assert "mae" in report.metrics
    assert "rmse" in report.metrics
    assert "r2" in report.metrics
    assert report.residuals_summary is not None
