"""Unit tests for ModelFactory instantiation logic."""

from src.agents.machine_learning.model_factory import ModelFactory


def test_model_factory_classification():
    """Test loading valid classifier estimators."""
    candidates = ["logistic_regression", "random_forest", "unsupported_alg"]
    models = ModelFactory.get_candidate_models("classification", candidates)

    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "unsupported_alg" not in models

    # Assert correct type class instantiated
    from sklearn.linear_model import LogisticRegression

    assert isinstance(models["logistic_regression"], LogisticRegression)


def test_model_factory_regression():
    """Test loading valid regressor estimators."""
    candidates = ["linear_regression", "ridge"]
    models = ModelFactory.get_candidate_models("regression", candidates)

    assert "linear_regression" in models
    assert "ridge" in models
