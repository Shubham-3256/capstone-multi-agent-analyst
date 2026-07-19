"""Integration tests focusing on binary classification models, ROC AUC computation, and metric reports."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.machine_learning.agent import MachineLearningAgent
from src.agents.machine_learning.cross_validator import CrossValidator
from src.agents.machine_learning.evaluator import Evaluator
from src.agents.machine_learning.model_factory import ModelFactory
from src.database.base import Base


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_binary_classification_modeling(clean_db):
    """Verify binary classification model instantiation, evaluation metrics, and model run pipelines."""
    # 1. Setup binary class data (20 samples)
    X = pd.DataFrame(
        {
            "feat1": [1.1, 2.5, 3.2, 4.6, 5.0, 6.2, 7.8, 8.1, 9.4, 10.2] * 2,
            "feat2": [0.5, 1.2, 2.3, 1.1, 3.0, 2.5, 4.1, 5.2, 4.8, 5.0] * 2,
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2)

    # 2. Check ModelFactory registry loaders
    candidates = ModelFactory.get_candidate_models(
        task_type="classification",
        candidate_names=["logistic_regression", "random_forest"],
        random_seed=42,
        n_samples=20,
    )
    assert "logistic_regression" in candidates
    assert "random_forest" in candidates

    # 3. Check CrossValidator returns stratified fold splits scores
    model = candidates["logistic_regression"]
    cv_res = CrossValidator.evaluate(
        model=model, X=X, y=y, task_type="classification", cv_folds=3
    )
    assert len(cv_res.fold_scores) == 3
    assert cv_res.mean_score >= 0.0

    model.fit(X, y)
    report = Evaluator.evaluate_model(model, X, y, "classification", is_binary=True)
    assert report.metrics["accuracy"] >= 0.0
    assert report.metrics["precision"] >= 0.0
    assert report.metrics["recall"] >= 0.0
    assert report.metrics["f1"] >= 0.0
    assert report.metrics["roc_auc"] >= 0.0

    # 5. Check MachineLearningAgent fits and returns Classification TaskReport
    agent = MachineLearningAgent(clean_db)
    res = agent.run(
        train_data=X, train_target=y, validation_data=X, validation_target=y
    )
    assert res.best_model_name != ""
    assert res.task_report.is_binary is True
    assert res.task_report.task_type == "classification"
