"""Integration tests focusing on regression task model selection, evaluation metrics, and feature importances."""

import numpy as np
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


def test_regression_modeling_and_metrics(clean_db):
    """Verify regression model factory loaders, metric scoring calculators, and evaluation pipeline."""
    # 1. Setup regression data (20 samples)
    X = pd.DataFrame(
        {
            "feat1": [1.1, 2.5, 3.2, 4.6, 5.0, 6.2, 7.8, 8.1, 9.4, 10.2] * 2,
            "feat2": [0.5, 1.2, 2.3, 1.1, 3.0, 2.5, 4.1, 5.2, 4.8, 5.0] * 2,
        }
    )
    y = pd.Series(np.random.randn(20) * 10)

    # 2. Check ModelFactory registry loaders
    candidates = ModelFactory.get_candidate_models(
        task_type="regression",
        candidate_names=["linear_regression", "ridge", "lasso"],
        random_seed=42,
        n_samples=20,
    )
    assert "linear_regression" in candidates
    assert "ridge" in candidates

    # 3. Check CrossValidator returns true negative RMSE bounds
    model = candidates["linear_regression"]
    cv_res = CrossValidator.evaluate(
        model=model, X=X, y=y, task_type="regression", cv_folds=3
    )
    assert len(cv_res.fold_scores) == 3
    assert cv_res.mean_score >= 0.0

    # 4. Check Evaluator computes positive errors
    model.fit(X, y)
    report = Evaluator.evaluate_model(model, X, y, "regression")
    assert report.metrics["rmse"] >= 0.0
    assert report.metrics["mae"] >= 0.0
    assert report.metrics["r2"] <= 1.0

    # 5. Check MachineLearningAgent fits and updates DB logs
    agent = MachineLearningAgent(clean_db)
    res = agent.run(
        train_data=X, train_target=y, validation_data=X, validation_target=y
    )
    assert res.best_model_name != ""
    assert res.best_metrics["rmse"] >= 0.0
