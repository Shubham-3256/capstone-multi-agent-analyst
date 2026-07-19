"""Integration tests focusing on multiclass task classification models, task detection, and metric reports."""

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


def test_multiclass_classification_modeling(clean_db):
    """Verify multiclass classification model loading, metric scores evaluations, and pipeline completion."""
    # 1. Setup multiclass data (3 classes, 30 samples)
    X = pd.DataFrame(
        {
            "feat1": [1.1, 2.5, 3.2, 4.6, 5.0, 6.2, 7.8, 8.1, 9.4, 10.2] * 3,
            "feat2": [0.5, 1.2, 2.3, 1.1, 3.0, 2.5, 4.1, 5.2, 4.8, 5.0] * 3,
        }
    )
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 3)

    # 2. Check ModelFactory loaders
    candidates = ModelFactory.get_candidate_models(
        task_type="classification",
        candidate_names=["random_forest", "extra_trees"],
        random_seed=42,
        n_samples=30,
    )
    assert "random_forest" in candidates
    assert "extra_trees" in candidates

    # 3. Check CrossValidator evaluates macro metrics
    model = candidates["random_forest"]
    cv_res = CrossValidator.evaluate(
        model=model, X=X, y=y, task_type="classification", cv_folds=3
    )
    assert len(cv_res.fold_scores) == 3
    assert cv_res.mean_score >= 0.0

    # 4. Check Evaluator computes multiclass metrics
    model.fit(X, y)
    report = Evaluator.evaluate_model(model, X, y, "classification", is_binary=False)
    assert report.metrics["accuracy"] >= 0.0
    assert report.metrics["precision"] >= 0.0
    assert report.metrics["recall"] >= 0.0
    assert report.metrics["f1"] >= 0.0

    # 5. Check MachineLearningAgent fits and returns Multiclass TaskReport
    agent = MachineLearningAgent(clean_db)
    res = agent.run(
        train_data=X, train_target=y, validation_data=X, validation_target=y
    )
    assert res.best_model_name != ""
    assert res.task_report.is_binary is False
    assert len(res.task_report.classes) == 3
    assert res.task_report.task_type == "classification"
