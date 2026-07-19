"""Integration tests for MachineLearningAgent execution runs."""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.machine_learning.agent import MachineLearningAgent
from src.database.base import Base
from src.database.models import ExecutionLog


@pytest.fixture
def db_session():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_machine_learning_agent_run(db_session):
    """Test full AutoML pipeline runs: Task detect -> candidate fits -> evaluator -> ranker -> persist -> db log."""
    agent = MachineLearningAgent(db_session)

    # 15 rows of customer training data
    train_df = pd.DataFrame(
        {
            "age": [
                0.1,
                0.5,
                0.2,
                0.9,
                0.3,
                0.4,
                0.8,
                0.6,
                0.3,
                0.7,
                0.5,
                0.2,
                0.1,
                0.8,
                0.4,
            ],
            "charges": [
                0.5,
                0.8,
                0.2,
                0.9,
                0.4,
                0.3,
                0.7,
                0.6,
                0.1,
                0.8,
                0.9,
                0.3,
                0.2,
                0.7,
                0.5,
            ],
        }
    )
    train_y = pd.Series([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0])

    # Validation data (5 rows)
    val_df = pd.DataFrame(
        {"age": [0.2, 0.7, 0.4, 0.1, 0.6], "charges": [0.3, 0.9, 0.5, 0.2, 0.8]}
    )
    val_y = pd.Series([0, 1, 0, 0, 1])

    result = agent.run(
        train_data=train_df,
        train_target=train_y,
        validation_data=val_df,
        validation_target=val_y,
    )

    assert result.best_model_name != ""
    assert Path(result.best_model_path).exists()
    assert len(result.leaderboard.entries) > 0
    assert "accuracy" in result.best_metrics
    assert len(result.feature_importances) > 0

    # Verify SQL logging completes
    logs = db_session.query(ExecutionLog).all()
    assert len(logs) == 1
    assert logs[0].task_name == "machine_learning_pipeline"
    assert logs[0].status == "completed"
