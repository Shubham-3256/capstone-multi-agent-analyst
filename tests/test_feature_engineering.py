"""Integration tests for the FeatureEngineeringAgent pipeline orchestration."""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.feature_engineering.agent import FeatureEngineeringAgent
from src.database.base import Base
from src.database.models import DatasetRecord, ExecutionLog


@pytest.fixture
def db_session():
    """Fixture providing a clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_feature_engineering_agent_run(db_session):
    """Test full Feature Engineering pipeline run (Detect -> Leakage -> Split -> Pipeline -> DB Logs)."""
    agent = FeatureEngineeringAgent(db_session)

    df = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
                "C006",
                "C007",
                "C008",
                "C009",
                "C010",
            ],
            "age": [25, 42, 19, 31, 28, 56, 38, 44, 23, 29],
            "city": [
                "NY",
                "London",
                "NY",
                "Chicago",
                "London",
                "NY",
                "London",
                "Chicago",
                "NY",
                "London",
            ],
            "target": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        }
    )

    # Run the feature engineering pipeline
    result = agent.run(dataframe=df, target_column="target")

    assert result.is_success is True
    assert "age" in result.feature_types
    assert result.split_report.train_shape[0] == 7  # 70% of 10 rows
    assert (
        result.split_report.val_shape[0] == 1
    )  # 15% of 10 rows (train_test_split limits)
    assert result.split_report.test_shape[0] == 2  # Remaining test size
    assert Path(result.train_filepath).exists()
    assert Path(result.val_filepath).exists()
    assert Path(result.test_filepath).exists()

    # Assert database logs created
    logs = db_session.query(ExecutionLog).all()
    assert len(logs) == 1
    assert logs[0].task_name == "feature_engineering_pipeline"
    assert logs[0].status == "completed"

    records = db_session.query(DatasetRecord).all()
    assert len(records) > 0
