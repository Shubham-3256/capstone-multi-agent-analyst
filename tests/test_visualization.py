"""Integration tests for VisualizationAgent runs."""

import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import ExecutionLog
from src.agents.visualization.agent import VisualizationAgent


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


def test_visualization_agent_run(db_session):
    """Test full visualization orchestration (generating missing, correlation, distribution plots & DB log)."""
    agent = VisualizationAgent(db_session)
    
    df = pd.DataFrame({
        "age": [25, 42, 19, 31, 28],
        "income": [50000, 80000, 30000, 60000, 45000],
        "city": ["NY", "London", "NY", "Chicago", "London"]
    })

    result = agent.run(
        dataset_profile=df
    )

    assert result.is_success is True
    assert len(result.report.charts) >= 3
    assert Path(result.report.charts[0].file_path).exists()
    assert Path(result.report.charts[0].html_path).exists()
    
    # Assert database log exists
    logs = db_session.query(ExecutionLog).all()
    assert len(logs) == 1
    assert logs[0].task_name == "visualization_pipeline"
    assert logs[0].status == "completed"
