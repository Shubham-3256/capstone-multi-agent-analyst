"""Integration tests for BusinessInsightAgent."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import ExecutionLog
from src.agents.feature_engineering.models import FeatureEngineeringResult
from src.agents.machine_learning.models import MachineLearningResult, TaskReport, Leaderboard
from src.agents.visualization.models import VisualizationResult, VisualizationReport
from src.agents.business_insights.agent import BusinessInsightAgent


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


def test_business_insights_agent_run(db_session):
    """Test full business insights agent orchestrator and database logs updates."""
    agent = BusinessInsightAgent(db_session)
    
    # Setup mock preceding result contexts
    dataset_profile = "Completeness: 100%, 15 rows."
    
    ml_result = MachineLearningResult(
        task_report=TaskReport(task_type="classification", classes=[0, 1], is_binary=True),
        best_model_name="RandomForest",
        best_model_path="workspace/artifacts/models/best_model.joblib",
        leaderboard=Leaderboard(entries=[]),
        best_metrics={"accuracy": 0.90},
        feature_importances=[],
        duration_seconds=1.2
    )

    result = agent.run(
        dataset_profile=dataset_profile,
        ml_result=ml_result
    )

    assert result.executive_summary.headline != ""
    assert result.dataset_insight.completeness_score == 1.0
    assert len(result.recommendations) > 0
    assert result.estimated_cost_usd >= 0.0
    
    # Assert database logging
    logs = db_session.query(ExecutionLog).all()
    assert len(logs) == 1
    assert logs[0].task_name == "business_insights_pipeline"
    assert logs[0].status == "completed"
