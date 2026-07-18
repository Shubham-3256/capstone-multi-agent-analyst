"""Integration tests for ReportGenerationAgent orchestrator runs."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import ExecutionLog, ReportRecord
from src.agents.report_generation.agent import ReportGenerationAgent


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


def test_report_generation_agent_run(db_session):
    """Test full ReportGenerationAgent run pipeline and SQLite database logging."""
    agent = ReportGenerationAgent(db_session)
    
    # Mock profiles and preceding phase metadata strings
    dataset_profile = "Completeness: 100%, 15 rows."
    
    result = agent.run(
        dataset_profile=dataset_profile,
        feature_result=None,
        ml_result=None,
        visualization_result=None,
        business_result=None,
        template_type="executive"
    )

    # Assert output formats exist
    assert "markdown" in result.output_paths
    assert "html" in result.output_paths
    assert "pdf" in result.output_paths
    assert "docx" in result.output_paths
    
    # Assert database logging
    logs = db_session.query(ExecutionLog).all()
    assert len(logs) == 1
    assert logs[0].task_name == "report_generation_pipeline"
    assert logs[0].status == "completed"

    records = db_session.query(ReportRecord).all()
    assert len(records) == 1
    assert records[0].report_id == result.manifest.report_id
    assert "markdown" in records[0].output_paths_json
