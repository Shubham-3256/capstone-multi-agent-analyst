"""Integration tests verifying ReportGenerationAgent compilation capabilities and document validation."""

import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import ExecutionLog, ReportRecord
from src.agents.report_generation.agent import ReportGenerationAgent
from src.agents.report_generation.context_builder import ContextBuilder
from src.agents.report_generation.validator import ReportValidator
from src.agents.visualization.models import VisualizationResult, VisualizationReport
from src.agents.machine_learning.models import MachineLearningResult, TaskReport, Leaderboard
from src.core.paths import Paths


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_report_generation_lifecycle(clean_db):
    """Test full document pipeline: context compilation, template renders, multiformat export, and validation check."""
    # 1. Setup mock reports context inputs
    dataset_profile = "Completeness: 100%, Shape: (100, 5), Name: Mock Dataset"
    
    ml_result = MachineLearningResult(
        task_report=TaskReport(task_type="classification", classes=[0, 1], is_binary=True),
        best_model_name="RandomForest",
        best_model_path="workspace/artifacts/models/best_model.joblib",
        leaderboard=Leaderboard(entries=[]),
        best_metrics={"accuracy": 0.95, "f1_macro": 0.94},
        feature_importances=[],
        duration_seconds=1.5
    )
    
    viz_result = VisualizationResult(
        report=VisualizationReport(charts=[]),
        is_success=True,
        output_directory="workspace/artifacts/plots",
        duration_seconds=0.5
    )

    agent = ReportGenerationAgent(clean_db)
    
    # 2. Run document generation pipeline
    result = agent.run(
        dataset_profile=dataset_profile,
        ml_result=ml_result,
        visualization_result=viz_result,
        template_type="executive"
    )

    # 3. Assert outputs were written and cataloged
    assert result.is_success is True
    assert "markdown" in result.output_paths
    assert "html" in result.output_paths
    assert "pdf" in result.output_paths
    assert "docx" in result.output_paths

    for fmt, path_str in result.output_paths.items():
        path = Path(path_str)
        assert path.exists()
        assert path.stat().st_size > 0

    # 4. Check DB entries
    report_rec = clean_db.query(ReportRecord).first()
    assert report_rec is not None
    assert report_rec.template_type == "executive"
    assert "markdown" in report_rec.output_paths_json
