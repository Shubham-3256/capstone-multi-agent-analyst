"""Integration tests for the DataIntelligenceAgent pipeline orchestration."""

import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import DatasetRecord, ExecutionLog
from src.agents.data_intelligence.agent import DataIntelligenceAgent


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


from src.core.paths import Paths


@pytest.fixture
def mock_dataset_file():
    """Fixture generating a mock dataset CSV file inside the workspace bounds."""
    temp_dir = Paths.WORKSPACE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    csv_file = temp_dir / "mock_churn.csv"
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "age": [32, None, 45, 23, 38],
        "monthly_charges": [65.5, 80.0, 35.4, 110.25, 70.1],
        "churn": [0, 1, 0, 1, 0]
    })
    df.to_csv(csv_file, index=False)
    yield csv_file
    
    # Cleanup file after execution
    if csv_file.exists():
        csv_file.unlink()


def test_data_intelligence_agent_run(db_session, mock_dataset_file):
    """Test full pipeline (Validate -> Clean -> Profile) and verify DB persistence."""
    agent = DataIntelligenceAgent(db_session)
    
    # Run the pipeline
    result = agent.run(
        file_path=mock_dataset_file,
        target_column="churn",
        imputation_strategies={"age": "median"},
        datatype_conversions={"age": "int"}
    )
    
    # Assert validation & outputs
    assert result.is_valid is True
    assert result.profile is not None
    assert result.profile.recommended_ml_task == "classification"
    assert result.cleaned_filepath is not None
    
    # Check cleaning action
    assert len(result.cleaning_report.transformations) > 0
    age_action = next(t for t in result.cleaning_report.transformations if t.column == "age")
    assert age_action.action_type == "impute_missing"
    
    # Check database persistence
    # 1. Dataset records created (original + clean)
    records = db_session.query(DatasetRecord).all()
    assert len(records) == 2  # mock_churn.csv + clean_mock_churn.csv
    
    # 2. Execution log created
    logs = db_session.query(ExecutionLog).all()
    assert len(logs) == 1
    assert logs[0].task_name == "data_intelligence_pipeline"
    assert logs[0].status == "completed"
