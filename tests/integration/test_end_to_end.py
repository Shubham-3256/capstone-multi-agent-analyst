"""Exhaustive Integration test verifying complete end-to-end data analytics workflow execution."""

import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import ExecutionLog, DatasetRecord, ReportRecord, WorkflowExecution
from src.orchestration.graph import WorkflowGraph
from src.orchestration.config import WorkflowConfig


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


def test_complete_end_to_end_pipeline(clean_db):
    """Verify full workflow runs successfully (Upload -> Data Intel -> FE -> ML -> Viz -> Business -> Report -> History)."""
    # 1. Create a dummy dataset inside workspace
    from src.core.paths import Paths
    temp_dir = Paths.WORKSPACE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = temp_dir / "e2e_dataset.csv"
    
    # 20 samples to ensure robust StratifiedKFold validation splits
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(20)],
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 2,
        "income": [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000] * 2,
        "satisfaction_score": [1.2, 3.4, 2.5, 4.0, 3.1, 4.5, 3.8, 4.9, 2.0, 4.1] * 2,
        "churn": [0, 0, 1, 0, 1, 0, 0, 1, 1, 0] * 2
    })
    df.to_csv(dataset_file, index=False)

    # 2. Instantiate graph running E2E with mock LLM key
    graph = WorkflowGraph(config=WorkflowConfig(checkpoint_mode="none", persist_execution=True))
    graph.db = clean_db
    graph.nodes.callbacks = graph._default_callbacks()

    # 3. Execute graph run
    result = graph.run(str(dataset_file), target_column="churn")

    # 4. Verify E2E Workflow completion
    assert result.is_success is True
    assert result.state.dataset_path != ""
    assert result.state.ml_result is not None
    assert result.state.report_result is not None
    assert result.state.report_result.is_success is True

    # 5. Verify Database Records Logging
    # A. DatasetRecord is registered
    dataset_record = clean_db.query(DatasetRecord).first()
    assert dataset_record is not None
    assert dataset_record.filename == "e2e_dataset.csv"
    assert dataset_record.column_count == 5
    assert dataset_record.row_count == 20

    # B. Execution logs generated for each node
    logs = clean_db.query(ExecutionLog).all()
    assert len(logs) > 0
    task_names = [log.task_name for log in logs]
    assert "data_intelligence_pipeline" in task_names
    assert "feature_engineering_pipeline" in task_names
    assert "machine_learning_pipeline" in task_names
    assert "visualization_pipeline" in task_names
    assert "business_insights_pipeline" in task_names
    assert "report_generation_pipeline" in task_names

    # C. ReportRecord exists
    report_record = clean_db.query(ReportRecord).first()
    assert report_record is not None
    assert report_record.title != ""
    assert "markdown" in report_record.output_paths_json
    assert "pdf" in report_record.output_paths_json

    # D. WorkflowExecution is created
    wf_exec = clean_db.query(WorkflowExecution).first()
    assert wf_exec is not None
    assert wf_exec.status == "completed"
