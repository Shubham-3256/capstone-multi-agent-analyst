"""Integration tests verifying database schemas, ORM model operations, constraints, and referential integrity."""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import DatasetRecord, ExecutionLog, ReportRecord, WorkflowExecution


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


def test_dataset_record_operations(clean_db):
    """Test DatasetRecord model transactions, timestamps, and column structures."""
    # 1. Create a dataset record entry
    record = DatasetRecord(
        filename="test_db.csv",
        filepath="/workspace/uploads/test_db.csv",
        file_hash="uniquehash12345",
        file_size_bytes=1024,
        row_count=100,
        column_count=5,
        status="uploaded"
    )
    clean_db.add(record)
    clean_db.commit()

    # 2. Query and verify fields and timestamps
    retrieved = clean_db.query(DatasetRecord).filter_by(file_hash="uniquehash12345").first()
    assert retrieved is not None
    assert retrieved.filename == "test_db.csv"
    assert retrieved.status == "uploaded"
    # Verify created_at timestamp exists
    assert retrieved.created_at is not None

    # 3. Test uniqueness constraint on file_hash
    duplicate = DatasetRecord(
        filename="duplicate.csv",
        filepath="/workspace/uploads/duplicate.csv",
        file_hash="uniquehash12345",
        file_size_bytes=2048,
        row_count=200,
        column_count=6,
        status="uploaded"
    )
    clean_db.add(duplicate)
    with pytest.raises(IntegrityError):
        clean_db.commit()


def test_execution_log_records(clean_db):
    """Test ExecutionLog record inserts, status updates, and json serialization logic."""
    log = ExecutionLog(
        task_name="profiling",
        agent_name="DataIntelligenceAgent",
        status="running",
        parameters='{"target": "label"}'
    )
    clean_db.add(log)
    clean_db.commit()

    # Verify duration and fields
    retrieved = clean_db.query(ExecutionLog).filter_by(task_name="profiling").first()
    assert retrieved.agent_name == "DataIntelligenceAgent"
    assert retrieved.status == "running"
    
    # Update and serialize
    retrieved.status = "completed"
    retrieved.duration_seconds = 0.5
    clean_db.commit()
    
    log_dict = retrieved.to_dict()
    assert log_dict["status"] == "completed"
    assert log_dict["duration_seconds"] == 0.5


def test_report_and_workflow_execution_records(clean_db):
    """Verify ReportRecord and WorkflowExecution ORM entries mapping."""
    report = ReportRecord(
        report_id="rep-111",
        title="Performance Report",
        template_type="technical",
        dataset_hash="hash123",
        manifest_json='{}',
        output_paths_json='{"pdf": "report.pdf"}',
        duration_seconds=2.5
    )
    clean_db.add(report)
    clean_db.commit()

    wf = WorkflowExecution(
        workflow_id="wf-111",
        status="completed",
        history_json='{}',
        errors_json='[]',
        timing_json='{}'
    )
    clean_db.add(wf)
    clean_db.commit()

    assert clean_db.query(ReportRecord).filter_by(report_id="rep-111").first() is not None
    assert clean_db.query(WorkflowExecution).filter_by(workflow_id="wf-111").first() is not None
