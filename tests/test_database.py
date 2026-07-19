"""Tests for the database models, connections, and repository patterns."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import DatasetRecord, ExecutionLog
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.log_repository import ExecutionLogRepository


@pytest.fixture
def db_session():
    """Fixture providing a clean transactional in-memory SQLite database session."""
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


def test_dataset_repository_crud(db_session):
    """Test standard CRUD operations for DatasetRecord inside DatasetRepository."""
    repo = DatasetRepository(db_session)

    # 1. Create
    record = DatasetRecord(
        filename="test.csv",
        filepath="/tmp/test.csv",
        file_hash="dummy_hash_123",
        file_size_bytes=1024,
        status="uploaded",
    )
    saved = repo.create(record)
    db_session.commit()

    assert saved.id is not None
    assert saved.filename == "test.csv"

    # 2. Read (by ID)
    found_by_id = repo.get_by_id(saved.id)
    assert found_by_id is not None
    assert found_by_id.file_hash == "dummy_hash_123"

    # 3. Read (by hash)
    found_by_hash = repo.get_by_hash("dummy_hash_123")
    assert found_by_hash is not None
    assert found_by_hash.id == saved.id

    # 4. Update
    updated = repo.update_status(saved.id, "cleaned")
    db_session.commit()
    assert updated.status == "cleaned"

    # 5. Delete
    deleted = repo.delete(saved.id)
    db_session.commit()
    assert deleted is True

    # Verify non-existence
    assert repo.get_by_id(saved.id) is None


def test_execution_log_repository(db_session):
    """Test CRUD operations for ExecutionLog inside ExecutionLogRepository."""
    repo = ExecutionLogRepository(db_session)

    # 1. Create log
    log = ExecutionLog(
        task_name="profiling",
        agent_name="ProfilerAgent",
        status="running",
        parameters='{"dataset_id": "test_uuid"}',
    )
    saved = repo.create(log)
    db_session.commit()

    assert saved.id is not None
    assert saved.status == "running"

    # 2. Read
    found = repo.get_by_id(saved.id)
    assert found is not None
    assert found.task_name == "profiling"

    # 3. Update status
    updated = repo.update_status(saved.id, "completed", duration_seconds=1.25)
    db_session.commit()
    assert updated.status == "completed"
    assert updated.duration_seconds == 1.25

    # 4. Get by task
    tasks = repo.get_by_task("profiling")
    assert len(tasks) == 1
    assert tasks[0].id == saved.id
