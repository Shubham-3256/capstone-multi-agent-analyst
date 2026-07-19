"""Unit tests for services (FileService, DatasetService, MetadataService)."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.exceptions import ValidationException
from src.database.base import Base
from src.database.models import DatasetRecord
from src.services.file_service import FileService
from src.services.metadata_service import MetadataService


@pytest.fixture
def test_workspace(tmp_path):
    """Fixture providing a temporary folder path representing a mock workspace."""
    return tmp_path


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


def test_file_service_operations(test_workspace):
    """Test moving, copying, renaming, and deleting files inside workspace limits."""
    # Initialize service restricted to test_workspace bounds
    service = FileService(base_dir=test_workspace)

    # 1. Create a dummy file
    source_file = test_workspace / "source.csv"
    source_file.write_text("a,b,c\n1,2,3")

    # 2. Rename
    renamed = service.rename_file(source_file, "renamed.csv")
    assert renamed.exists()
    assert renamed.name == "renamed.csv"
    assert not source_file.exists()

    # 3. Copy
    copy_dest = test_workspace / "copy.csv"
    copied = service.copy_file(renamed, copy_dest)
    assert copied.exists()
    assert renamed.exists()

    # 4. Move
    move_dest = test_workspace / "sub" / "moved.csv"
    moved = service.move_file(copied, move_dest)
    assert moved.exists()
    assert not copy_dest.exists()

    # 5. Delete
    deleted = service.delete_file(moved)
    assert deleted is True
    assert not moved.exists()


def test_file_service_path_traversal_guard(test_workspace):
    """Test that FileService correctly raises ValidationException on traversal attempts."""
    service = FileService(base_dir=test_workspace)

    # Attempting to manipulate a file outside the workspace root (e.g. parent folder)
    outside_file = test_workspace.parent / "secrets.txt"

    with pytest.raises(ValidationException):
        service.delete_file(outside_file)


def test_metadata_service_extraction():
    """Test schema and column profile extraction inside MetadataService."""
    df = pd.DataFrame(
        {
            "age": [25, 42, 19, 25],
            "city": ["New York", "London", "New York", "Tokyo"],
            "joined": pd.to_datetime(
                ["2020-01-01", "2021-06-15", "2019-11-23", "2020-01-01"]
            ),
        }
    )

    metadata = MetadataService.extract_metadata(
        df=df,
        dataset_id="test-dataset-id",
        filename="dummy.csv",
        filepath="/workspace/dummy.csv",
        file_hash="dummy-sha256",
        file_size_bytes=1024,
    )

    assert metadata.dataset_id == "test-dataset-id"
    assert metadata.row_count == 4
    assert metadata.column_count == 3
    assert "age" in metadata.columns
    assert "city" in metadata.columns
    assert "joined" in metadata.columns

    # Check column counts
    assert metadata.columns["age"].non_null_count == 4
    assert metadata.columns["city"].unique_count == 3

    assert "mean" in metadata.columns["age"].statistics
    assert metadata.columns["age"].statistics["mean"] == 27.75


# ==============================================================================
# Presentation / Dashboard Layer Service Tests
# ==============================================================================

from unittest.mock import MagicMock, patch

from app.services.history_service import HistoryService
from app.services.session import (
    clear_session,
    get_uploaded_dataset_path,
    get_workflow_result,
    initialize_session,
    set_uploaded_dataset_path,
    set_workflow_result,
)
from app.services.workflow_service import WorkflowService


class MockSessionState(dict):
    """Mock st.session_state with attribute access support."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def test_session_state_helpers():
    """Test that session-state helpers successfully mutate and query st.session_state."""
    mock_state = MockSessionState()

    with patch("streamlit.session_state", mock_state):
        initialize_session()
        assert "workflow_result" in mock_state
        assert mock_state["uploaded_dataset_path"] is None

        set_workflow_result("result-obj")
        assert get_workflow_result() == "result-obj"

        set_uploaded_dataset_path("some/path.csv")
        assert get_uploaded_dataset_path() == "some/path.csv"

        clear_session()
        assert mock_state["workflow_result"] is None
        assert mock_state["uploaded_dataset_path"] is None


def test_workflow_service_save_upload_unsupported():
    """Test that WorkflowService throws ValueError for unsupported extensions."""
    mock_upload = MagicMock()
    mock_upload.name = "data.txt"

    service = WorkflowService()
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        service.save_upload(mock_upload)


def test_workflow_service_run():
    """Test that WorkflowService calls graph run correctly."""
    mock_graph = MagicMock()
    mock_graph_factory = MagicMock(return_value=mock_graph)

    service = WorkflowService(graph_factory=mock_graph_factory)
    service.run("dummy.csv", target_column="churn")

    mock_graph_factory.assert_called_once()
    mock_graph.run.assert_called_once_with(
        dataset_path="dummy.csv", target_column="churn"
    )


def test_history_service_queries(db_session):
    """Test that HistoryService correctly queries records from the DB session."""

    from src.database.models import ReportRecord, WorkflowExecution

    # Setup test records in db
    w = WorkflowExecution(
        workflow_id="wf-123",
        status="completed",
        history_json="[]",
        errors_json="[]",
        timing_json="{}",
    )
    r = ReportRecord(
        report_id="rep-123",
        title="Report Title",
        template_type="executive",
        dataset_hash="hash-123",
        manifest_json="{}",
        output_paths_json="{}",
        duration_seconds=1.2,
    )
    d = DatasetRecord(
        filename="test.csv",
        filepath="uploads/test.csv",
        file_hash="hash-123",
        file_size_bytes=1000,
        status="uploaded",
    )

    db_session.add_all([w, r, d])
    db_session.commit()

    with patch(
        "src.database.database.DatabaseManager.get_session", return_value=db_session
    ):
        workflows = HistoryService.workflows()
        reports = HistoryService.reports()
        datasets = HistoryService.datasets()

        assert len(workflows) == 1
        assert workflows[0].workflow_id == "wf-123"

        assert len(reports) == 1
        assert reports[0].report_id == "rep-123"

        assert len(datasets) == 1
        assert datasets[0].filename == "test.csv"

        # Test specific record loading
        single_w = HistoryService.workflow_by_id("wf-123")
        assert single_w is not None
        assert single_w.status == "completed"

        single_r = HistoryService.report_by_id("rep-123")
        assert single_r is not None
        assert single_r.title == "Report Title"
