"""Integration tests verifying workflow execution resumption, state checkpoints, and recovery."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import WorkflowExecution
from src.orchestration.config import WorkflowConfig
from src.orchestration.graph import WorkflowGraph
from src.orchestration.state import WorkflowState


@pytest.fixture
def clean_db():
    """Fixture providing clean in-memory SQLite database session."""
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


def test_workflow_checkpoint_and_resume(clean_db):
    """Verify that workflow state can be persisted, loaded, and resumed from checkpoints."""
    # 1. Setup small dataset inside workspace
    from src.core.paths import Paths

    temp_dir = Paths.WORKSPACE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = temp_dir / "resume_dataset.csv"

    df = pd.DataFrame(
        {
            "customer_id": [f"ID{i}" for i in range(10)],
            "age": [20, 30, 40, 50, 60] * 2,
            "label": [0, 1, 0, 1, 0] * 2,
        }
    )
    df.to_csv(dataset_file, index=False)

    # 2. Instantiate workflow graph with persist checkpoint settings enabled
    config = WorkflowConfig(checkpoint_mode="file", persist_execution=True)
    graph = WorkflowGraph(config=config)
    graph.db = clean_db
    graph.nodes.callbacks = graph._default_callbacks()

    # 3. Simulate partial run or initial state mapping
    import uuid

    workflow_uuid = f"wf-test-{uuid.uuid4()}"

    from src.orchestration.state import WorkflowMetadata

    state = WorkflowState(
        dataset_path=str(dataset_file),
        metadata=WorkflowMetadata(
            workflow_id=workflow_uuid, target_column="label", template_type="executive"
        ),
    )

    # Save initial execution state checkpoint in database
    execution_record = WorkflowExecution(
        workflow_id=workflow_uuid,
        status="paused",
        history_json=state.model_dump_json(),
        errors_json="[]",
        timing_json="{}",
    )
    clean_db.add(execution_record)
    clean_db.commit()

    # 4. Resume workflow from database execution ID
    resumed_graph = WorkflowGraph(config=config)
    resumed_graph.db = clean_db
    resumed_graph.nodes.callbacks = resumed_graph._default_callbacks()

    # Query and reload the state payload from checkpoint DB
    wf_record = (
        clean_db.query(WorkflowExecution).filter_by(workflow_id=workflow_uuid).first()
    )
    assert wf_record is not None

    # Reload and run workflow to completion starting from the state checkpoint
    result = resumed_graph.run(
        str(dataset_file), target_column="label", workflow_id=workflow_uuid
    )

    assert result.is_success is True
    assert result.state.ml_result is not None
    assert result.state.report_result.is_success is True

    # Check updated execution status in database
    updated_rec = (
        clean_db.query(WorkflowExecution).filter_by(workflow_id=workflow_uuid).first()
    )
    assert updated_rec.status == "completed"
