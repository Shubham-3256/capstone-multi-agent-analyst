"""Tests for file checkpoint persistence."""

from src.orchestration.checkpoint import FileCheckpointStore
from src.orchestration.state import WorkflowMetadata, WorkflowState


def test_file_checkpoint_round_trip(tmp_path):
    """File checkpoints restore typed workflow state."""
    store = FileCheckpointStore(tmp_path)
    state = WorkflowState(
        dataset_path="data.csv", metadata=WorkflowMetadata(workflow_id="checkpoint")
    )
    store.save(state)
    assert store.load("checkpoint").dataset_path == "data.csv"
