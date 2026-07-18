"""Tests for strongly typed workflow state schemas."""

from src.orchestration.state import WorkflowMetadata, WorkflowState


def test_workflow_state_has_isolated_mutable_defaults():
    """Each workflow state receives independent lifecycle collections."""
    first = WorkflowState(dataset_path="first.csv", metadata=WorkflowMetadata(workflow_id="one"))
    second = WorkflowState(dataset_path="second.csv", metadata=WorkflowMetadata(workflow_id="two"))
    first.errors.append("failure")
    assert second.errors == []
