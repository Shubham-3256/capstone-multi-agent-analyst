"""Tests for conditional workflow routing."""

from src.orchestration.router import WorkflowRouter
from src.orchestration.state import WorkflowMetadata, WorkflowState


def test_router_skips_modeling_without_target():
    """No target column routes directly to visualization."""
    state = WorkflowState(dataset_path="data.csv", dataset_profile={"rows": 1}, metadata=WorkflowMetadata(workflow_id="one"))
    assert WorkflowRouter.after_data_intelligence(state) == "visualization"
