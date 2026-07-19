"""Checkpoint utilities for in-memory and file-based workflow recovery."""

import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from src.orchestration.state import WorkflowState


class FileCheckpointStore:
    """Persists resumable workflow state as JSON files outside agent business logic."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: WorkflowState) -> Path:
        """Write the current state snapshot and return its path."""
        path = self.directory / f"{state.metadata.workflow_id}.json"
        # Runtime payloads may include pandas objects or third-party agent results.
        # Preserve a durable audit/resume snapshot without serializing implementation objects.
        payload = state.model_dump(mode="python")
        payload["dataframe"] = None
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return path

    def load(self, workflow_id: str) -> WorkflowState | None:
        """Load a prior snapshot when it exists."""
        path = self.directory / f"{workflow_id}.json"
        return (
            WorkflowState.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )


def memory_checkpointer() -> MemorySaver:
    """Create LangGraph's standard in-memory checkpointer."""
    return MemorySaver()
