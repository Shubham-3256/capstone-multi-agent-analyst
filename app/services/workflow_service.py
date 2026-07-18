"""Presentation-facing service delegating dataset execution to WorkflowGraph."""

from pathlib import Path
from typing import Any, Callable, Optional

from src.orchestration.events import EventBus, WorkflowEvent
from src.orchestration.graph import WorkflowGraph

from app.services.config import SUPPORTED_EXTENSIONS, UPLOAD_DIR


class WorkflowService:
    """Uploads datasets and invokes the existing orchestration layer."""

    def __init__(self, graph_factory: Callable[..., WorkflowGraph] = WorkflowGraph) -> None:
        self.graph_factory = graph_factory

    @staticmethod
    def save_upload(upload: Any) -> Path:
        """Persist an uploaded supported dataset in the workspace uploads directory."""
        suffix = Path(upload.name).suffix.lower().lstrip(".")
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported dataset type: {suffix}")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = UPLOAD_DIR / Path(upload.name).name
        destination.write_bytes(upload.getbuffer())
        return destination

    def run(self, dataset_path: str, target_column: Optional[str] = None, on_event: Optional[Callable[[WorkflowEvent], None]] = None) -> Any:
        """Run WorkflowGraph and optionally forward lifecycle events to the UI."""
        events = EventBus()
        if on_event:
            events.subscribe(on_event)
        workflow = self.graph_factory(event_bus=events)
        return workflow.run(dataset_path=dataset_path, target_column=target_column)
